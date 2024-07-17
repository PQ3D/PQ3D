import torch
import torch.nn as nn

from modules.build import GROUNDING_REGISTRY
from modules.utils import layer_repeat
from modules.weights import _init_weights_bert
from modules.utils import get_activation_fn
from modules.layers.transformers import MultiHeadAttentionSpatial


@GROUNDING_REGISTRY.register()
class QueryEncoder(nn.Module):
    def __init__(self, cfg, memories=[], memory_dropout=0.0, hidden_size=768, num_attention_heads=12, num_layers=4,
                share_layer=False, spatial_selfattn=False, structure='sequential', drop_memories_test=[]):
        super().__init__()

        self.spatial_selfattn = spatial_selfattn
        query_encoder_layer = QueryEncoderLayer(hidden_size, num_attention_heads, memories, spatial_selfattn=spatial_selfattn, structure=structure)
        self.unified_encoder = layer_repeat(query_encoder_layer, num_layers, share_layer)

        self.apply(_init_weights_bert)
        self.memory_dropout = memory_dropout
        self.scene_meomories = [x for x in memories if x != 'prompt']
        self.drop_memories_test = drop_memories_test

    def dropout_memory(self, input_dict):
        for memory in self.scene_meomories:
            feat, mask, pos = input_dict[memory]
            if self.training:
                drop_mask = torch.rand(feat.shape[0], device=feat.device) < self.memory_dropout
            elif memory in self.drop_memories_test:
                drop_mask = torch.ones(feat.shape[0], device=feat.device, dtype=torch.bool)
            else:
                drop_mask = torch.zeros(feat.shape[0], device=feat.device, dtype=torch.bool)
            feat[drop_mask] = 0.
            pos[drop_mask] = 0.

    def forward(self, input_dict, pairwise_locs):
        if (self.training and self.memory_dropout > 0) or (not self.training and self.drop_memories_test):
            self.dropout_memory(input_dict)

        query = input_dict['query'][0]
        voxel_feat = input_dict['voxel'][0] if 'voxel' in input_dict.keys() else None
        for i, layer in enumerate(self.unified_encoder):
            if isinstance(voxel_feat, list):
                input_dict['voxel'][0] = voxel_feat[i]  # select voxel features from multi-scale
            query = layer(query, input_dict, pairwise_locs)

        return query

@GROUNDING_REGISTRY.register()
class QueryMaskEncoder(nn.Module):
    def __init__(self, cfg, memories=[], memory_dropout=0.0, hidden_size=768, num_attention_heads=12, num_layers=4,
                share_layer=False, spatial_selfattn=False, structure='sequential', drop_memories_test=[], use_self_mask=False, num_blocks=1):
        super().__init__()

        self.spatial_selfattn = spatial_selfattn
        query_encoder_layer = QueryEncoderLayer(hidden_size, num_attention_heads, memories, spatial_selfattn=spatial_selfattn, structure=structure, memory_dropout=memory_dropout, drop_memories_test=drop_memories_test)
        self.unified_encoder = layer_repeat(query_encoder_layer, num_layers, share_layer)

        self.apply(_init_weights_bert)
        self.memory_dropout = memory_dropout
        self.scene_meomories = [x for x in memories if x != 'prompt']
        self.drop_memories_test = drop_memories_test
        self.use_self_mask = use_self_mask
        self.num_heads = num_attention_heads
        self.num_blocks = num_blocks

    def forward(self, input_dict, pairwise_locs, mask_head=None):
            
        predictions_class, predictions_mask = [], []
        
        query = input_dict['query'][0]
        voxel_feat = input_dict['voxel'][0] if 'voxel' in input_dict.keys() else None

        for block_counter in range(self.num_blocks):
            for i, layer in enumerate(self.unified_encoder):
                if mask_head is not None:
                    output_class, outputs_mask, attn_mask = mask_head(query)
                    predictions_class.append(output_class)
                    predictions_mask.append(outputs_mask)  
                if self.use_self_mask:
                    attn_mask[attn_mask.all(-1)] = False # prevent query to attend to no point
                    attn_mask = attn_mask.repeat_interleave(self.num_heads, 0)
                    for memory in input_dict.keys():
                        if memory in ['query', 'prompt']:
                            continue
                        input_dict[memory][1] = attn_mask
                        
                if isinstance(voxel_feat, list):
                    input_dict['voxel'][0] = voxel_feat[i]  # select voxel features from multi-scale
                query = layer(query, input_dict, pairwise_locs)

        return query, predictions_class, predictions_mask

class QueryEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, memories, dim_feedforward=2048, dropout=0.1, activation="relu", prenorm=False, spatial_selfattn=False, structure='mixed', memory_dropout=0, drop_memories_test=[]):
        super().__init__()
        if spatial_selfattn:
            self.self_attn = SpatialSelfAttentionLayer(d_model, nhead, dropout=dropout, activation=activation, normalize_before=prenorm, batch_first=True)
        else:
            self.self_attn = SelfAttentionLayer(d_model, nhead, dropout=dropout, activation=activation, normalize_before=prenorm, batch_first=True)
        cross_attn_layer = CrossAttentionLayer(d_model, nhead, dropout=dropout, activation=activation, normalize_before=prenorm, batch_first=True) 
        self.cross_attn_list = layer_repeat(cross_attn_layer, len(memories))
        self.memory2ca = {memory:ca for memory, ca in zip(memories, self.cross_attn_list)}
        self.ffn = FFNLayer(d_model, dim_feedforward, dropout=dropout, activation=activation, normalize_before=prenorm)
        self.structure = structure
        self.memories = memories
        self.memory_dropout = memory_dropout
        self.drop_memories_test = drop_memories_test
        if structure == 'gate':
            self.gate_proj = nn.Linear(d_model, d_model)

    def forward(self, query, input_dict, pairwise_locs=None):
        _, query_masks, query_pos = input_dict['query']

        def sequential_ca(query, memories):
            for memory in memories:
                cross_attn = self.memory2ca[memory]
                feat, mask, pos = input_dict[memory] 
                if mask.ndim == 2:
                    memory_key_padding_mask = mask
                    attn_mask = None
                else:
                    memory_key_padding_mask = None
                    attn_mask = mask
                query = cross_attn(tgt=query, memory=feat, attn_mask=attn_mask, memory_key_padding_mask = memory_key_padding_mask, query_pos = query_pos, pos = pos)
            return query

        def parallel_ca(query, memories):
            assert 'prompt' not in memories
            query_list = []
            for memory in memories:
                cross_attn = self.memory2ca[memory]
                feat, mask, pos = input_dict[memory] 
                if mask.ndim == 2:
                    memory_key_padding_mask = mask
                    attn_mask = None
                else:
                    memory_key_padding_mask = None
                    attn_mask = mask
                update = cross_attn(tgt=query, memory=feat, attn_mask=attn_mask, memory_key_padding_mask = memory_key_padding_mask, query_pos = query_pos, pos = pos)
                query_list.append(update)
            # training time memory dropout
            if self.training and self.memory_dropout > 0.0:
                dropout_mask = torch.rand(query.shape[0], len(memories), device=query.device) > self.memory_dropout
                num_remained_memories = dropout_mask.sum(dim=1)
                dropout_mask = torch.logical_or(dropout_mask, num_remained_memories.unsqueeze(-1) == 0)
                num_remained_memories = dropout_mask.sum(dim=1)
                query_tensor = torch.stack(query_list, dim=1)
                query = (query_tensor * dropout_mask.unsqueeze(-1).unsqueeze(-1)).sum(dim=1) / num_remained_memories.unsqueeze(-1).unsqueeze(-1).float()
            else:
                query = torch.stack(query_list, dim=1).mean(dim=1)
            return query
        
        memories = self.memories if self.training else [m for m in self.memories if m not in self.drop_memories_test]
        
        if self.structure == 'sequential':
            query = sequential_ca(query, memories)
        elif self.structure == 'parallel':
            query = parallel_ca(query, memories)
        elif self.structure == 'mixed':
            # [mv,pc,vx] + prompt
            query = parallel_ca(query, [m for m in memories if m != 'prompt'])
            query = sequential_ca(query, ['prompt'])
        elif self.structure == 'gate':
            prompt = sequential_ca(query, ['prompt'])
            gate = torch.sigmoid(self.gate_proj(prompt))
            update = parallel_ca(query, [m for m in self.memories if m != 'prompt'])
            query = (1. - gate) * query + gate * update
        else:
            raise NotImplementedError(f"Unknow structure type: {self.structure}")

        if isinstance(self.self_attn, SpatialSelfAttentionLayer):
            query = self.self_attn(query, tgt_key_padding_mask = query_masks, query_pos = query_pos, 
                                   pairwise_locs = pairwise_locs)
        else:
            query = self.self_attn(query, tgt_key_padding_mask = query_masks, query_pos = query_pos)
        query = self.ffn(query)

        return query


class SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
        batch_first=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self, tgt, attn_mask=None, tgt_key_padding_mask=None, query_pos=None
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt,
            attn_mask=attn_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(
        self, tgt, attn_mask=None, tgt_key_padding_mask=None, query_pos=None
    ):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt2,
            attn_mask=attn_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(
        self, tgt, attn_mask=None, tgt_key_padding_mask=None, query_pos=None
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt, attn_mask, tgt_key_padding_mask, query_pos
            )
        return self.forward_post(
            tgt, attn_mask, tgt_key_padding_mask, query_pos
        )


class CrossAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
        batch_first=False,
    ):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, add_zero_attn=True
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        attn_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=attn_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        attn_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        tgt2 = self.norm(tgt)

        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=attn_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(
        self,
        tgt,
        memory,
        attn_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                attn_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
        return self.forward_post(
            tgt, memory, attn_mask, memory_key_padding_mask, pos, query_pos
        )


class FFNLayer(nn.Module):
    def __init__(
        self,
        d_model,
        dim_feedforward=2048,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)

    
class SpatialSelfAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
        batch_first=False,
        spatial_multihead=True, spatial_dim=5, spatial_attn_fusion='mul'
    ):
        super().__init__()
        self.self_attn = MultiHeadAttentionSpatial(
            d_model, nhead, dropout=dropout,
            spatial_multihead=spatial_multihead,
            spatial_dim=spatial_dim,
            spatial_attn_fusion=spatial_attn_fusion,
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self, tgt, attn_mask=None, tgt_key_padding_mask=None, query_pos=None,
        pairwise_locs=None
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q,
            k,
            tgt,
            key_padding_mask=tgt_key_padding_mask,
            pairwise_locs=pairwise_locs,
        )[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(
        self, tgt, attn_mask=None, tgt_key_padding_mask=None, query_pos=None,
        pairwise_locs=None
    ):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q,
            k,
            tgt,
            key_padding_mask=tgt_key_padding_mask,
            pairwise_locs=pairwise_locs,
        )[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(
        self, tgt, attn_mask=None, tgt_key_padding_mask=None, query_pos=None,
        pairwise_locs=None
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt, attn_mask, tgt_key_padding_mask, query_pos,
                pairwise_locs
            )
        return self.forward_post(
            tgt, attn_mask, tgt_key_padding_mask, query_pos,
            pairwise_locs
        )
