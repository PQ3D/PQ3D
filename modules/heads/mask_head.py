import torch
import torch.nn as nn
import MinkowskiEngine as ME
from MinkowskiEngine.MinkowskiPooling import MinkowskiAvgPooling

from modules.build import HEADS_REGISTRY
from modules.utils import get_mlp_head, layer_repeat


@HEADS_REGISTRY.register()
class MaskHeadSegLevel(nn.Module):
    def __init__(self, cfg, hidden_size, num_targets, memories_for_match=['voxel'], filter_out_classes=None, dropout=0.1):
        super().__init__()

        # cls head
        self.cls_head = get_mlp_head(hidden_size, hidden_size, num_targets, dropout=dropout)
        self.filter_out_classes = filter_out_classes

        # mask head
        memories_for_match = [mem for mem in memories_for_match if mem in ['voxel', 'mv', 'pc']]
        mask_pred_layer = MaskPredictionLayer(hidden_size)
        self.mask_pred_list = layer_repeat(mask_pred_layer, len(memories_for_match))

    def forward(self, query, seg_fts_for_match, seg_masks, offline_attn_masks=None, skip_prediction=False):
        if skip_prediction:
            return None, None, offline_attn_masks
        cls_logits = self.cls_head(query)
        cls_logits[..., self.filter_out_classes] = float("-inf")
        
        mask_logits_list = []
        pad_mask_list = []
        for seg_fts, mask_pred_layer in zip(seg_fts_for_match, self.mask_pred_list):
            feat, mask, pos = seg_fts
            mask_logits = mask_pred_layer(query, feat)
            mask_logits_list.append(mask_logits * mask[..., None].logical_not())
            pad_mask_list.append(mask[..., None].logical_not())
        mask_logits = sum(mask_logits_list) / (sum(pad_mask_list) + 1e-8)
        mask_logits[seg_masks] = -1e6
            
        if offline_attn_masks is not None:
            attn_mask = offline_attn_masks
        else:
            attn_mask = mask_logits.sigmoid().permute(0, 2, 1).detach() < 0.5
        return cls_logits, mask_logits, attn_mask

class MaskPredictionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size, False)
    
    def forward(self, query, key):
        query = self.q_proj(query)
        key = self.k_proj(key)
        logits = torch.einsum('bld, bmd -> blm', key, query)
        return logits
