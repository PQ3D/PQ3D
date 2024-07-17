import os
import json
import einops
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.utils import get_mlp_head, get_mixup_function
from modules.weights import _init_weights_bert
from modules.layers.pointnet import PointNetPP
from modules.build import VISION_REGISTRY


@VISION_REGISTRY.register()
class ObjectEncoder(nn.Module):
    def __init__(self, cfg, backbone='none', input_feat_size=768, hidden_size=768, freeze_backbone=False, use_projection=False,
                 tgt_cls_num=607, pretrained=None, dropout=0.1, use_cls_head=True):
        super().__init__()
        self.cfg = cfg
        self.freeze_backbone = freeze_backbone
        self.context = torch.no_grad if freeze_backbone else nullcontext
        if backbone == 'pointnet++':
            self.backbone = PointNetPP(
                sa_n_points=[32, 16, None],
                sa_n_samples=[32, 32, None],
                sa_radii=[0.2, 0.4, None],
                sa_mlps=[[3, 64, 64, 128], [128, 128, 128, 256], [256, 256, 512, 768]],
            )
        if use_cls_head:
            self.cls_head = get_mlp_head(input_feat_size, input_feat_size // 2, tgt_cls_num, dropout=0.3)

        self.use_projection = use_projection
        if use_projection:
            self.input_feat_proj = nn.Sequential(nn.Linear(input_feat_size, hidden_size), nn.LayerNorm(hidden_size))
        else:
            assert input_feat_size == hidden_size, "input_feat_size should be equal to hidden_size!"
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)

        # load weights
        self.apply(_init_weights_bert)
        if pretrained:
            print("load pretrained weights from {}".format(pretrained))
            pre_state_dict = torch.load(pretrained)
            state_dict = {}
            for k, v in pre_state_dict.items():
                if k[0] in ['0', '2', '4']: # key mapping for voxel
                    k = 'cls_head.' + k
                k = k.replace('vision_encoder.vis_cls_head.', 'cls_head.') # key mapping for mv
                k = k.replace('point_cls_head.', 'cls_head.') # key mapping for pc 
                k = k.replace('point_feature_extractor.', 'backbone.')
                state_dict[k] = v
            warning = self.load_state_dict(state_dict, strict=False)
            print(warning)

    def freeze_bn(self, m):
        for layer in m.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, obj_feats, data_dict=None, **kwargs):
        if self.freeze_backbone and hasattr(self, 'backbone'):
            self.freeze_bn(self.backbone)

        batch_size, num_objs  = obj_feats.shape[:2]
        with self.context():
            if hasattr(self, 'backbone'):
                obj_feats = self.backbone(einops.rearrange(obj_feats, 'b o p d -> (b o) p d'))
                obj_feats = einops.rearrange(obj_feats, '(b o) d -> b o d', b=batch_size)

        obj_embeds = self.input_feat_proj(obj_feats) if self.use_projection else obj_feats
        if hasattr(self, 'dropout'):
            obj_embeds = self.dropout(obj_embeds)

        if hasattr(self, 'cls_head'):
            obj_cls_logits = self.cls_head(obj_feats)
            return obj_embeds, obj_cls_logits
        else:
            return obj_embeds


@VISION_REGISTRY.register()
class SemanticEncoder(nn.Module):
    def __init__(self, cfg, hidden_size=768, use_matmul_label=False, mixup_strategy=None, 
                 mixup_stage1=None, mixup_stage2=None, load_path=None, embed_type='glove'):
        super().__init__()
        if embed_type == 'clip':
            semantic_embedding = torch.load(os.path.join(load_path, "scannetv2_raw_categories_clip_embeds.pt"))
        elif embed_type == 'glove':
            categories = json.load(open(os.path.join(load_path, "scannetv2_raw_categories.json"), 'r'))
            cat2vec = json.load(open(os.path.join(load_path, "cat2glove42b.json"), 'r'))
            semantic_embedding = torch.stack([torch.Tensor(cat2vec[c]) for c in categories])
        else:
            assert False, "embed_type should be 'clip' or 'glove'!"
        self.register_buffer("semantic_embedding", semantic_embedding)
        self.sem_emb_proj = nn.Sequential(nn.Linear(semantic_embedding.shape[1], hidden_size),
                                                 nn.LayerNorm(hidden_size),
                                                 nn.Dropout(0.1))

        # build mixup strategy
        self.mixup_strategy = mixup_strategy
        self.mixup_function = get_mixup_function(mixup_strategy, mixup_stage1, mixup_stage2)
        self.use_matmul_label = use_matmul_label
        
    def forward(self, cls_logits_list, obj_labels, cur_step, max_steps):
        obj_cls_logits = sum(cls_logits_list).div_(len(cls_logits_list))
        obj_sem_cls = F.softmax(obj_cls_logits, dim=2).detach()
        obj_sem_cls_mix = self.mixup_function(obj_sem_cls, obj_labels, cur_step, max_steps) if self.mixup_strategy else obj_sem_cls
        
        if self.use_matmul_label:
            obj_cls_embeds = torch.matmul(obj_sem_cls_mix, self.semantic_embeddings)  # N, O, 607 matmul ,607, 300
        else:
            obj_sem_cls_mix = torch.argmax(obj_sem_cls_mix, dim=2)
            obj_cls_embeds = self.semantic_embedding[obj_sem_cls_mix]
        obj_cls_embeds = self.sem_emb_proj(obj_cls_embeds)

        return obj_cls_embeds, obj_cls_logits
