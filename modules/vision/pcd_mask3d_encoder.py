from contextlib import nullcontext
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.nn import functional as F
from torch_scatter import scatter_max, scatter_mean, scatter_min

import MinkowskiEngine as ME
from MinkowskiEngine.MinkowskiPooling import MinkowskiAvgPooling, MinkowskiMaxPooling

import modules.third_party.mask3d as mask3d_models
from modules.third_party.mask3d.common import conv
from modules.third_party.mask3d.helpers_3detr import GenericMLP
from modules.third_party.mask3d.position_embedding import PositionEmbeddingCoordsSine
from modules.third_party.pointnet2.pointnet2_utils import furthest_point_sample

from modules.build import VISION_REGISTRY

@VISION_REGISTRY.register()
class PCDMask3DEncoder(nn.Module):
    def __init__(self, cfg, backbone_kwargs, query_dim, hlevels):
        super().__init__()
        self.backbone = getattr(mask3d_models, "Res16UNet34C")(**backbone_kwargs)
        self.mask_features_head = conv(
            self.backbone.PLANES[7], query_dim, kernel_size=1, stride=1, bias=True, D=3
        )
        self.pooling = MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=3)
        self.sizes = self.backbone.PLANES[-5:]
        self.scatter_fn = scatter_mean
        self.hlevels = hlevels
        self.lin_squeeze = nn.ModuleList()
        tmp_squeeze_attention = nn.ModuleList()
        for i, hlevel in enumerate(hlevels):
            tmp_squeeze_attention.append(nn.Linear(self.sizes[hlevel], query_dim))
        self.lin_squeeze.append(tmp_squeeze_attention)
    
    def forward(self, x, voxel_features, point2segment):
        # minkowski backbone
        pcds_features, aux = self.backbone(x)
        # mask features
        mask_features = self.mask_features_head(pcds_features)
        # mask segments
        mask_segments = []
        for i, mask_feature in enumerate(mask_features.decomposed_features):
            mask_segments.append(self.scatter_fn(mask_feature, point2segment[i], dim=0))
        # get coordinates
        with torch.no_grad():
            coordinates = ME.SparseTensor(features=voxel_features[:, -3:], coordinate_manager=aux[-1].coordinate_manager, coordinate_map_key=aux[-1].coordinate_map_key, device=aux[-1].device)
            coords = [coordinates]
            for _ in reversed(range(len(aux)-1)):
                coords.append(self.pooling(coords[-1]))
            coords.reverse()
        # select multi-scale features and coordinates
        multi_scale_features = []
        multi_scale_coordinates = []
        for i, hlevel in enumerate(self.hlevels):
            decomposed_aux = aux[hlevel].decomposed_features
            multi_scale_features.append([self.lin_squeeze[0][i](f) for f in decomposed_aux])
            multi_scale_coordinates.append(coords[hlevel])
        return mask_features, mask_segments, coordinates, multi_scale_features, multi_scale_coordinates 
        

@VISION_REGISTRY.register()
class PCDMask3DSwin3DEncoder(nn.Module):
    def __init__(self, cfg, backbone_kwargs, query_dim, hlevels):
        super().__init__()
        self.backbone = getattr(mask3d_models, "Swin3DUNet")(**backbone_kwargs)
        self.mask_features_head = conv(
            self.backbone.channels[0], query_dim, kernel_size=1, stride=1, bias=True, D=3
        )
        self.pooling = MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=3)
        self.sizes = self.backbone.channels[::-1]
        self.scatter_fn = scatter_mean
        self.hlevels = hlevels
        self.lin_squeeze = nn.ModuleList()
        tmp_squeeze_attention = nn.ModuleList()
        for i, hlevel in enumerate(hlevels):
            tmp_squeeze_attention.append(nn.Linear(self.sizes[hlevel], query_dim))
        self.lin_squeeze.append(tmp_squeeze_attention)

    def forward(self, sp, coordinate_sp, voxel_features, point2segment, batch_indices):
        # Swin3D backbone
        # pcds_features, aux = self.backbone(sp, coordinate_sp)
        mask_features, aux = self.backbone(sp, coordinate_sp)

        # mask features
        # mask_features = self.mask_features_head(pcds_features)

        # mask segments
        mask_segments = []
        for i, mask_feature in enumerate(mask_features.decomposed_features):
            mask_segments.append(self.scatter_fn(mask_feature, point2segment[i], dim=0))
        # for i in range(len(point2segment)):
        #     mask_feature = mask_features[torch.where(batch_indices == i)]
        #     mask_segments.append(self.scatter_fn(mask_feature, point2segment[i], dim=0))

        # get coordinates
        with torch.no_grad():
            coordinates = ME.SparseTensor(features=voxel_features[:, -3:], coordinate_manager=aux[-1].coordinate_manager, coordinate_map_key=aux[-1].coordinate_map_key, device=aux[-1].device)
            coords = [coordinates]
            for _ in reversed(range(len(aux)-1)):
                coords.append(self.pooling(coords[-1]))
            coords.reverse()
        # select multi-scale features and coordinates
        multi_scale_features = []
        multi_scale_coordinates = []
        for i, hlevel in enumerate(self.hlevels):
            decomposed_aux = aux[hlevel].decomposed_features
            multi_scale_features.append([self.lin_squeeze[0][i](f) for f in decomposed_aux])
            multi_scale_coordinates.append(coords[hlevel])
        return mask_features, mask_segments, coordinates, multi_scale_features, multi_scale_coordinates


@VISION_REGISTRY.register()
class PCDMask3DSegLevelEncoder(nn.Module):
    def __init__(self, cfg, backbone_kwargs, hidden_size, hlevels, freeze_backbone=False, dropout=0.1):
        super().__init__()
        # free backbone or not
        self.context = torch.no_grad if freeze_backbone else nullcontext
        self.backbone = getattr(mask3d_models, "Res16UNet34C")(**backbone_kwargs)
        self.scatter_fn = scatter_mean
        self.sizes = self.backbone.PLANES[-5:]
        self.hlevels = hlevels + [4] # 4 is for the last level, always used for mask seg features
        self.feat_proj_list = nn.ModuleList([
                                    nn.Sequential(
                                        nn.Linear(self.sizes[hlevel], hidden_size), 
                                        nn.LayerNorm(hidden_size),
                                        nn.Dropout(dropout)
                                    ) for hlevel in self.hlevels])
        self.pooltr = ME.MinkowskiPoolingTranspose(kernel_size=2, stride=2, dilation=1, dimension=3)

    def upsampling(self, feat, hlevel):
        n_pooltr = 4 - hlevel # 4 levels in totoal
        for _ in range(n_pooltr):
            feat = self.pooltr(feat)
        return feat
            
    def forward(self, x, point2segment, max_seg):
        with self.context():
            # minkowski backbone
            pcds_features, aux = self.backbone(x)

        multi_scale_seg_feats = []
        for hlevel, feat_proj in zip(self.hlevels, self.feat_proj_list):
            feat = aux[hlevel]
            feat = self.upsampling(feat, hlevel)
            assert feat.shape[0] == pcds_features.shape[0]
            batch_feat = [self.scatter_fn(f, p2s, dim=0, dim_size=max_seg) for f, p2s in zip(feat.decomposed_features, point2segment)]
            batch_feat = torch.stack(batch_feat)
            batch_feat = feat_proj(batch_feat)
            multi_scale_seg_feats.append(batch_feat)
        
        return multi_scale_seg_feats
