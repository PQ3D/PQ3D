import torch
from torch import nn

from modules.build import VISION_REGISTRY
from modules.utils import get_mlp_head
from modules.layers.pointnet import PointNetPP


@VISION_REGISTRY.register()
class PcdObjEncoder(nn.Module):
    def __init__(self, cfg, path=None, freeze=False):
        super().__init__()

        self.pcd_net = PointNetPP(
            sa_n_points=[32, 16, None],
            sa_n_samples=[32, 32, None],
            sa_radii=[0.2, 0.4, None],
            sa_mlps=[[3, 64, 64, 128], [128, 128, 128, 256], [256, 256, 512, 768]],
        )

        self.obj3d_clf_pre_head = get_mlp_head(768, 384, 607, dropout=0.3)

        self.dropout = nn.Dropout(0.1)

        if path is not None:
            self.load_state_dict(torch.load(path))

        self.freeze = freeze
        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def freeze_bn(self, m):
        '''Freeze BatchNorm Layers'''
        for layer in m.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, obj_pcds, obj_locs, obj_masks, obj_sem_masks, **kwargs):
        if self.freeze:
            self.freeze_bn(self.pcd_net)

        batch_size, num_objs, _, _ = obj_pcds.size()
        # obj_embeds = self.pcd_net(
        #     einops.rearrange(obj_pcds, 'b o p d -> (b o) p d')
        # )
        # obj_embeds = einops.rearrange(obj_embeds, '(b o) d -> b o d', b=batch_size)

        # TODO: due to the implementation of PointNetPP, this way consumes less GPU memory
        obj_embeds = []
        for i in range(batch_size):
            obj_embeds.append(self.pcd_net(obj_pcds[i]))
        obj_embeds = torch.stack(obj_embeds, 0)
        obj_embeds = self.dropout(obj_embeds)
        # freeze
        if self.freeze:
            obj_embeds = obj_embeds.detach()
        # sem logits
        obj_sem_cls = self.obj3d_clf_pre_head(obj_embeds)
        return obj_embeds, obj_embeds, obj_sem_cls



