from typing import Any
import torch
import torch.nn.functional as F

from optim.loss.loss import LOSS_REGISTRY
from optim.loss.instseg_loss import batch_dice_loss, batch_mask_loss

heads = ['ground', 'generation', 'query_cls', 'mv_cls', 'pc_cls', 'voxel_cls', 'txt_cls', 'sem_cls', 'prompt_cls', 'qa']

def cross_entropy(logits, label):
    """ calculate cross entropy along the last dim. """
    logits = torch.clamp(logits, min=-100)
    if label.shape == logits.shape: # label is a 0-1 vector and we use BCE loss.
        logits = logits.view(-1, logits.shape[-1])
        label = label.view(-1, label.shape[-1]).float()
        return F.binary_cross_entropy_with_logits(logits, label)
    else:
        logits = logits.view(-1, logits.shape[-1])
        label = label.view(-1)
        return F.cross_entropy(logits, label)

for head in heads:
    # 'head=head' is the magic to avoid the late-binding issue in lambda functions. Ask ChatGPT about late-binding to learn more.
    loss = lambda cfg, head=head: lambda data_dict: cross_entropy(data_dict[head + '_logits'], data_dict[head + '_label'])
    loss.__name__ = head + '_loss'
    LOSS_REGISTRY.register(loss)
    
def mask_loss(data_dict):
    mask_gt = data_dict['gt_attn_mask'].logical_not()
    instance_labels = data_dict['instance_labels']
    obj_masks = data_dict['obj_masks']
    padding_mask = data_dict['padding_mask']
    total_loss = 0
    for mask_pred, mask_cls in zip(data_dict['predictions_mask'], data_dict['predictions_class']):
        mask_pred = mask_pred.permute(0, 2, 1)
        total_loss += batch_mask_loss(mask_pred, mask_gt.float(), padding_mask) * 5 + batch_dice_loss(mask_pred, mask_gt.float(), padding_mask) * 2
        total_loss += (F.cross_entropy(mask_cls.view(-1, mask_cls.shape[-1]), instance_labels.view(-1), reduction='none') * obj_masks.view(-1)).sum() / (obj_masks.view(-1).sum() + 1e-6) * 2
    return total_loss
    
for head in ['mask']:
    loss = lambda cfg, head=head: lambda data_dict: mask_loss(data_dict)
    loss.__name__ = head + '_loss'
    LOSS_REGISTRY.register(loss)
    