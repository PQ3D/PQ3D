import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from optim.loss.loss import LOSS_REGISTRY

from modules.third_party.mask3d.matcher import HungarianMatcher
from modules.third_party.mask3d.criterion import SetCriterion

@LOSS_REGISTRY.register()
class InstSegLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        loss_cfg = cfg.model.get(self.__class__.__name__)
        self.criterion_type = loss_cfg.get('criterion_type', 'set')
        assert self.criterion_type in ['set', 'direct']

        # training objective
        self.mask_type = "segment_mask"
        # matcher
        matcher = HungarianMatcher(**loss_cfg.matcher)
        # loss weight
        weight_dict = {"loss_ce": matcher.cost_class,
                    "loss_mask": matcher.cost_mask,
                    "loss_dice": matcher.cost_dice}
        aux_weight_dict = {}
        for i in range(len(cfg.model.voxel_encoder.args.hlevels) * cfg.model.unified_encoder.args.num_blocks):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
        self.weight_dict = weight_dict
        # build criterion
        if self.criterion_type == 'set':
            self.set_criterion = SetCriterion(matcher=matcher, weight_dict=weight_dict, **loss_cfg.criterion)
        elif self.criterion_type == 'direct':
            self.direct_criterion = DirectCriterion(**loss_cfg.criterion)
    
    def forward(self, data_dict):
        predictions_class = data_dict['predictions_class']
        predictions_mask = data_dict['predictions_mask']
        # compute loss
        if self.criterion_type == 'direct':
            losses = self.direct_criterion(predictions_mask, predictions_class, data_dict['target_masks'],
                                    data_dict['target_masks_pad_masks'], data_dict['target_labels'])
        else:
            losses, indices = self.set_criterion(predictions_mask, predictions_class, data_dict['instance_labels'], data_dict['segment_masks'])
            data_dict['indices'] = indices
        # 
        for k in list(losses.keys()):
            losses[k] *= self.weight_dict[k]
        
        return [sum(losses.values()), losses]


def batch_dice_loss(logits, targets, padding_mask):
    """Compute the DICE loss, similar to generalized IOU for masks"""
    probs = logits.sigmoid()
    
    # Compute the intersection and union, consider only the pixels that are not padding (mask is True)
    intersection = (probs * targets * padding_mask).sum(dim=-1)
    union = ((probs + targets) * padding_mask).sum(dim=-1)
    
    # Compute Dice loss for each instance
    dice_score = (2. * intersection + 1e-6) / (union + 1e-6)
    dice_loss = 1. - dice_score
    
    # Only consider non-padded instances for the loss
    # Create a mask for instances to consider (where the sum of the padding mask is not 0)
    instance_mask = padding_mask.sum(dim=-1) > 0
    
    # Apply instance mask to the dice loss
    dice_loss[~instance_mask] = 0.
    
    # Calculate mean loss only over non-padding elements
    dice_loss = dice_loss.sum() / instance_mask.sum()
    
    return dice_loss

def batch_mask_loss(logits, targets, padding_mask):
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    loss = (loss * padding_mask).sum(-1)/ (padding_mask.sum(-1) + 1e-6)

    instance_mask = padding_mask.sum(dim=-1) > 0
    loss[~instance_mask] = 0.
    loss = loss.sum() / instance_mask.sum()
    return loss


class DirectCriterion(nn.Module):
    def __init__(self, losses, ignore_label, **kwargs):
        """Create the criterion for using gt mask. Not require hungarian match."""
        super().__init__()
        self.losses = losses
        self.ignore_label = ignore_label

    def loss_labels(self, logits, labels):
        if self.ignore_label != -100:
            labels[labels==self.ignore_label] = -100
        logits = logits.view(-1, logits.shape[-1])
        labels = labels.view(-1)
        loss = F.cross_entropy(logits, labels)
        return {'loss_ce': loss}

    def loss_masks(self, pred_masks, gt_masks, padding_mask):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        pred_masks: B * S * N
        gt_masks: B * N * S
        padding_mask: B * N * S (False for padding pixels and instances)
        """
        pred_masks = pred_masks.permute(0, 2, 1)
        mask_loss = batch_mask_loss(pred_masks, gt_masks, padding_mask)
        dice_loss = batch_dice_loss(pred_masks, gt_masks, padding_mask)
        return {'loss_mask': mask_loss, 'loss_dice': dice_loss}

    def get_loss(self, loss, outputs, targets):
        if loss == 'labels':
            return self.loss_labels(outputs['pred_logits'], targets['labels'])
        if loss == 'masks':
            return self.loss_masks(outputs['pred_masks'], targets['masks'], targets['padding_mask'])

    def forward(self, predictions_mask, predictions_class, target_masks, target_masks_pad_masks, target_labels):
        losses = {}
        targets = {'labels': target_labels, 'masks': target_masks, 'padding_mask': target_masks_pad_masks}
        # loss for last layer
        for loss in self.losses:
            losses.update(self.get_loss(loss, {'pred_logits': predictions_class[-1], 'pred_masks': predictions_mask[-1]}, targets))
        # loss for other layers
            for i in range(len(predictions_mask) - 1):
                for loss in self.losses:
                    l_dict = self.get_loss(loss, {'pred_logits': predictions_class[i], 'pred_masks': predictions_mask[i]}, targets)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses
    