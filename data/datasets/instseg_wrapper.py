import numpy as np
import torch
from torch.utils.data import Dataset, default_collate
from transformers import AutoTokenizer
import MinkowskiEngine as ME

from data.datasets.constant import PromptType

from .dataset_wrapper import DATASETWRAPPER_REGISTRY
from ..data_utils import pad_sequence, pad_sequence_2d


@DATASETWRAPPER_REGISTRY.register()
class InstSegDatasetWrapper(Dataset):
    def __init__(self, cfg, dataset) -> None:
        super().__init__()
        self.cfg = cfg
        self.dataset = dataset
        self.offline_mask_source = self.dataset.offline_mask_source
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def collate_fn(self, batch):
        new_batch = {}

        # sparse collate voxel features
        input_dict = {
            "coords": [sample.pop('voxel_coordinates') for sample in batch], 
            "feats": [sample.pop('voxel_features') for sample in batch],
        }
        voxel_coordinates, voxel_features = ME.utils.sparse_collate(**input_dict, dtype=torch.int32)
        new_batch['voxel_coordinates'] = voxel_coordinates
        new_batch['voxel_features'] = voxel_features
        
        # load offline mask
        if self.offline_mask_source is not None:
            # build the gt attn mask from the gt segment masks. True as masked.
            if self.offline_mask_source == 'gt':
                padded_segment_masks, padding_mask = pad_sequence_2d([sample['segment_masks'] for sample in batch], return_mask=True)
                new_batch['offline_attn_mask'] = padded_segment_masks.logical_not()
                
                # build labels and masks for loss
                labels = pad_sequence([sample['instance_labels'] for sample in batch], pad=-100)
                new_batch['target_labels'] = labels
                new_batch['target_masks'] = padded_segment_masks.float()
                new_batch['target_masks_pad_masks'] = padding_mask.logical_not()
            else: 
                raise NotImplementedError(f'{self.offline_mask_source} is not implemented')
            
        # list collate
        list_keys = ['voxel2segment', 'coordinates', 'voxel_to_full_maps', 'segment_to_full_maps', 'raw_coordinates', 'instance_ids', 'instance_labels', 'full_masks', 'segment_masks', 'scan_id']
        for k in list_keys:
            new_batch[k] = [sample.pop(k) for sample in batch]

        # pad collate: ['coord_min', 'coord_max', 'seg_center', 'seg_pad_mask', 'obj_center', 'obj_pad_mask']
        padding_keys = ['coord_min', 'coord_max', 'obj_center', 'obj_pad_masks', 'seg_center', 'seg_pad_masks', 'query_locs', 'query_pad_masks',
                        'voxel_seg_pad_masks', 'mv_seg_fts', 'mv_seg_pad_masks', 'pc_seg_fts', 'pc_seg_pad_masks', 'prompt', 'prompt_pad_masks']
        padding_keys =  [k for k in padding_keys if k in batch[0].keys()]
        for k in padding_keys:
            tensors = [sample.pop(k) for sample in batch]
            padded_tensor = pad_sequence(tensors)
            new_batch[k] = padded_tensor
        
          # multi-scale feature collate
        if "voxel_seg_fts" in batch[0].keys():
            new_voxel_seg_fts = []
            for i in range(len(batch[0]['voxel_seg_fts'])):
                cur_level_fts = [sample['voxel_seg_fts'][i].squeeze(0) for sample in batch]
                cur_level_fts = pad_sequence(cur_level_fts)
                new_voxel_seg_fts.append(cur_level_fts)
            new_batch['voxel_seg_fts'] = new_voxel_seg_fts
            for sample in batch:
                sample.pop('voxel_seg_fts')

        # default collate
        new_batch.update(default_collate(batch))
        return new_batch
 