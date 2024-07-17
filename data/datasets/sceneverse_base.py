import os
import collections
import json
import pickle
import random
import jsonlines
from torch_scatter import scatter_mean, scatter_add
from tqdm import tqdm
from scipy import sparse
import numpy as np
import torch
from torch.utils.data import Dataset

from common.misc import rgetattr
from ..data_utils import convert_pc_to_box, LabelConverter, build_rotate_mat, load_matrix_from_txt, \
                        construct_bbox_corners, eval_ref_one_sample  
from copy import deepcopy

SCAN_DATA = {'ScanNet': {}, '3RScan': {}}

'''
Usage of this class
init()
init_dataset_params()
load_split()
init_scan_data()

'''
class SceneVerseBase(Dataset):
    def __init__(self, cfg, dataset_name, split) -> None:
        # basic settings
        self.cfg = cfg
        self.dataset_name = dataset_name # ['ScanNet', '3RScan']
        self.split = split
        self.base_dir = cfg.data.scene_verse_base
        self.aux_dir = cfg.data.scene_verse_aux
        self.pred_dir = cfg.data.scene_verse_pred
        self.load_scan_options = cfg.data.get('load_scan_options', {})
        # label converter
        self.int2cat = json.load(open(os.path.join(self.base_dir,
                                            "ScanNet/annotations/meta_data/scannetv2_raw_categories.json"),
                                            'r', encoding="utf-8"))
        self.cat2int = {w: i for i, w in enumerate(self.int2cat)}
        self.label_converter = LabelConverter(os.path.join(self.base_dir,
                                            "ScanNet/annotations/meta_data/scannetv2-labels.combined.tsv"))
    
    def __len__(self):
        return len(self.lang_data)
    
    def __getitem__(self, index):
        scan_id, tgt_object_id, tgt_object_name, sentence, data_dict = self.get_lang(index)
        scene_dict = self.get_scene(scan_id, tgt_object_id, tgt_object_name, sentence)
        data_dict.update(scene_dict)
        return data_dict
    
    # init datasets
    def init_dataset_params(self, dataset_cfg):
        if dataset_cfg is None:
            dataset_cfg = {}
        self.pc_type = dataset_cfg.get('pc_type', 'gt') # 'gt' or 'pred'
        self.max_obj_len = dataset_cfg.get('max_obj_len', 80)
        self.num_points = dataset_cfg.get('num_points', 1024)
        self.filter_lang = dataset_cfg.get('filter_lang', False)
        self.rot_aug = dataset_cfg.get('rot_aug', True)
        self.train_duplicate = dataset_cfg.get('train_duplicate', 1)
        self.sem_type = dataset_cfg.get('sem_type', 607)
        assert self.pc_type in ['gt', 'pred']

    def _load_split(self, cfg, split):
        if self.dataset_name == '3RScan':
            split_file = os.path.join(self.aux_dir, f"3RScan/meta/3RScan.json")
            scan_ids = []
            with open(split_file, 'r') as f:
                rscan_json = json.load(f)
                for scene in rscan_json:
                    if split in scene['type']:
                        scan_ids.append(scene['reference'])
                        for sub_scene in scene['scans']:
                            scan_ids.append(sub_scene['reference'])
            # exclude scenes with points downsampled
            ignore_scan_ids = ['4e858ca1-fd93-2cb4-84b6-490997979830', 'd7d40d62-7a5d-2b36-955e-86a394caeabb', '4a9a43d8-7736-2874-86fc-098deb94c868']
            
            # exclude scenes with too many segments during training
            if split == 'train':
                exlcude_ids_file = os.path.join(self.aux_dir, "3RScan/meta/exclude_ids.json")
                with open(exlcude_ids_file, 'r') as f:
                    exclude_ids = json.load(f)
                    ignore_scan_ids += exclude_ids
            scan_ids = list(set(scan_ids) - set(ignore_scan_ids))
        elif self.dataset_name == 'ScanNet':
            if split == 'train':
                split_file = os.path.join(self.base_dir, 'ScanNet/annotations/splits/scannetv2_' + split + "_sort.json")
                with open(split_file, 'r') as f:
                    scan_ids = json.load(f)
            else:
                split_file = os.path.join(self.base_dir, 'ScanNet/annotations/splits/scannetv2_' + split + ".txt")
                scan_ids = {x.strip() for x in open(split_file, 'r', encoding="utf-8")}
                scan_ids = sorted(scan_ids)
        else:
            raise NotImplemented(f'data set name {self.dataset_name}')
        
        if cfg.debug.flag and cfg.debug.debug_size != -1:
            scan_ids = list(scan_ids)[:cfg.debug.debug_size]
        return scan_ids
        
    def init_scan_data(self):
        self.scan_data = self._load_scans(self.scan_ids)

        # build unique multiple look up
        if self.load_scan_options.get('load_inst_info', True):
            for scan_id in self.scan_ids:
                inst_labels = self.scan_data[scan_id]['inst_labels']
                self.scan_data[scan_id]['label_count'] = collections.Counter(
                                    [self.label_converter.id_to_scannetid[l] for l in inst_labels])
        
    def _load_scans(self, scan_ids):
        process_num = self.load_scan_options.get('process_num', 0)
        unloaded_scan_ids = [scan_id for scan_id in scan_ids if scan_id not in SCAN_DATA[self.dataset_name]]
        print(f"Loading scans: {len(unloaded_scan_ids)} / {len(scan_ids)}")
        scans = {}
        if process_num > 1:
            from joblib import Parallel, delayed
            res_all = Parallel(n_jobs=process_num)(
                delayed(self._load_one_scan)(scan_id) for scan_id in tqdm(unloaded_scan_ids))
            for scan_id, one_scan in tqdm(res_all):
                scans[scan_id] = one_scan
        else:
            for scan_id in tqdm(unloaded_scan_ids):
                _, one_scan = self._load_one_scan(scan_id)
                scans[scan_id] = one_scan

        SCAN_DATA[self.dataset_name].update(scans)
        scans = {scan_id: SCAN_DATA[self.dataset_name][scan_id] for scan_id in scan_ids}
        return scans
    
    def _load_one_scan(self, scan_id):
        options = self.load_scan_options
        one_scan = {}
        
        if options.get('load_pc_info', True):
            # load pcd data
            if self.dataset_name == '3RScan':
                pcd_data = torch.load(os.path.join("/mnt/fillipo/Datasets/SceneVerse_v3/3RScan", "scan_data/pcd_with_global_alignment", f'{scan_id}.pth'))
                inst_to_label = torch.load(os.path.join("/mnt/fillipo/Datasets/SceneVerse_v3/3RScan", 'scan_data/instance_id_to_label', f'{scan_id}.pth'))
                # process point cloud
                points, colors, instance_labels, ori_vertex_id  = pcd_data[0], pcd_data[1], pcd_data[2], pcd_data[3]
                one_scan['ori_vertex_id'] = ori_vertex_id
            else:
                pcd_data = torch.load(os.path.join(self.base_dir, self.dataset_name, "scan_data/pcd_with_global_alignment", f'{scan_id}.pth'))
                inst_to_label = torch.load(os.path.join(self.base_dir, self.dataset_name, 'scan_data/instance_id_to_label', f'{scan_id}.pth')) 
                points, colors, instance_labels  = pcd_data[0], pcd_data[1], pcd_data[-1]
            colors = colors / 127.5 - 1
            pcds = np.concatenate([points, colors], 1)
            one_scan['pcds'] = pcds
            one_scan['instance_labels'] = instance_labels
            one_scan['inst_to_label'] = inst_to_label
            # convert to gt object
            obj_pcds = []
            inst_ids = []
            inst_labels = []
            bg_indices = np.full((points.shape[0], ), 1, dtype=np.bool_)
            for inst_id in inst_to_label.keys():
                if inst_to_label[inst_id] in self.cat2int.keys():
                    mask = instance_labels == inst_id
                    if np.sum(mask) == 0:
                        continue
                    obj_pcds.append(pcds[mask])
                    inst_ids.append(inst_id)
                    inst_labels.append(self.cat2int[inst_to_label[inst_id]])
                    if inst_to_label[inst_id] not in ['wall', 'floor', 'ceiling']:
                        bg_indices[mask] = False
            one_scan['obj_pcds'] = obj_pcds
            one_scan['inst_labels'] = inst_labels
            one_scan['inst_ids'] = inst_ids
            one_scan['bg_pcds'] = pcds[bg_indices]
            # calculate box for matching
            obj_center = []
            obj_box_size = []
            for obj_pcd in obj_pcds:
                _c, _b = convert_pc_to_box(obj_pcd)
                obj_center.append(_c)
                obj_box_size.append(_b)
            one_scan['obj_loc'] = obj_center
            one_scan['obj_box'] = obj_box_size
            # convert pc to pred object
            obj_mask_path = os.path.join(self.pred_dir, self.dataset_name, "mask", str(scan_id) + ".mask" + ".npz")
            if os.path.exists(obj_mask_path):
                obj_label_path = os.path.join(self.pred_dir, self.dataset_name, "mask", str(scan_id) + ".label" + ".npy")
                # load mask and labels
                topk = 50
                obj_masks = np.array(sparse.load_npz(obj_mask_path).todense(), dtype=bool)[:topk] # TODO: only keep the top-50 bbox for now
                obj_labels = np.load(obj_label_path)[:topk]
                # save obj_pcds_pred, inst_labels_pred, bg_pcds_pred, obj_loc_pred, obj_box_pred
                obj_pcds = []
                inst_labels = []
                bg_indices = np.full((pcds.shape[0], ), 1, dtype=np.bool_)
                for i in range(obj_masks.shape[0]):
                    mask = obj_masks[i]
                    if pcds[mask == 1, :].shape[0] > 0:
                        obj_pcds.append(pcds[mask == 1, :])
                        inst_labels.append(obj_labels[i])
                        # if not self.int2cat[obj_labels[i]] in ['wall', 'floor', 'ceiling']:
                        bg_indices[mask == 1] = False
                one_scan['obj_pcds_pred'] = obj_pcds
                one_scan['inst_labels_pred'] = inst_labels
                one_scan['bg_pcds_pred'] = pcds[bg_indices]
                # calculate box for pred
                obj_center_pred = []
                obj_box_size_pred = []
                for obj_pcd in obj_pcds:
                    _c, _b = convert_pc_to_box(obj_pcd)
                    obj_center_pred.append(_c)
                    obj_box_size_pred.append(_b)
                one_scan['obj_loc_pred'] = obj_center_pred
                one_scan['obj_box_pred'] = obj_box_size_pred
                
                self.match_gt_to_pred(one_scan)
                
        if options.get('load_segment_info', False):
            one_scan["segment_id"] = np.load(os.path.join(self.aux_dir, self.dataset_name,  "segment_id", f'{scan_id}.npy')).astype(int)
            if self.dataset_name == '3RScan':
                segment_id = one_scan['segment_id'][one_scan['ori_vertex_id']]
                if segment_id.max() != one_scan['segment_id'].max():
                    segment_id[-1] = one_scan['segment_id'].max()
                one_scan['segment_id'] = segment_id
                    
        if options.get('load_image_segment_feat', False):
            one_scan['image_segment_feat'] = torch.load(os.path.join(self.aux_dir, self.dataset_name, "image_seg_feat", f'{scan_id}.pth'), map_location='cpu')
            
        if options.get('load_point_segment_feat', False):
            one_scan['point_segment_feat'] = torch.load(os.path.join(self.aux_dir, self.dataset_name, "point_seg_feat", f'{scan_id}.pth'), map_location='cpu')
        
        if options.get('load_image_obj_feat', False):
            # load image feat gt
            feat_pth = os.path.join(self.pred_dir, self.dataset_name, f'image_obj_feat_gt', scan_id + '.pth')
            if os.path.exists(feat_pth):
                feat_dict = torch.load(feat_pth)
                feat_dim = next(iter(feat_dict.values())).shape[0]
                n_obj = len(one_scan['inst_ids'])  # the last one is for missing objects.
                feat = torch.zeros((n_obj, feat_dim), dtype=torch.float32)
                for i, cid in enumerate(one_scan['inst_ids']):
                    if cid in feat_dict.keys():
                        feat[i] =  feat_dict[cid]
                one_scan['image_obj_feat_gt'] = feat
            # load image feat pred
            feat_pth = os.path.join(self.pred_dir, self.dataset_name, f'image_obj_feat_pred', scan_id + '.pth')
            if os.path.exists(feat_pth):
                feat_dict = torch.load(feat_pth)
                feat_dim = next(iter(feat_dict.values())).shape[0]
                n_obj = len(one_scan['obj_pcds_pred'])  
                feat = torch.zeros((n_obj, feat_dim), dtype=torch.float32)
                for i in range(n_obj):
                    feat[i] =  feat_dict[i] 
                one_scan['image_obj_feat_pred'] = feat
        
        if options.get('load_voxel_obj_feat', False):
            # load voxel feat gt
            feat_pth = os.path.join(self.pred_dir, self.dataset_name, f'voxel_obj_feat_gt', scan_id + '.pth')
            if os.path.exists(feat_pth):
                feat_dict = torch.load(feat_pth)
                feat_dim = next(iter(feat_dict.values())).shape[0]
                n_obj = len(one_scan['inst_ids'])  # the last one is for missing objects.
                feat = torch.zeros((n_obj, feat_dim), dtype=torch.float32)
                for i, cid in enumerate(one_scan['inst_ids']):
                    if cid in feat_dict.keys():
                        feat[i] =  feat_dict[cid]
                one_scan['voxel_obj_feat_gt'] = feat
            # load voxel feat pred
            feat_pth = os.path.join(self.pred_dir, self.dataset_name, f'voxel_obj_feat_pred', scan_id + '.pth')
            if os.path.exists(feat_pth):
                feat_dict = torch.load(feat_pth)
                feat_dim = next(iter(feat_dict.values())).shape[0]
                n_obj = len(one_scan['obj_pcds_pred'])  
                feat = torch.zeros((n_obj, feat_dim), dtype=torch.float32)
                for i in range(n_obj):
                    feat[i] =  feat_dict[i] 
                one_scan['voxel_obj_feat_pred'] = feat

        return (scan_id, one_scan)
    
    def _load_lang(self, cfg):
        raise NotImplementedError
    
    def get_lang(self, index):
        raise NotImplementedError
    
    def get_scene(self, scan_id, tgt_object_id_list, tgt_object_name_list, sentence):
        if not isinstance(tgt_object_id_list, list):
            tgt_object_id_list = [tgt_object_id_list]
        if not isinstance(tgt_object_name_list, list):
            tgt_object_name_list = [tgt_object_name_list]
        # load gt obj
        scan_data = self.scan_data[scan_id]
        obj_labels = scan_data['inst_labels'] # N
        obj_pcds = scan_data['obj_pcds']
        obj_ids = scan_data['inst_ids']
        # convert tgt_object_id to local id
        tgt_object_id_list = [obj_ids.index(tid) for tid in tgt_object_id_list]
        # build target info 
        tgt_obj_boxes = np.array([scan_data['obj_loc'][i] + scan_data['obj_box'][i] for i in range(len(scan_data['obj_loc']))])[tgt_object_id_list]
        tgt_object_label_list = [obj_labels[x] for x in tgt_object_id_list]
        is_multiple = sum([scan_data['label_count'][self.label_converter.id_to_scannetid[x]] 
                           for x in tgt_object_label_list]) > 1
        tgt_object_id_iou25_list = tgt_object_id_list
        tgt_object_id_iou50_list = tgt_object_id_list
        # pred masks
        if self.pc_type == 'pred':
            obj_pcds = scan_data['obj_pcds_pred']
            obj_labels = scan_data['inst_labels_pred']

            # rebulid tgt_object_id_list
            iou_list = [j for i in tgt_object_id_list for j in scan_data['iou25_list'][i]]
            tgt_object_id_iou25_list = sorted(set(iou_list))
            iou_list = [j for i in tgt_object_id_list for j in scan_data['iou50_list'][i]]
            tgt_object_id_iou50_list = sorted(set(iou_list))
            tgt_object_id_list = [scan_data['matched_list'][i] for i in tgt_object_id_list]
            tgt_object_label_list = [obj_labels[x] for x in tgt_object_id_list]
        # object filter
        excluded_labels = ['wall', 'floor', 'ceiling']
        def keep_obj(i, obj_label):
            # do not filter for predicted labels, because these labels are not accurate
            if self.pc_type != 'gt' or i in tgt_object_id_list:
                return True
            category = self.int2cat[obj_label]
            # filter out background
            if category in excluded_labels:
                return False
            # filter out objects not mentioned in the sentence
            if self.filter_lang and category not in sentence:
                return False
            return True
        selected_obj_idxs = [i for i, obj_label in enumerate(obj_labels) if keep_obj(i, obj_label)]
        # crop objects to max_obj_len
        if self.max_obj_len < len(selected_obj_idxs):
            pre_selected_obj_idxs = selected_obj_idxs
            # select target first
            selected_obj_idxs = tgt_object_id_list[:]
            selected_obj_idxs.extend(tgt_object_id_iou25_list)
            selected_obj_idxs.extend(tgt_object_id_iou50_list)
            selected_obj_idxs = list(set(selected_obj_idxs))
            # select object with same semantic class with tgt_object
            remained_obj_idx = []
            for i in pre_selected_obj_idxs:
                label = obj_labels[i]
                if i not in selected_obj_idxs:
                    if label in tgt_object_label_list:
                        selected_obj_idxs.append(i)
                    else:
                        remained_obj_idx.append(i)
                if len(selected_obj_idxs) >= self.max_obj_len:
                    break
            if len(selected_obj_idxs) < self.max_obj_len:
                random.shuffle(remained_obj_idx)
                selected_obj_idxs += remained_obj_idx[:(self.max_obj_len - len(selected_obj_idxs))]
            # assert len(selected_obj_idxs) == self.max_obj_len

        # reorganize ids
        tgt_object_id_list = [selected_obj_idxs.index(id) for id in tgt_object_id_list]
        tgt_object_id_iou25_list = [selected_obj_idxs.index(id) for id in tgt_object_id_iou25_list]
        tgt_object_id_iou50_list = [selected_obj_idxs.index(id) for id in tgt_object_id_iou50_list]

        obj_labels = [obj_labels[i] for i in selected_obj_idxs]
        obj_pcds = [obj_pcds[i] for i in selected_obj_idxs]
        
        # subsample points
        obj_pcds = np.array([obj_pcd[np.random.choice(len(obj_pcd), size=self.num_points,
                                        replace=len(obj_pcd) < self.num_points)] for obj_pcd in obj_pcds])
        obj_fts, obj_locs, obj_boxes, rot_matrix = self._obj_processing_post(obj_pcds, rot_aug=self.rot_aug)

        data_dict = {
            "scan_id": scan_id,
            "tgt_object_id": torch.LongTensor(tgt_object_id_list),
            "tgt_object_label": torch.LongTensor(tgt_object_label_list),
            "tgt_obj_boxes": tgt_obj_boxes.tolist(), # w/o augmentation, only use it for evaluation, tolist is for collate.
            "obj_fts": obj_fts.float(),
            "obj_locs": obj_locs.float(),
            "obj_labels": torch.LongTensor(obj_labels),
            "obj_boxes": obj_boxes, 
            "obj_pad_masks": torch.ones((len(obj_locs)), dtype=torch.bool), # used for padding in collate
            "tgt_object_id_iou25": torch.LongTensor(tgt_object_id_iou25_list),
            "tgt_object_id_iou50": torch.LongTensor(tgt_object_id_iou50_list), 
            'is_multiple': is_multiple,
        }
        # load feature
        feat_type = self.pc_type
        if self.load_scan_options.get('load_image_obj_feat', False):
            feats = scan_data['image_obj_feat_' + feat_type]
            valid = selected_obj_idxs
            data_dict['mv_seg_fts'] = feats[valid]
            data_dict['mv_seg_pad_masks'] = torch.ones(len(data_dict['mv_seg_fts']), dtype=torch.bool)

        if self.load_scan_options.get('load_voxel_obj_feat', False):
            feats = scan_data['voxel_obj_feat_' + feat_type]
            valid = selected_obj_idxs
            data_dict['voxel_seg_fts'] = feats[valid]
            data_dict['voxel_seg_pad_masks'] = torch.ones(len(data_dict['voxel_seg_fts']), dtype=torch.bool)
        
        data_dict['pc_seg_fts'] = obj_fts.float()
        data_dict['pc_seg_pad_masks'] = torch.ones(len(data_dict['pc_seg_fts']), dtype=torch.bool)
        # build query and seg pos
        data_dict['query_locs'] = data_dict['obj_locs'].clone()
        data_dict['query_pad_masks'] = data_dict['obj_pad_masks'].clone()
        data_dict['coord_min'] = data_dict['obj_locs'][:, :3].min(0)[0]
        data_dict['coord_max'] = data_dict['obj_locs'][:, :3].max(0)[0]
        data_dict['seg_center'] = obj_locs.float()
        data_dict['seg_pad_masks'] = data_dict['obj_pad_masks']
        return data_dict
    
    # some utility functions
    def match_gt_to_pred(self, scan_data):
        gt_boxes = [scan_data['obj_loc'][i] + scan_data['obj_box'][i] for i in range(len(scan_data['obj_loc']))]
        pred_boxes = [scan_data['obj_loc_pred'][i] + scan_data['obj_box_pred'][i] for i in range(len(scan_data['obj_loc_pred']))]

        def get_iou_from_box(gt_box, pred_box):
            gt_corners = construct_bbox_corners(gt_box[:3], gt_box[3:6])
            pred_corners = construct_bbox_corners(pred_box[:3], pred_box[3:6])
            iou = eval_ref_one_sample(gt_corners, pred_corners)
            return iou
        
        matched_list, iou25_list, iou50_list = [], [], []
        for cur_id in range(len(gt_boxes)):
            gt_box = gt_boxes[cur_id]
            max_iou = -1
            iou25, iou50 = [], []
            for i in range(len(pred_boxes)):
                pred_box = pred_boxes[i]
                iou = get_iou_from_box(gt_box, pred_box)
                if iou > max_iou:
                    max_iou = iou
                    object_id_matched = i
                # find tgt iou 25
                if iou >= 0.25:
                    iou25.append(i)
                # find tgt iou 50
                if iou >= 0.5:
                    iou50.append(i)
            matched_list.append(object_id_matched)
            iou25_list.append(iou25)
            iou50_list.append(iou50)
        
        scan_data['matched_list'] = matched_list
        scan_data['iou25_list'] = iou25_list
        scan_data['iou50_list'] = iou50_list

    def _obj_processing_post(self, obj_pcds, rot_aug=True):
        obj_pcds = torch.from_numpy(obj_pcds)
        rot_matrix = build_rotate_mat(self.split, rot_aug)
        if rot_matrix is not None:
            rot_matrix = torch.from_numpy(rot_matrix.transpose())
            obj_pcds[:, :, :3] @= rot_matrix
        
        xyz = obj_pcds[:, :, :3]
        center = xyz.mean(1)
        xyz_min = xyz.min(1).values
        xyz_max = xyz.max(1).values
        box_center = (xyz_min + xyz_max) / 2
        size = xyz_max - xyz_min
        obj_locs = torch.cat([center, size], dim=1)
        obj_boxes = torch.cat([box_center, size], dim=1)

        # centering
        obj_pcds[:, :, :3].sub_(obj_pcds[:, :, :3].mean(1, keepdim=True))

        # normalization
        max_dist = (obj_pcds[:, :, :3]**2).sum(2).sqrt().max(1).values
        max_dist.clamp_(min=1e-6)
        obj_pcds[:, :, :3].div_(max_dist[:, None, None])
        
        return obj_pcds, obj_locs, obj_boxes, rot_matrix