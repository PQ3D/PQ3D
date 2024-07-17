import gc
import math
import json
import os
from pathlib import Path

import numpy as np
import scipy
import torch
from torch_scatter import scatter_mean
from sklearn.cluster import DBSCAN

from evaluator.build import EVALUATOR_REGISTRY, BaseEvaluator
from common.metric_utils import IoU, ConfusionMatrix
from common.metric_utils import IoU
from common.eval_det import eval_det
from common.eval_instseg import eval_instseg
from common.misc import gather_dict
from data.datasets.constant import VALID_CLASS_IDS_200_VALIDATION, HEAD_CATS_SCANNET_200, COMMON_CATS_SCANNET_200, TAIL_CATS_SCANNET_200


@EVALUATOR_REGISTRY.register()
class InstSegEval(BaseEvaluator):
    def __init__(self, cfg, accelerator, dataloaders=None, **kwargs):
        # config
        self.config = cfg
        # record metrics
        self.eval_dict = {'target_metric': []}
        self.target_metric = cfg.eval.get('target_metric', 'all_ap')
        self.total_count = 0
        self.best_result = -np.inf
        # save dir
        self.save = cfg.eval.save
        if self.save:
            self.eval_results = []
            self.save_dir = Path(cfg.exp_dir) / "eval_results" / "InstSeg"
            self.save_dir.mkdir(parents=True, exist_ok=True)
        # record
        self.preds = dict()
        self.bbox_preds = dict()
        self.bbox_gt = dict()
        # misc
        self.dataset_name = cfg.eval.get('dataset_name', 'ScanNet')
        self.ignore_label = cfg.eval.ignore_label
        self.use_dbscan = cfg.eval.get("use_dbscan", False)
        self.filter_out_no_object_queries = cfg.eval.get('filter_out_no_object_queries', False)
        self.label_converter = dataloaders['test'].dataset.dataset.label_converter
        self.accelerator = accelerator
        self.device = self.accelerator.device
    
    def map_scannet200_id_to_scannet_raw_id(self, output):
        label_info = self.label_converter.scannet200_id_to_scannet_raw_id
        output = np.array(output)
        output_remapped = output.copy()
        for k, v in label_info.items():
            output_remapped[output == k] = v
        return output_remapped

    def update(self, data_dict):
        metrics = self.batch_metrics(data_dict)
        predictions_class = data_dict['predictions_class']
        predictions_mask = data_dict['predictions_mask']
        voxel_to_full_maps = data_dict['voxel_to_full_maps']
        voxel2segment = data_dict['voxel2segment']
        segment_to_full_maps = data_dict['segment_to_full_maps']
        scan_ids = data_dict['scan_id']
        raw_coordinates = data_dict['raw_coordinates']
        instance_labels = data_dict['instance_labels']
        full_masks = data_dict['full_masks']
        voxel_features = data_dict['voxel_features']
        self.eval_instance_step(predictions_class, predictions_mask, voxel_to_full_maps, voxel2segment, segment_to_full_maps, 
                                instance_labels, raw_coordinates, full_masks, scan_ids, voxel_features[:, -3:])
        self.total_count += metrics["total_count"]

    def batch_metrics(self, data_dict):
        metrics = {}
        metrics["total_count"] = len(data_dict['predictions_class'][0])
        return metrics

    def reset(self):
        for key in self.eval_dict.keys():
            self.eval_dict[key] = []
        self.total_count = 0
    
    def eval_instance_step(self, predictions_class, predictions_mask, voxel_to_full_maps, voxel2segment, segment_to_full_maps, 
                           instance_labels, raw_coordinates, full_masks, scan_ids, coordinates_per_voxel):
        # get all predictions
        pred_masks = predictions_mask[-1]
        pred_logits = torch.functional.F.softmax(predictions_class[-1], dim=-1) # ignore last logit (201 for no class), (B, N, 201)
        
        # save every thing
        all_pred_classes = list()
        all_pred_masks = list()
        all_pred_scores = list()
        all_heatmaps = list()
        
        # get masks
        offset_coords_idx = 0
        for bid in range(len(pred_masks)):
            # convert segment mask to voxel mask, (s, q) to (v, q)
            masks = pred_masks[bid].detach().cpu()[voxel2segment[bid].cpu()]
            logits = pred_logits[bid].detach().cpu() # (q, 201)
            
            # filter out no object queries
            if self.filter_out_no_object_queries:
                valid_query_mask = torch.argmax(logits, dim=-1) != 200
                masks = masks[:, valid_query_mask]
                logits = logits[valid_query_mask][..., :-1]
            else:
                logits = logits[..., :-1]
           
            # dbscan
            if self.use_dbscan:
                coordinates_per_voxel = coordinates_per_voxel[offset_coords_idx:offset_coords_idx + len(masks)].cpu()
                offset_coords_idx += len(masks)
                masks, logits = self.dbscan(coordinates_per_voxel, masks, logits)
                
            # compute mask, scores, heatmap
            scores, masks, classes, heatmap = self.get_mask_and_scores(logits, masks)
            
           # convert voxel to full resolution
            masks = self.get_full_res_mask(masks,
                                            voxel_to_full_maps[bid].cpu(),
                                            segment_to_full_maps[bid].cpu())

            heatmap = self.get_full_res_mask(heatmap,
                                                voxel_to_full_maps[bid].cpu(),
                                                segment_to_full_maps[bid].cpu(),
                                                is_heatmap=True)
            masks = masks.numpy()
            heatmap = heatmap.numpy()
            classes = classes.numpy()
            
            # sort scores
            sort_scores = scores.sort(descending=True)
            sort_scores_index = sort_scores.indices.cpu().numpy()
            sort_scores_values = sort_scores.values.cpu().numpy()
            # rearange score, mask, class, heatmap
            sort_classes = classes[sort_scores_index]
            sorted_masks = masks[:, sort_scores_index]
            sorted_heatmap = heatmap[:, sort_scores_index]
            # append
            all_pred_classes.append(sort_classes)
            all_pred_masks.append(sorted_masks)
            all_pred_scores.append(sort_scores_values)
            all_heatmaps.append(sorted_heatmap)
        
        # map to scannet raw and save
        self.record_preds(all_pred_scores, all_pred_masks, all_pred_classes, full_masks, raw_coordinates, instance_labels, scan_ids)

    def record(self):
        data = {'preds': list(self.preds.items()), 'bbox_preds': list(self.bbox_preds.items()), 'bbox_gt': list(self.bbox_gt.items())}
        data = gather_dict(self.accelerator, data)
        self.preds = dict(data['preds'])
        self.bbox_preds = dict(data['bbox_preds'])
        self.bbox_gt = dict(data['bbox_gt'])
        # record  box ap
        ap_results = {}
        
        box_ap_50 = eval_det(self.bbox_preds, self.bbox_gt, ovthresh=0.5, use_07_metric=False)
        box_ap_25 = eval_det(self.bbox_preds, self.bbox_gt, ovthresh=0.25, use_07_metric=False)
        mean_box_ap_25 = [v for k, v in box_ap_25[-1].items() if not math.isnan(v)]
        mean_box_ap_25 = sum(mean_box_ap_25) / len(mean_box_ap_25)
        mean_box_ap_50 = [v for k, v in box_ap_50[-1].items() if not math.isnan(v)]
        mean_box_ap_50 = sum(mean_box_ap_50) / len(mean_box_ap_50)
        print(f"mean_box_ap 50 {mean_box_ap_50}")
        print(f"mean_box_ap 25 {mean_box_ap_25}")

        ap_results[f"mean_box_ap_25"] = mean_box_ap_25
        ap_results[f"mean_box_ap_50"] = mean_box_ap_50

        for class_id in box_ap_50[-1].keys():
            class_name = self.label_converter.scannet_raw_id_to_raw_name[class_id]
            ap_results[f"{class_name}_val_box_ap_50"] = box_ap_50[-1][class_id]

        for class_id in box_ap_25[-1].keys():
            class_name = self.label_converter.scannet_raw_id_to_raw_name[class_id]
            ap_results[f"{class_name}_val_box_ap_25"] = box_ap_25[-1][class_id]
        
        # record instance ap
        instseg_results = eval_instseg(self.preds, self.dataset_name, self.config)
        ap_results.update(instseg_results)
        
        # record head common tail
        head_results, tail_results, common_results = [], [], []
        for class_name in VALID_CLASS_IDS_200_VALIDATION:
            if class_name not in ap_results['classes'].keys():
                continue
            ap = ap_results["classes"][class_name]["ap"]
            ap_50 = ap_results["classes"][class_name]["ap50%"]
            ap_25 = ap_results["classes"][class_name]["ap25%"]
            if class_name in HEAD_CATS_SCANNET_200:
                head_results.append(np.array((float(ap), float(ap_50), float(ap_25))))
            elif class_name in COMMON_CATS_SCANNET_200:
                common_results.append(np.array((float(ap), float(ap_50), float(ap_25))))
            elif class_name in TAIL_CATS_SCANNET_200:
                tail_results.append(np.array((float(ap), float(ap_50), float(ap_25))))
        head_results = np.stack(head_results)
        common_results = np.stack(common_results)
        tail_results = np.stack(tail_results)
        mean_tail_results = np.nanmean(tail_results, axis=0)
        mean_common_results = np.nanmean(common_results, axis=0)
        mean_head_results = np.nanmean(head_results, axis=0)
        ap_results[f"mean_tail_ap"] = mean_tail_results[0]
        ap_results[f"mean_common_ap"] = mean_common_results[0]
        ap_results[f"mean_head_ap"] = mean_head_results[0]
        ap_results[f"mean_tail_ap_50"] = mean_tail_results[1]
        ap_results[f"mean_common_ap_50"] = mean_common_results[1]
        ap_results[f"mean_head_ap_50"] = mean_head_results[1]
        ap_results[f"mean_tail_ap_25"] = mean_tail_results[2]
        ap_results[f"mean_common_ap_25"] = mean_common_results[2]
        ap_results[f"mean_head_ap_25"] = mean_head_results[2]
        overall_ap_results = np.nanmean(np.vstack((head_results, common_results, tail_results)), axis=0)
        ap_results[f"mean_ap"] = overall_ap_results[0]
        ap_results[f"mean_ap_50"] = overall_ap_results[1]
        ap_results[f"mean_ap_25"] = overall_ap_results[2]
        eval_results = ap_results
        
        # save results
        if self.save:
            with (self.save_dir / "results.json").open("w") as f:
                json.dump({'preds': self.preds, 'bbox_preds': self.bbox_preds, 'bbox_gt': self.bbox_gt}, f)
        if eval_results[self.target_metric] > self.best_result:
            is_best = True
            self.best_result = eval_results[self.target_metric]
        else:
            is_best = False

        # clean
        del self.preds
        del self.bbox_preds
        del self.bbox_gt

        gc.collect()

        self.preds = dict()
        self.bbox_preds = dict()
        self.bbox_gt = dict()
        torch.cuda.empty_cache()
        
        eval_results['target_metric'] = eval_results[self.target_metric]
        eval_results['best_result'] = self.best_result
        return is_best, eval_results
    
    def dbscan(self, coordinates_per_voxel, masks, logits, ):
        masks_list = list()
        logits_list = list()
        curr_coords = coordinates_per_voxel
        
        for curr_query in range(masks.shape[1]):
            curr_masks = masks[:, curr_query] > 0

            if curr_coords[curr_masks].shape[0] > 0:
                clusters = DBSCAN(eps=0.95,
                                    min_samples=1,
                                    n_jobs=-1).fit(curr_coords[curr_masks]).labels_

                new_mask = torch.zeros(curr_masks.shape, dtype=int)
                new_mask[curr_masks] = torch.from_numpy(clusters) + 1

                for cluster_id in np.unique(clusters):
                    original_pred_masks = masks[:, curr_query]
                    if cluster_id != -1:
                        masks_list.append(original_pred_masks * (new_mask == cluster_id + 1))
                        logits_list.append(
                            logits[curr_query])
                        
        logits_list = torch.stack(logits_list).cpu()
        masks_list = torch.stack(masks_list).T
        return masks_list, logits_list
        
    def get_full_res_mask(self, mask, voxel_to_full_maps, segment_to_full_maps, is_heatmap=False):
        mask = mask.detach().cpu()[voxel_to_full_maps]  # full res

        segment_to_full_maps = segment_to_full_maps
        if is_heatmap==False:
            mask = scatter_mean(mask, segment_to_full_maps, dim=0)  # full res segments
            mask = (mask > 0.5).float()
            mask = mask.detach().cpu()[segment_to_full_maps]  # full res points

        return mask

    def get_mask_and_scores(self, logits, masks):
        num_classes = logits.shape[1]
        num_queries = logits.shape[0]
        labels = torch.arange(num_classes).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)
        # topk selection, logits(q, 200), masks(v, q)
        if self.config.eval.topk_per_scene != -1 :
            scores_per_query, topk_indices = logits.flatten(0, 1).topk(self.config.eval.topk_per_scene, sorted=True)
        else:
            scores_per_query, topk_indices = logits.flatten(0, 1).topk(num_queries, sorted=True)

        labels_per_query = labels[topk_indices]
        topk_indices = torch.div(topk_indices, num_classes, rounding_mode='trunc')
        masks = masks[:, topk_indices]

        result_pred_mask = (masks > 0).float()
        heatmap = masks.float().sigmoid()

        mask_scores_per_image = (heatmap * result_pred_mask).sum(0) / (result_pred_mask.sum(0) + 1e-6)
        score = scores_per_query * mask_scores_per_image
        classes = labels_per_query

        return score, result_pred_mask, classes, heatmap
    
    def record_preds(self, all_pred_scores, all_pred_masks, all_pred_classes, full_masks, raw_coordinates, instance_labels, scan_ids):
        for bid in range(len(all_pred_masks)):
            all_pred_classes[bid] = self.map_scannet200_id_to_scannet_raw_id(all_pred_classes[bid])
            instance_labels[bid] = self.map_scannet200_id_to_scannet_raw_id(instance_labels[bid].cpu())

            # PREDICTION BOX
            bbox_data = []
            for query_id in range(all_pred_masks[bid].shape[1]):  # self.config.model.num_queries
                obj_coords = raw_coordinates[bid][all_pred_masks[bid][:, query_id].astype(bool), :]
                if obj_coords.shape[0] > 0:
                    obj_center = obj_coords.mean(axis=0)
                    obj_axis_length = obj_coords.max(axis=0) - obj_coords.min(axis=0)

                    bbox = np.concatenate((obj_center, obj_axis_length))

                    bbox_data.append((all_pred_classes[bid][query_id].item(), bbox,
                                        all_pred_scores[bid][query_id]
                    ))
            self.bbox_preds[scan_ids[bid]] = bbox_data
            
            # GT BOX 
            bbox_data = []
            for obj_id in range(full_masks[bid].shape[0]):
                if instance_labels[bid][obj_id].item() == self.ignore_label:
                    continue

                obj_coords = raw_coordinates[bid][full_masks[bid][obj_id, :].cpu().detach().numpy().astype(bool), :]
                if obj_coords.shape[0] > 0:
                    obj_center = obj_coords.mean(axis=0)
                    obj_axis_length = obj_coords.max(axis=0) - obj_coords.min(axis=0)

                    bbox = np.concatenate((obj_center, obj_axis_length))
                    bbox_data.append((instance_labels[bid][obj_id].item(), bbox))
            self.bbox_gt[scan_ids[bid]] = bbox_data
            
            self.preds[scan_ids[bid]] = {
                'pred_masks': all_pred_masks[bid],
                'pred_scores': all_pred_scores[bid],
                'pred_classes': all_pred_classes[bid]
            }

  