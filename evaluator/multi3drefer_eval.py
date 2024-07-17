"""adapted from https://github.com/3dlg-hcvc/M3DRef-CLIP/blob/main/m3drefclip/evaluation/multi3drefer_evaluator.py"""
from pathlib import Path
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

from data.data_utils import construct_bbox_corners, box3d_iou
from evaluator.build import EVALUATOR_REGISTRY, BaseEvaluator


@EVALUATOR_REGISTRY.register()
class Multi3DReferEval(BaseEvaluator):
    def __init__(self, cfg, accelerator, **kwargs):
        self.target_metric = 'iou50_overall'
        self.save_dir = Path(cfg.exp_dir) / "eval_results" / self.__class__.__name__
        super().__init__(cfg, accelerator, **kwargs)

    def reset(self):
        super().reset()

        self.iou25_f1_scores = []
        self.iou50_f1_scores = []

    def batch_metrics(self, data_dict, include_count=False):
        metrics = {}
        probs = torch.sigmoid(data_dict['og3d_logits'])
        obj_boxes = data_dict['obj_boxes']
        pred_obj_ids = probs > 0.5
        preds = [boxes[ids].cpu().numpy() for ids, boxes in zip(pred_obj_ids, obj_boxes)]

        eval_types = data_dict['eval_type']
        gts = data_dict['tgt_obj_boxes']
        f1_scores = [self.evaluate_one_query(pred, gt) for pred, gt in zip(preds, gts)]
        iou25_f1_scores = np.array([f1_score[0] for f1_score in f1_scores])
        iou50_f1_scores = np.array([f1_score[1] for f1_score in f1_scores])
        for sub_group in ("zt_w_d", "zt_wo_d", "st_w_d", "st_wo_d", "mt"):
            selected = np.array([e == sub_group for e in eval_types])
            metrics['iou25_' + sub_group] = (sum(iou25_f1_scores[selected]), sum(selected)) if np.any(selected) else (0, 0)
            metrics['iou50_' + sub_group] = (sum(iou50_f1_scores[selected]), sum(selected)) if np.any(selected) else (0, 0)
        metrics['iou25_overall'] = (sum(iou25_f1_scores), len(iou25_f1_scores))
        metrics['iou50_overall'] = (sum(iou50_f1_scores), len(iou50_f1_scores))

        if self.save:
            item_ids = data_dict['data_idx']
            for i in range(len(item_ids)):
                self.eval_results.append({
                    "scene_id": item_ids[i],
                    "eval_type": eval_types[i],
                    "sentence": data_dict['sentence'][i],
                    "pred": preds[i].tolist(),
                    "gt": gts[i],
                })

        return metrics

    @staticmethod
    def evaluate_one_query(pred_info, gt_info):
        pred_bboxes_count = len(pred_info)
        gt_bboxes_count = len(gt_info)

        if pred_bboxes_count == 0 and gt_bboxes_count == 0:
            return 1, 1
        if pred_bboxes_count == 0 or gt_bboxes_count == 0:
            return 0, 0

        # initialize true positives
        iou25_tp = 0
        iou50_tp = 0

        # initialize the cost matrix
        square_matrix_len = max(gt_bboxes_count, pred_bboxes_count)
        iou_matrix = np.zeros(shape=(square_matrix_len, square_matrix_len), dtype=np.float32)
        # TODO: convert to batch process
        for i, pred_box in enumerate(pred_info):
            for j, gt_box in enumerate(gt_info):
                obj_center, obj_box_size = pred_box[:3], pred_box[3:]
                gt_center, gt_box_size = gt_box[:3], gt_box[3:]
                iou_matrix[i, j] = box3d_iou(construct_bbox_corners(obj_center, obj_box_size),
                                            construct_bbox_corners(gt_center, gt_box_size))

        # apply matching algorithm
        row_idx, col_idx = linear_sum_assignment(iou_matrix * -1)

        # iterate matched pairs, check ious
        for i in range(pred_bboxes_count):
            iou = iou_matrix[row_idx[i], col_idx[i]]
            # calculate true positives
            if iou >= 0.25:
                iou25_tp += 1
            if iou >= 0.5:
                iou50_tp += 1

        # calculate precision, recall and f1-score for the current scene
        iou25_f1_score = 2 * iou25_tp / (pred_bboxes_count + gt_bboxes_count)
        iou50_f1_score = 2 * iou50_tp / (pred_bboxes_count + gt_bboxes_count)
        return iou25_f1_score, iou50_f1_score