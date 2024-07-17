import os
import json
import collections
from pathlib import Path
import torch

from evaluator.build import EVALUATOR_REGISTRY, BaseEvaluator
from data.data_utils import ScanQAAnswer
from common.box_utils import get_3d_box
from evaluator.build import EVALUATOR_REGISTRY


@EVALUATOR_REGISTRY.register()
class ScanQAEval(BaseEvaluator):
    def __init__(self, cfg, accelerator, **kwargs):
        self.target_metric = 'ans1_acc'
        self.save_dir = Path(cfg.exp_dir) / "eval_results" / self.__class__.__name__
        super().__init__(cfg, accelerator, **kwargs)

        if self.save:
            train_data = json.load(open(os.path.join(cfg.data.scan_family_base,
                                    'annotations/qa/ScanQA_v1.0_train.json'), encoding='utf-8'))
            answer_counter = sum([data['answers'] for data in train_data], [])
            answer_counter = collections.Counter(sorted(answer_counter))
            answer_cands = answer_counter.keys()
            self.answer_vocab = ScanQAAnswer(answer_cands)

    def batch_metrics(self, data_dict, include_count=False):
        metrics = {}
        total_count = len(data_dict['answer_scores'])
        # ans
        answer_label = data_dict['answer_label']
        choice_1 = data_dict['answer_scores'].argmax(dim=-1)
        choice_10 = torch.topk(data_dict['answer_scores'], 10, -1)[1]

        correct1_mask = answer_label[torch.arange(total_count), choice_1] == 1
        correct1 = correct1_mask.sum().item()
        correct10_mask = answer_label[torch.arange(total_count)[:, None], choice_10] == 1
        correct10 = correct10_mask.any(dim=-1).sum().item()
        metrics['ans1_acc'] = correct1
        metrics['ans10_acc'] = correct10 

        for key in metrics:
            if isinstance(metrics[key], tuple):
                # already has count
                continue
            metrics[key] = (metrics[key], total_count)

        if self.save:
            for i in range(total_count):
                answer_top10 = [self.answer_vocab.itos(choice_10[i, j].item()) for j in range(10)]
                pred_data = {
                    "scene_id": data_dict["scan_id"][i],
                    "question_id": data_dict["data_idx"][i],
                    "answer_top10": answer_top10,
                    "bbox": [0] * 6, # fake box
                }
                self.eval_results.append(pred_data)

        if not include_count:
            for key, v in metrics.items():
                metrics[key] = v[0] / max(v[1], 1)

        return metrics


@EVALUATOR_REGISTRY.register()
class ScanQAGenEval(ScanQAEval):
    def __init__(self, cfg, accelerator, **kwargs):
        super().__init__(cfg, accelerator, **kwargs)

    def batch_metrics(self, data_dict, include_count=False):
        metrics = {}
        answer_preds = data_dict['answer_pred']
        answer_gts = data_dict['answers']
        total_count = len(answer_preds)
        correct = len([1 for pred, gts in zip(answer_preds, answer_gts) if pred in gts])
        metrics['ans1_acc'] = (correct, total_count)

        if not include_count:
            for key, v in metrics.items():
                metrics[key] = v[0] / max(v[1], 1)
        
        if self.save:
            for i in range(total_count):
                answer_preds[i] = 'empty' if answer_preds[i] == '' else answer_preds[i] # avoid empty string
                answer_top10 = [answer_preds[i]] * 10 # currently only support top-1, repeated by 10 times.
                pred_data = {
                    "scene_id": data_dict["scan_id"][i],
                    "question_id": data_dict["data_idx"][i],
                    "sentence": data_dict['sentence'][i],
                    "answer_top10": answer_top10,
                    "gt": answer_gts[i],
                    "bbox": [0] * 6, # fake box
                }
                self.eval_results.append(pred_data)
        
        return metrics