import os
import json
import collections
from pathlib import Path
import torch

from data.data_utils import SQA3DAnswer, SQA_TYPES, clean_answer
from evaluator.build import EVALUATOR_REGISTRY, BaseEvaluator


@EVALUATOR_REGISTRY.register()
class SQA3DEval(BaseEvaluator):
    def __init__(self, cfg, accelerator, **kwargs):
        self.target_metric = 'ans1_acc'
        self.save_dir = Path(cfg.exp_dir) / "eval_results" / self.__class__.__name__
        super().__init__(cfg, accelerator, **kwargs)

        if self.save:
            answer_data = json.load(open(os.path.join(cfg.data.scan_family_base,
                            'annotations/sqa_task/answer_dict.json'), encoding='utf-8'))[0]
            answer_counter = []
            for data in answer_data.keys():
                answer_counter.append(data)
            answer_counter = collections.Counter(sorted(answer_counter))
            answer_cands = answer_counter.keys()
            self.answer_vocab = SQA3DAnswer(answer_cands)

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

        # acc by sqa types
        types = data_dict['sqa_type']
        unique_types = torch.unique(types)
        type_mask = types[:, None] == unique_types
        correct_by_type = (correct1_mask[:, None] & type_mask).sum(dim=0)
        count_by_type = type_mask.sum(dim=0)
        for type, correct, count in zip(unique_types, correct_by_type, count_by_type):
            metrics[SQA_TYPES[type.item()]] = (correct.item(), count.item())

        for key in metrics:
            if isinstance(metrics[key], tuple):
                # already has count
                continue
            metrics[key] = (metrics[key], total_count)

        if self.save:
            for i in range(total_count):
                answer_top10 = [self.answer_vocab.itos(choice_10[i, j].item()) for j in range(10)]
                self.eval_results.append({
                    "answer_top10": answer_top10,
                    "scene_id": data_dict['scan_id'][i],
                    "question_id": data_dict['data_idx'][i],
                    "sentence": data_dict['sentence'][i],
                })

        if not include_count:
            for key, v in metrics.items():
                metrics[key] = v[0] / max(v[1], 1)

        return metrics


def answer_match(pred, gts):
    for gt in gts:
        if pred == gt:
            return True
        elif ''.join(pred.split()) in ''.join(gt.split()):
            return True
        elif ''.join(gt.split()) in ''.join(pred.split()):
            return True
    return False

@EVALUATOR_REGISTRY.register()
class SQA3DGenEval(SQA3DEval):
    def __init__(self, cfg, accelerator, **kwargs):
        super().__init__(cfg, accelerator, **kwargs)

    def batch_metrics(self, data_dict, include_count=False):
        metrics = {}
        answer_preds = [clean_answer(a) for a in data_dict['answer_pred']]
        answer_gts = [list(map(clean_answer, a)) for a in data_dict['answers']]
        total_count = len(answer_preds)
        correct1_mask = torch.tensor([answer_match(pred, gts) for pred, gts in zip(answer_preds, answer_gts)]).to(bool)
        correct = correct1_mask.sum().item()

        metrics['ans1_acc'] = (correct, total_count)
        
        types = data_dict['sqa_type'].cpu()
        unique_types = torch.unique(types)
        type_mask = types[:, None] == unique_types
        correct_by_type = (correct1_mask[:, None] & type_mask).sum(dim=0)
        count_by_type = type_mask.sum(dim=0)
        for type, correct, count in zip(unique_types, correct_by_type, count_by_type):
            metrics[SQA_TYPES[type.item()]] = (correct.item(), count.item())

        if not include_count:
            for key, v in metrics.items():
                metrics[key] = v[0] / max(v[1], 1)

        if self.save:
            for i in range(total_count):
                self.eval_results.append({
                    "scene_id": data_dict['scan_id'][i],
                    "question_id": data_dict['data_idx'][i],
                    "sentence": data_dict['sentence'][i],
                    "pred": answer_preds[i],
                    "gt": answer_gts[i],
                })
        
        return metrics