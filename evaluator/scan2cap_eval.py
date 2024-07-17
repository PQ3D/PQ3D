from pathlib import Path
import torch
import numpy as np
from collections import OrderedDict

from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pairwise_cos_sim

from evaluator.build import EVALUATOR_REGISTRY, BaseEvaluator
from evaluator.build import EVALUATOR_REGISTRY
from evaluator.capeval.cider.cider import Cider
from evaluator.capeval.bleu.bleu import Bleu
from evaluator.capeval.meteor.meteor import Meteor
from evaluator.capeval.rouge.rouge import Rouge


@EVALUATOR_REGISTRY.register()
class Scan2CapEval(BaseEvaluator):
    def __init__(self, cfg, accelerator, **kwargs):
        self.target_metric = 'cider'
        self.save_dir = Path(cfg.exp_dir) / "eval_results" / self.__class__.__name__
        super().__init__(cfg, accelerator, **kwargs)

        self.txt_max_len = 30
        self.metric2scorer = {'cider': Cider(), 'bleu': Bleu(), 'meteor': Meteor(), 'rouge': Rouge()}
        self.gts = torch.load(cfg.data.Scan2CapSceneVerse.corpus_path)
        empty_preds = {obj_key: [self.process_caption("")] for obj_key in self.gts.keys()}
        self.empty_scores = {}
        print('CapEval: computing empty scores...')
        for metric_name in ['cider', 'bleu', 'meteor', 'rouge']:
            scores = self.metric2scorer[metric_name].compute_score(self.gts, empty_preds)[1]
            if metric_name == 'bleu':
                scores = scores[-1] # bleu-4
            scores = np.array(scores)
            print(f'{metric_name}: {np.round(scores.mean(), 4)}, ', end='')
            self.empty_scores[metric_name] = dict(zip(self.gts.keys(), scores))
        print()

    def reset(self):
        super().reset()

        self.preds = []
        self.obj_keys = []
        self.iou25_flags = []
        self.iou50_flags = []

    def process_caption(self, sentence):
        if sentence == '':
            return 'sos eos'
        sentence = ' '.join(word_tokenize(sentence)[:self.txt_max_len])
        return f'sos {sentence} eos' 

    def batch_metrics(self, data_dict, include_count=False):
        metrics = {}
        iou25_flags = data_dict['iou25_flag'].cpu().numpy()
        iou50_flags = data_dict['iou50_flag'].cpu().numpy()
        all_flags = np.array([True] * len(iou25_flags))

        preds = [[self.process_caption(text)] for text in data_dict['caption_pred']]
        gts = [self.gts[obj_key] for obj_key in data_dict['obj_key']]
        preds = OrderedDict(enumerate(preds))
        gts = OrderedDict(enumerate(gts))

        for metric_name in ['cider', 'bleu', 'meteor', 'rouge']:
            scores = self.metric2scorer[metric_name].compute_score(gts, preds)[1]
            if metric_name == 'bleu':
                scores = scores[-1] # bleu-4
            scores = np.array(scores)
            empty_scores = self.empty_scores[metric_name]
            empty_scores = np.array([empty_scores[obj_key] for obj_key in data_dict['obj_key']])
            for prefix, flags in zip(['', 'iou25_', 'iou50_'], [all_flags, iou25_flags, iou50_flags]):
                metrics[prefix + metric_name] = (scores[flags].sum() + empty_scores[~flags].sum(), len(flags))

        if self.save:
            item_ids = data_dict['data_idx']
            for i in range(len(item_ids)):
                self.eval_results.append({
                    "scene_id": item_ids[i],
                    "pred": preds[i],
                    "gt": gts[i],
                    "bbox": data_dict['prompt'][i].cpu().numpy().tolist()
                })
        
        return metrics