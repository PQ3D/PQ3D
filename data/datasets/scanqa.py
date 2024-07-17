import collections
import os
import json
import torch

from data.datasets.sceneverse_base import SceneVerseBase

from ..build import DATASET_REGISTRY
from ..data_utils import ScanQAAnswer

@DATASET_REGISTRY.register()
class ScanQASceneVerse(SceneVerseBase):
    def __init__(self, cfg, split):
        dataset_cfg = cfg.data.get(self.__class__.__name__)
        if dataset_cfg.get("use_val_for_test", False) and split == 'test':
            split = 'val'
        super().__init__(cfg, 'ScanNet', split)
        self.init_dataset_params(dataset_cfg)
        if self.split == 'train':
            self.pc_type = 'gt'
 
        self.is_test = self.split == 'test'
        self.use_unanswer = dataset_cfg.split.use_unanswer if hasattr(dataset_cfg, split) else True
        self.use_val_for_train = dataset_cfg.get('use_val_for_train', False)

        print(f'Loading {self.__class__.__name__}')
        self.lang_data, self.scan_ids = self._load_lang()
        self.init_scan_data()

    def get_lang(self, index):
        item = self.lang_data[index]
        item_id = item['question_id']
        scan_id = item['scene_id']
        question = item['question']
        tgt_object_id = [] if self.is_test else item['object_ids']
        tgt_object_name = [] if self.is_test else item['object_names']
        answer_list = [''] if self.is_test else item['answers'] # cannot be empty
        answer_id_list = [self.answer_vocab.stoi(answer) 
                            for answer in answer_list if self.answer_vocab.stoi(answer) >= 0]
        
        answer_label = torch.zeros(self.num_answers).long()
        for id in answer_id_list:
            answer_label[id] = 1
        
        sentence = question
        data_dict = {
            "data_idx": item_id,
            "question": question,
            "sentence": sentence,
            "answers": answer_list,
            "answer_label": answer_label,
        }

        return scan_id, tgt_object_id, tgt_object_name, sentence, data_dict

    def _load_lang(self):
        self.num_answers, self.answer_vocab, self.answer_cands = self.build_answer()
        
        lang_data = []
        scan_ids = set()

        if self.split == 'test':
            json_data = []
            for type in ['w_obj', 'wo_obj']:
                anno_file = os.path.join(self.base_dir, f'ScanNet/annotations/qa/ScanQA_v1.0_{self.split}_{type}.json')
                json_data.extend(json.load(open(anno_file, 'r', encoding='utf-8')))
        else:
            anno_file = os.path.join(self.base_dir, f'ScanNet/annotations/qa/ScanQA_v1.0_{self.split}.json')
            json_data = json.load(open(anno_file, 'r', encoding='utf-8'))
        if self.split == 'train' and self.use_val_for_train:
            anno_file = os.path.join(self.base_dir, f'ScanNet/annotations/qa/ScanQA_v1.0_val.json')
            val_json_data = json.load(open(anno_file, 'r', encoding='utf-8'))
            json_data.extend(val_json_data)
        for item in json_data:
            if self.use_unanswer or (len(set(item['answers']) & set(self.answer_cands)) > 0):
                scan_ids.add(item['scene_id'])
                lang_data.append(item)
        print(f'{self.split} unanswerable question {len(json_data) - len(lang_data)},'
              + f'answerable question {len(lang_data)}')
        
        if self.cfg.debug.flag and self.cfg.debug.debug_size != -1:
            scan_ids = sorted(list(scan_ids))[:self.cfg.debug.debug_size]
            lang_data = [item for item in lang_data if item['scene_id'] in scan_ids]

        return lang_data, scan_ids

    def build_answer(self):
        train_data = json.load(open(os.path.join(self.base_dir,
                                'ScanNet/annotations/qa/ScanQA_v1.0_train.json'), encoding='utf-8'))
        answer_counter = sum([data['answers'] for data in train_data], [])
        answer_counter = collections.Counter(sorted(answer_counter))
        num_answers = len(answer_counter)
        answer_cands = answer_counter.keys()
        answer_vocab = ScanQAAnswer(answer_cands)
        print(f"total answers is {num_answers}")
        return num_answers, answer_vocab, answer_cands
