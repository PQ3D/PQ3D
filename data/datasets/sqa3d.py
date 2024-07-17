import collections
import copy
import os
import json
import random
import torch

from data.datasets.sceneverse_base import SceneVerseBase

from ..build import DATASET_REGISTRY
from ..data_utils import SQA3DAnswer, get_sqa_question_type


@DATASET_REGISTRY.register()
class SQA3DSceneVerse(SceneVerseBase):
    r"""
    questions json file: dict_keys(['info', 'license', 'data_type', 'data_subtype', 'task_type', 'questions'])
        'questions': List
        'questions'[0]: {
            'scene_id': 'scene0050_00',
            'situation': 'I am standing by the ottoman on my right facing a couple of toolboxes.',
            'alternative_situation': [
                'I just placed two backpacks on the ottoman on my right side before I went to play the piano in front of me to the right.',
                'I stood up from the ottoman and walked over to the piano ahead of me.'
            ],
            'question': 'What instrument in front of me is ebony and ivory?',
            'question_id': 220602000002
        }

    annotations json file: dict_keys(['info', 'license', 'data_type', 'data_subtype', 'annotations'])
        'annotations': List
        'annotations'[0]: {
            'scene_id': 'scene0050_00',
            'question_type': 'N/A',
            'answer_type': 'other',
            'question_id': 220602000002,
            'answers': [{'answer': 'piano', 'answer_confidence': 'yes', 'answer_id': 1}],
            'rotation': {'_x': 0, '_y': 0, '_z': -0.9995736030415032, '_w': -0.02919952230128897},
            'position': {'x': 0.7110268899979686, 'y': -0.03219739162793617, 'z': 0}
        }
    """
    def __init__(self, cfg, split):
        super().__init__(cfg, 'ScanNet', split)

        dataset_cfg = cfg.data.get(self.__class__.__name__)
        self.init_dataset_params(dataset_cfg)
        if self.split == 'train':
            self.pc_type = 'gt'
        self.use_unanswer = dataset_cfg.split.use_unanswer if hasattr(dataset_cfg, split) else True
        self.use_val_for_train = dataset_cfg.get('use_val_for_train', False)

        print(f'Loading {self.__class__.__name__}')
        self.lang_data, self.scan_ids = self._load_lang()
        self.init_scan_data()

    def get_lang(self, index):
        item = self.lang_data[index]
        item_id = item['question_id']
        scan_id = item['scene_id']
        tgt_object_id = []
        tgt_object_name = []
        answer_list = [answer['answer'] for answer in item['answers']]
        answer_id_list = [self.answer_vocab.stoi(answer)
                          for answer in answer_list if self.answer_vocab.stoi(answer) >= 0]
        situation = item['situation']
        question = item['question']
        sentence = situation + ' ' + question
        question_type = get_sqa_question_type(question)

        answer_label = torch.zeros(self.num_answers).long()
        for id in answer_id_list:
            answer_label[id] = 1

        data_dict = {
            "data_idx": str(item_id),
            "question": question,
            "answers": answer_list,
            "answer_label": answer_label,
            "situation": situation,
            "situation_pos": item['position'],
            "situation_rot": item['rotation'],
            "sentence": sentence,
            "sqa_type": question_type,
        }

        return scan_id, tgt_object_id, tgt_object_name, sentence, data_dict

    def _load_lang(self):
        self.num_answers, self.answer_vocab, self.answer_cands = self.build_answer()
        
        lang_data = []
        scan_ids = set()

        anno_file = os.path.join(self.base_dir,
            f'ScanNet/annotations/sqa_task/balanced/v1_balanced_sqa_annotations_{self.split}_scannetv2.json')
        json_data = json.load(open(anno_file, 'r', encoding='utf-8'))['annotations']
        if self.split == 'train' and self.use_val_for_train:
            anno_file = os.path.join(self.base_dir,
                f'ScanNet/annotations/sqa_task/balanced/v1_balanced_sqa_annotations_val_scannetv2.json')
            val_json_data = json.load(open(anno_file, 'r', encoding='utf-8'))['annotations']
            json_data.extend(val_json_data)
        for item in json_data:
            if self.use_unanswer or (len(set(item['answers']) & set(self.answer_cands)) > 0):
                scan_ids.add(item['scene_id'])
                lang_data.append(item)
        print(f'{self.split} unanswerable question {len(json_data) - len(lang_data)},'
              + f'answerable question {len(lang_data)}')

        # add question and situation to item
        questions_map = self._load_question()
        lang_data_with_alternative_situation = []
        for item in lang_data:
            scan_id = item['scene_id']
            item_id = item['question_id']
            situations = questions_map[scan_id][item_id]['situation']
            question = questions_map[scan_id][item_id]['question']
            item['situation'] = situations[0]
            item['question'] = question
            for situation in situations[1:]:
                item = copy.deepcopy(item)
                item['situation'] = situation
                lang_data_with_alternative_situation.append(item)
        
        if self.split == 'train':
            lang_data = lang_data + lang_data_with_alternative_situation

        if self.cfg.debug.flag and self.cfg.debug.debug_size != -1:
            scan_ids = sorted(list(scan_ids))[:self.cfg.debug.debug_size]
            lang_data = [item for item in lang_data if item['scene_id'] in scan_ids]

        return lang_data, scan_ids

    def build_answer(self):
        answer_data = json.load(
            open(os.path.join(self.base_dir,
                              'ScanNet/annotations/sqa_task/answer_dict.json'), encoding='utf-8')
            )[0]
        answer_counter = []
        for data in answer_data.keys():
            answer_counter.append(data)
        answer_counter = collections.Counter(sorted(answer_counter))
        num_answers = len(answer_counter)
        answer_cands = answer_counter.keys()
        answer_vocab = SQA3DAnswer(answer_cands)
        print(f"total answers is {num_answers}")
        return num_answers, answer_vocab, answer_cands

    def _load_question(self):
        questions_map = {}
        anno_file = os.path.join(self.base_dir, 
            f'ScanNet/annotations/sqa_task/balanced/v1_balanced_questions_{self.split}_scannetv2.json')
        json_data = json.load(open(anno_file, 'r', encoding='utf-8'))['questions']
        if self.split == 'train' and self.use_val_for_train:
            anno_file = os.path.join(self.base_dir, 
                f'ScanNet/annotations/sqa_task/balanced/v1_balanced_questions_val_scannetv2.json')
            val_json_data = json.load(open(anno_file, 'r', encoding='utf-8'))['questions']
            json_data.extend(val_json_data)
        for item in json_data:
            if item['scene_id'] not in questions_map.keys():
                questions_map[item['scene_id']] = {}
            questions_map[item['scene_id']][item['question_id']] = {
                'situation': [item['situation']] + item['alternative_situation'],   # list of sentences
                'question': item['question']   # sentence
            }

        return questions_map
