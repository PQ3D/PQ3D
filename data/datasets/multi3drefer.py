import os
import json

from data.datasets.sceneverse_base import SceneVerseBase

from ..build import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class Multi3DReferSceneVerse(SceneVerseBase):
    def __init__(self, cfg, split):
        # TODO: hack test split to be the same as val
        if split == 'test':
            split = 'val'
        super().__init__(cfg, 'ScanNet', split)
        dataset_cfg = cfg.data.get(self.__class__.__name__)
        self.init_dataset_params(dataset_cfg)
        if self.split == 'train':
            self.pc_type = 'gt'
        print(f'Loading {self.__class__.__name__}')
        self.lang_data, self.scan_ids = self._load_lang()
        self.init_scan_data()

    def get_lang(self, index):
        item = self.lang_data[index]
        scan_id = item['scene_id']
        tgt_object_id = item['object_ids']
        tgt_object_name = [item['object_name'].replace('_', ' ')] * len(tgt_object_id)
        sentence = item['description']

        data_dict = {
            "data_idx": scan_id,
            "sentence": sentence,
            "eval_type": item["eval_type"],
        }

        return scan_id, tgt_object_id, tgt_object_name, sentence, data_dict

    def _load_lang(self):
        anno_file = os.path.join(self.aux_dir, f'ScanNet/annotations/multi3drefer_{self.split}.json')
        lang_data = json.load(open(anno_file))
        scan_ids = set(x['scene_id'] for x in lang_data)

        if self.cfg.debug.flag and self.cfg.debug.debug_size != -1:
            scan_ids = sorted(list(scan_ids))[:self.cfg.debug.debug_size]
            lang_data = [x for x in lang_data if x['scene_id'] in scan_ids]
        
        return lang_data, scan_ids