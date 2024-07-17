import os
import jsonlines

from data.datasets.sceneverse_base import SceneVerseBase

from ..build import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class ScanReferSceneVerse(SceneVerseBase):
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
        item_id = item['item_id']
        scan_id = item['scan_id']
        tgt_object_id = int(item['target_id'])
        tgt_object_name = item['instance_type']
        sentence = item['utterance']
        obj_key = f"{scan_id}|{tgt_object_id}|{tgt_object_name}" # used to group the captions for a single object

        data_dict = {
            "data_idx": item_id,
            "sentence": sentence,
            "obj_key": obj_key
        }

        return scan_id, tgt_object_id, tgt_object_name, sentence, data_dict

    def _load_lang(self):
        split_scan_ids = self._load_split(self.cfg, self.split)
        lang_data = []
        scan_ids = set()

        anno_file = os.path.join(self.base_dir, 'ScanNet/annotations/refer/scanrefer.jsonl')
        with jsonlines.open(anno_file, 'r') as _f:
            for item in _f:
                if item['scan_id'] in split_scan_ids:
                    scan_ids.add(item['scan_id'])
                    lang_data.append(item)

        return lang_data, scan_ids