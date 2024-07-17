import os
import collections
import jsonlines

from data.datasets.sceneverse_base import SceneVerseBase

from ..build import DATASET_REGISTRY
from ..data_utils import is_explicitly_view_dependent

class ReferIt3DSceneVerse(SceneVerseBase):
    def __init__(self, cfg, split, anno_type, dataset_cfg):
        # TODO: hack test split to be the same as val
        if split == 'test':
            split = 'val'
        super().__init__(cfg, 'ScanNet', split)
        self.init_dataset_params(dataset_cfg)
        if self.split == 'train':
            self.pc_type = 'gt'

        self.anno_type = anno_type 
        self.sr3d_plus_aug = dataset_cfg.get('sr3d_plus_aug', False)

        self.lang_data, self.scan_ids = self._load_lang()
        self.init_scan_data()

         # build referit3d count, which is different from scanrefer
        for scan_id in self.scan_ids:
            inst_labels = self.scan_data[scan_id]['inst_labels']
            self.scan_data[scan_id]['label_count_referit3d'] = collections.Counter(
                                    [l for l in inst_labels])

    def __getitem__(self, index):
        data_dict = super().__getitem__(index)
        tgt_object_label = data_dict['tgt_object_label'][0]
        scan_id = data_dict['scan_id']
        data_dict['is_multiple'] = self.scan_data[scan_id]['label_count_referit3d'][tgt_object_label.item()] > 1
        data_dict['is_hard'] = self.scan_data[scan_id]['label_count_referit3d'][tgt_object_label.item()] > 2
        return data_dict

    def get_lang(self, index):
        item = self.lang_data[index]
        item_id = item['item_id']
        scan_id = item['scan_id']
        tgt_object_id = int(item['target_id'])
        tgt_object_name = item['instance_type']
        sentence = item['utterance']
        is_view_dependent = is_explicitly_view_dependent(item['tokens'])

        data_dict = {
            "data_idx": item_id,
            "sentence": sentence,
            "is_view_dependent": is_view_dependent,
        }

        return scan_id, tgt_object_id, tgt_object_name, sentence, data_dict

    def _load_lang(self):
        split_scan_ids = self._load_split(self.cfg, self.split)
        lang_data = []
        scan_ids = set()

        anno_file = os.path.join(self.base_dir, f'ScanNet/annotations/refer/{self.anno_type}.jsonl')
        with jsonlines.open(anno_file, 'r') as _f:
            for item in _f:
                if item['scan_id'] in split_scan_ids and len(item['tokens']) <= 24:
                    scan_ids.add(item['scan_id'])
                    lang_data.append(item)

        if self.sr3d_plus_aug and self.split == 'train':
            anno_file = os.path.join(self.base_dir, 'ScanNet/annotations/refer/sr3d+.jsonl')
            with jsonlines.open(anno_file, 'r') as _f:
                for item in _f:
                    if item['scan_id'] in split_scan_ids and len(item['tokens']) <= 24:
                        scan_ids.add(item['scan_id'])
                        lang_data.append(item)

        return lang_data, scan_ids


@DATASET_REGISTRY.register()
class Sr3DSceneVerse(ReferIt3DSceneVerse):
    def __init__(self, cfg, split):
        print(f'Loading {self.__class__.__name__}')
        dataset_cfg = cfg.data.get(self.__class__.__name__)
        super().__init__(cfg, split, 'sr3d', dataset_cfg)


@DATASET_REGISTRY.register()
class Nr3DSceneVerse(ReferIt3DSceneVerse):
    def __init__(self, cfg, split):
        print(f'Loading {self.__class__.__name__}')
        dataset_cfg = cfg.data.get(self.__class__.__name__)
        super().__init__(cfg, split, 'nr3d', dataset_cfg)