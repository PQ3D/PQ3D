from ..build import DATASET_REGISTRY
from .scanrefer import ScanReferSceneVerse

@DATASET_REGISTRY.register()
class Scan2CapSceneVerse(ScanReferSceneVerse):
    def __init__(self, cfg, split):
        super().__init__(cfg, split)

        dataset_cfg = cfg.data.get(self.__class__.__name__)
        txt_max_len = dataset_cfg.get('txt_max_len', 35)
        if split != 'train':
            # for evaluation, we only need to keep one item for each unique object
            self.lang_data = self.get_unique_obj()
        for item in self.lang_data:
            item['utterance'] = ' '.join(item['tokens'][:txt_max_len])
    
    def get_unique_obj(self):
        included_objs = set()
        unique_objs = []
        for item in self.lang_data:
            scan_id =  item['scan_id']
            tgt_object_id = int(item['target_id'])
            tgt_object_name = item['instance_type']
            obj_key = f"{scan_id}|{tgt_object_id}|{tgt_object_name}"
            if obj_key not in included_objs:
                unique_objs.append(item)
                included_objs.add(obj_key)
        return unique_objs

    def __getitem__(self, index):
        data_dict = super().__getitem__(index)
        data_dict['iou25_flag'] = len(data_dict['tgt_object_id_iou25']) > 0
        data_dict['iou50_flag'] = len(data_dict['tgt_object_id_iou50']) > 0
        return data_dict