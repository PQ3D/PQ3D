import torch
from torch.utils.data import Dataset, default_collate
from transformers import AutoTokenizer

from .dataset_wrapper import DATASETWRAPPER_REGISTRY


@DATASETWRAPPER_REGISTRY.register()
class ScanFamilyDatasetWrapper(Dataset):
    def __init__(self, cfg, dataset):
        self.dataset = dataset
        tokenizer_name = getattr(cfg.data_wrapper, 'tokenizer', 'bert-base-uncased')
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenized_txt = self.tokenize_txt()
    
    def tokenize_txt(self):
        encoded_input = self.tokenizer([self.dataset.get_lang(i)[-1]['sentence'] for i in range(len(self.dataset))], 
                                       add_special_tokens=True, truncation=True)
        return encoded_input.input_ids

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data_dict = self.dataset[idx]
        data_dict['txt_ids'] = torch.LongTensor(self.tokenized_txt[idx])
        data_dict['txt_masks'] = torch.ones((len(self.tokenized_txt[idx]))).bool()
        return data_dict
    
    def collate_fn(self, batch):
        new_batch = {}
        padding_keys = [k for k, v in batch[0].items() if isinstance(v, torch.Tensor) and v.ndim > 0]
        for k in padding_keys:
            tensors = [sample.pop(k) for sample in batch]
            padding_value = -100 if k == 'obj_labels' else 0
            padded_tensor = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=padding_value)
            new_batch[k] = padded_tensor.to(tensors[0].dtype)
        
        list_keys = [k for k, v in batch[0].items() if isinstance(v, list)]
        for k in list_keys:
            new_batch[k] = [sample.pop(k) for sample in batch]
        
        new_batch.update(default_collate(batch))

        return new_batch
