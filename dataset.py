import torch
import json
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import logging


class CriminalDataset(Dataset):
    def __init__(self, args, tokenizer, data_path, max_input_length, max_target_length, data_type):
        self.args = args
        self.data_type = data_type
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.data = self.load_data(data_path)
       
    def load_data(self, data_path):
        data = []
        f = open(data_path,'r',encoding='utf8')
     
        for line in f:
            line = json.loads(line)   
            data.append(line)
        return data

    def __show_example__(self, example):
        input_ids = example['input_ids']
        labels = example['label']
        logging.info('inputs: {}'.format(self.tokenizer.decode(input_ids, add_special_tokens=True))) 
        logging.info('outputs: {}'.format(self.tokenizer.decode(labels, add_special_tokens=True))) 
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx]
        fact = line['fact'].strip('\n')
        charge = line['charge']
        view = line['court_view'].strip('\n')
        
        # vanilla: input=fact, Label-conditioned: input=charge [SEP] fact
        if self.args.model_name == 'vanilla':
            __input = self.tokenizer(
                fact,
                padding=True, 
                max_length=self.max_input_length,
                truncation=True, 
                return_tensors="pt"
            )
        elif self.args.model_name == 'label_cond':
            __input = self.tokenizer(
                "罪名: "  + charge + "[SEP]" + "基本事实: "+ fact,
                padding=True, 
                max_length=self.max_input_length + 20,
                truncation=True, 
                return_tensors="pt"
            )
        else:
            raise NotImplementedError
        
        with self.tokenizer.as_target_tokenizer():
            if self.data_type == "train":
                __input['label'] = self.tokenizer(view, max_length=self.max_target_length, padding=True, truncation=True, return_tensors="pt")['input_ids']
            # 验证、测试时，要用全部label
            elif self.data_type in ["valid", "test"]:
                __input['label'] = self.tokenizer(view, padding=True, truncation=False, return_tensors="pt")['input_ids']
            
        end_token_index = torch.where(__input['label'] == self.tokenizer.eos_token_id)[1]
        for idx_, end_idx in enumerate(end_token_index):
            __input['label'][idx_][end_idx+1:] = -100

        __input['input_ids'] = __input['input_ids'].squeeze()
        __input['attention_mask'] = __input['attention_mask'].squeeze()
        __input['label'] = __input['label'].squeeze()
        return __input

    def collate_fn(self, batch):
        input_ids  = [torch.tensor(f['input_ids']) for f in batch]
        attention_mask  = [torch.tensor(f['attention_mask']) for f in batch]
        label = [torch.tensor(f['label']) for f in batch]
       
        # 将文本序列填充至同一长度
        padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        padded_attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        padded_labels = pad_sequence(label, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        
        return dict(input_ids = padded_input_ids,
                attention_mask = padded_attention_mask,
                labels = padded_labels,
                )
        