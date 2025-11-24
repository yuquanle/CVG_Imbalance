from transformers import BartForConditionalGeneration,BartPretrainedModel, BertTokenizer,MT5ForConditionalGeneration, T5ForConditionalGeneration
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.distributions.bernoulli import Bernoulli
import numpy as np
import datetime
import logging
import os


class CriminalCourtViewGen(nn.Module):
    """
    label_cond: CGED
    """
    def __init__(self, args=None):
        super(CriminalCourtViewGen, self).__init__()
        self.args = args
        self.model_name = args.model_name
        logging.info(f'backbone model name: {args.backbone_model_name}')
        logging.info(f'model name: {args.model_name}')
        
        backbone_model_path = os.path.join("/home/leyuquan/projects/LLMs/PLM_Backbones/", args.backbone_model_name)
        
        if self.args.backbone_model_name == 'bart-base-chinese':
            if self.model_name in ['vanilla', 'label_cond']:
                self.model = BartForConditionalGeneration.from_pretrained(backbone_model_path, return_dict=True)
            else:
                raise NotImplementedError
        elif self.args.backbone_model_name == 'mengzi-t5-base':
            if self.model_name in ['vanilla', 'label_cond']:
                self.model = T5ForConditionalGeneration.from_pretrained(backbone_model_path)
            else:
                raise NotImplementedError
        elif self.args.backbone_model_name in ['mt5base', 'Randeng-T5-784M']:    
            if self.model_name in ['vanilla', 'label_cond']:
                self.model = MT5ForConditionalGeneration.from_pretrained(backbone_model_path)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        self.model = self.model.to(args.device)

    def forward(self, source_input_ids, source_attention_mask, target_input_ids):
        if self.model_name in ['vanilla', 'label_cond']:
            output = self.model(input_ids = source_input_ids, attention_mask = source_attention_mask, labels = target_input_ids)
        else:
            raise NotImplementedError
        
        return output