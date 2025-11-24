import logging
import os
import argparse
import random
from tqdm import tqdm, trange
import csv
import glob 
import json
from sklearn import metrics
import numpy as np
import torch
import os
import sys
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from transformers import AdamW, AutoTokenizer
from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from utils import *
from model import *
from dataset import CriminalDataset
# 设置最大递归深度，避免计算rouge-L报错
sys.setrecursionlimit(2100 * 210 + 10)


parser = argparse.ArgumentParser(description='VMask classificer')
# batch 128, gpu 10000M
parser.add_argument('--debug', type=bool, default=False, help='debug')
parser.add_argument('--dataset', type=str, default='CJO')
parser.add_argument('--model_name', type=str, default='RNP', help='model name')
parser.add_argument('--gpu_id', default='0', type=str, help='gpu id')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
parser.add_argument('--model_path', type=str, default='', help='model_path')
parser.add_argument('--max_input_length', type=int, default=800, help='max_input_length')
parser.add_argument('--max_target_length', type=int, default=200, help='max_target_length')
parser.add_argument('--backbone_model_name', type=str,  help='backbone_model_name')


args = parser.parse_args()

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO,
                    )
logger = logging.getLogger(__name__)

set_random_seed(args.seed)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(args.device)

test_data_path = '/home/leyuquan/projects/LLMs/CVG/datasets/c3vg_dataset/CJO_test.json'
max_input_length = args.max_input_length 
max_target_length = args.max_target_length
args.backbone_model_path = os.path.join("/home/leyuquan/projects/LLMs/PLM_Backbones", args.backbone_model_name)

for k, v in vars(args).items():
    logger.info("{:20} : {:10}".format(k, str(v)))

tokenizer = AutoTokenizer.from_pretrained(args.backbone_model_path)
# load model
model = CriminalCourtViewGen(args)

test_dataset = CriminalDataset(args, tokenizer, test_data_path, max_input_length, max_target_length, data_type="test")
test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, num_workers=5, drop_last=False, collate_fn = test_dataset.collate_fn)
print(test_dataset[0])

model = model.to(args.device)

# 加载最佳模型
model.load_state_dict(torch.load(args.model_path))
# 在测试集上进行预测
model.eval()

# load evaluate model
bert_model_path = "/home/leyuquan/projects/LLMs/PLM_Backbones/bert-base-chinese/snapshots/84b432f646e4047ce1b5db001d43a348cd3f6bd0/"

# BertScorer metrics (using all 12 layer of bert-base-chinese)
bert_scorer = BERTScorer(model_type=bert_model_path,
                         num_layers=12,
                         device=args.device)

def test_bart_model(model, params):
    
    file_name = './output/test_pred_results/' + args.backbone_model_name + '_' + args.model_name + '_test_result_' + str(args.max_target_length) + ".json"
    print(file_name)
    
    predictions_list, references_list = eval_test_prediction(model,test_dataloader, file_name, params, args, tokenizer)

    metrics_dict = calculate_metrics(
        predictions_list=predictions_list, 
        references_list=references_list, 
        bert_scorer=bert_scorer,
        )
    logger.info(f"test_BLEU1: {metrics_dict['bleu_score1']:>0.4f}")
    logger.info(f"test_BLEU2: {metrics_dict['bleu_score2']:>0.4f}")
    logger.info(f"test_BLEU3: {metrics_dict['bleu_score3']:>0.4f}")
    logger.info(f"test_BLEU4: {metrics_dict['bleu_score4']:>0.4f}")
    logger.info(f"test_BLEUN: {metrics_dict['bleu_scoren']:>0.4f}")
    logger.info(f"test_ruoge1: {metrics_dict['rouge_score1']:>0.4f}")
    logger.info(f"test_ruoge2: {metrics_dict['rouge_score2']:>0.4f}")
    logger.info(f"test_ruogeL: {metrics_dict['rouge_scoreL']:>0.4f}")
    logger.info(f"test_bert_score_P: {metrics_dict['bertscore_P']:>0.4f}")
    logger.info(f"test_bert_score_R: {metrics_dict['bertscore_R']:>0.4f}")
    logger.info(f"test_bert_score_F1: {metrics_dict['bertscore_F1']:>0.4f}")
    
params = {'max_length': args.max_target_length, 'do_sample': False, 'num_beams': 2}

test_bart_model(model, params)
