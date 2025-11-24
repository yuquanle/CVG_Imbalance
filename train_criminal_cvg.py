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
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from transformers import AdamW, AutoTokenizer
from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from bert_score import BERTScorer

from utils import *
from model import *
from dataset import CriminalDataset
from torch.utils.tensorboard import SummaryWriter  # 导入tensorboard的类


parser = argparse.ArgumentParser(description='VMask classificer')
# batch 128, gpu 10000M
parser.add_argument('--debug', type=bool, default=False, help='debug')
parser.add_argument('--model_name', type=str, default='RNP', help='model name')
parser.add_argument('--lr', type=float, default=5e-5, help='initial learning rate')
parser.add_argument('--max_len', type=int, default=300, help='max_len')
parser.add_argument('--weight_decay', default=0, type=float, help='adding l2 regularization')
parser.add_argument('--dropout', type=float, default=0.2, help='the probability for dropout')
parser.add_argument('--alpha_rationle', type=float, default=0.2, help='alpha_rationle')
parser.add_argument('--hidden_dim', type=int, default=768, help='number of hidden dimension')
parser.add_argument("--activation", type=str, dest="activation", default="tanh", help='the choice of \
        non-linearity transfer function')
parser.add_argument('--gpu_id', default='0', type=str, help='gpu id')
parser.add_argument('--seed', type=int, default=42, help='random seed')
# parser.add_argument('--types', type=str, default='legal', help='data_type')
parser.add_argument('--epochs', type=int, default=16, help='number of epochs for training')
parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
parser.add_argument('--accumulation_steps', type=int, default=1, help='accumulation steps')
parser.add_argument('--save_path', type=str, default='', help='save_path')
parser.add_argument('--max_input_length', type=int, default=600, help='')
parser.add_argument('--max_target_length', type=int, default=400, help='')
parser.add_argument('--backbone_model_name', type=str, default='', help='backbone_model_name')
parser.add_argument('--tensorboard_summary_log_path', type=str, default='', help='tensorboard_summary_log_path')

args = parser.parse_args()

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO,
                    )
logger = logging.getLogger(__name__)


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
set_random_seed(2022)

args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# args.device = 'cpu'
print(args.device)

for k, v in vars(args).items():
    logger.info("{:20} : {:10}".format(k, str(v)))

# load dataset
train_data_path = '/home/leyuquan/projects/LLMs/CVG/datasets/c3vg_dataset/CJO_train_split_train.json'
eval_data_path ='/home/leyuquan/projects/LLMs/CVG/datasets/c3vg_dataset/CJO_train_split_valid.json'
test_data_path = '/home/leyuquan/projects/LLMs/CVG/datasets/c3vg_dataset/CJO_test.json'
max_input_length = args.max_input_length 
max_target_length = args.max_target_length


 # 定义SummaryWriter，log_dir是日志文件存储路径
summary_writer = SummaryWriter(log_dir=args.tensorboard_summary_log_path)

tokenizer = AutoTokenizer.from_pretrained(os.path.join("/home/leyuquan/projects/LLMs/PLM_Backbones/", args.backbone_model_name))

# load dataset
train_dataset = CriminalDataset(args, tokenizer, train_data_path, max_input_length, max_target_length, data_type='train')
train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=5, drop_last=False,collate_fn = train_dataset.collate_fn)

# show an example of input/output
train_dataset.__show_example__(train_dataset[0])
#print(train_dataset[0])
  
eval_dataset = CriminalDataset(args, tokenizer, eval_data_path, max_input_length, max_target_length, data_type='valid')
eval_dataloader = DataLoader(eval_dataset, batch_size = args.batch_size, shuffle=False, num_workers=5, drop_last=False, collate_fn = eval_dataset.collate_fn)

test_dataset = CriminalDataset(args, tokenizer, test_data_path, max_input_length, max_target_length, data_type='test')
test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, num_workers=5, drop_last=False, collate_fn = test_dataset.collate_fn)

# load model
model = CriminalCourtViewGen(args=args)
model = model.to(args.device)

# load semantic metric model
# BertScore (bert_base_chinese) metric
# add bertsocre path
bert_scorer = BERTScorer(model_type="/home/leyuquan/projects/LegalNLU/PLMs/models--bert-base-chinese/snapshots/84b432f646e4047ce1b5db001d43a348cd3f6bd0/",
                         num_layers=12)


val_max_blue = 0.0
optimizer = AdamW(model.parameters(), lr=args.lr)
total_steps = len(train_dataset) * args.epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
model.train()


for epoch in range(0, args.epochs):
    model.train()

    logger.info("Trianing Epoch: {}/{}".format(epoch, int(args.epochs)))
    for step, batch in enumerate(tqdm(train_dataloader)):
        # if step>2:break
        batch = tuple(v.to(args.device) for k,v in batch.items()) 
        
        source_inut_ids, source_attention_mask, target_input_ids = batch
    
        output = model(source_inut_ids, source_attention_mask, target_input_ids)

        loss = output.loss
        # 记录训练集的step loss数据到tensorboard
        summary_writer.add_scalar('train_loss', loss.item(), epoch+1)
        
        # loss regularization
        loss = loss / args.accumulation_steps
        loss.backward() # 梯度累加，反向传播
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 累加到指定的 steps 后再更新参数
        if (step + 1) % args.accumulation_steps == 0:
            optimizer.step() # 更新参数
            optimizer.zero_grad() # 梯度清零
        
            # 调整学习率
            scheduler.step()

        if step % 1000 == 0:
            print(f"Epoch {epoch}, Step {step}: Loss = {loss.item()}")

    # for each epoch, save bast model for vailid dataset.
    model.eval()

    ##############################  eval  ##############################
    path = './output/valid_results/' + args.backbone_model_name + '_' + args.model_name + '_valid_pred_' + str(epoch) + '.json'
    params = {"max_length": args.max_target_length,
                "do_sample":False,
                "num_beams":3,
                "top_k":5,
                "top_p":0.95,
                "temperature":0.0,
                "repetition_penalty": 1.2
            }
    print("eval generation params: >>>>>",json.dumps(params))

    preds_list, candicate_list = eval_test_prediction(model, eval_dataloader, path, params, args, tokenizer)

    valid_metrics_dict = caluate_matrix(preds_list, candicate_list, tokenizer, bert_scorer)
    logger.info(f"eval_metrics: {valid_metrics_dict}\n")
    bleu_scoren = valid_metrics_dict['bleu_scoren']
    if bleu_scoren > val_max_blue: 
        print(f'save model for {epoch} epoch')
        torch.save(model.state_dict(), args.save_path)
        val_max_blue = bleu_scoren


# 关闭SummaryWriter
summary_writer.close()
print('train finish')