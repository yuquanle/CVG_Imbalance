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
from utils import *
from model import *
from torch.utils.data import TensorDataset, DataLoader
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import prettytable as pt
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge


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
parser.add_argument('--save_path', type=str, default='', help='save_path')
parser.add_argument('--max_target_length', type=int, default=15, help='save_path')


args = parser.parse_args()

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO,
                    )
logger = logging.getLogger(__name__)


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id


def set_random_seed(seed):
    """
    设置随机种子

    Args:
        seed (int): 随机种子
    """
    # 设置Python的随机种子
    random.seed(seed)

    # 设置NumPy的随机种子
    np.random.seed(seed)

    # 设置PyTorch的随机种子
    torch.manual_seed(seed)

    # 如果使用GPU，还需要设置以下随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 如果使用CUDNN库，还需要设置以下随机种子
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_random_seed(2022)

args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# args.device = 'cpu'
print(args.device)
process = Data_Process(args)


for k, v in vars(args).items():
    logger.info("{:20} : {:10}".format(k, str(v)))



def  caluate_matrix(preds_list,candicate_list):
    rouge_score1_ = []
    rouge_score2_ = []
    rouge_scorel_ = []

    bleu_score1_ = []
    bleu_score2_ = []
    bleu_score3_ = []
    bleu_score4_ = []
    rouge = Rouge()
    for pred,candicate in zip(preds_list,candicate_list):
        rouge_score = rouge.get_scores(pred, candicate)
        rouge_score1_.append(rouge_score[0]["rouge-1"]['f'])
        rouge_score2_.append(rouge_score[0]["rouge-2"]['f'])
        rouge_scorel_.append(rouge_score[0]["rouge-l"]['f'])


        reference = [candicate.split(' ')]
        pred = pred.split(' ')
        bleu_score1_.append(sentence_bleu(reference, pred, weights=(1, 0, 0, 0)))
        bleu_score2_.append(sentence_bleu(reference, pred, weights=(0.5, 0.5, 0, 0)))
        bleu_score3_.append(sentence_bleu(reference, pred, weights=(0.33, 0.33, 0.33, 0)))
        bleu_score4_.append(sentence_bleu(reference, pred, weights=(0.25, 0.25, 0.25, 0.25)))

    bleu_score1 = np.mean(bleu_score1_)
    bleu_score2 = np.mean(bleu_score2_)
    bleu_score3 = np.mean(bleu_score3_)
    bleu_score4 = np.mean(bleu_score4_)
    rouge_score1 = np.mean(rouge_score1_)
    rouge_score2 = np.mean(rouge_score2_)
    rouge_scorel = np.mean(rouge_scorel_)

    logger.info(f"test_BLEU1: {bleu_score1:>0.4f}\n")
    logger.info(f"test_BLEU2: {bleu_score2:>0.4f}\n")
    logger.info(f"test_BLEU3: {bleu_score3:>0.4f}\n")
    logger.info(f"test_BLEU4: {bleu_score4:>0.4f}\n")
    logger.info("test_BLEUN: {}".format(np.mean([bleu_score1,bleu_score2,bleu_score3,bleu_score4])))

    logger.info("test_ruoge1: {}\n".format(np.mean(rouge_score1)))
    logger.info("test_ruoge2: {}\n".format(np.mean(rouge_score2)))
    logger.info("test_ruogeL: {}\n".format(np.mean(rouge_scorel)))
    return bleu_score1, bleu_score2, bleu_score3,bleu_score4,rouge_score1, rouge_score2, rouge_scorel 



model = eval(args.model_name)()
train_dataset = process.process_data('train', model.model, args.debug)
print(len(train_dataset))
train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=5, drop_last=False)

eval_dataset = process.process_data('eval',model.model, args.debug)
print(len(eval_dataset))
eval_dataloader = DataLoader(eval_dataset, batch_size = args.batch_size, shuffle=False, num_workers=0, drop_last=False)

model = model.to(args.device)

val_max_blue = 0.0
optimizer = AdamW(model.parameters(), lr=args.lr)
total_steps = len(train_dataset) * args.epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
model.train()

BEST_MODEL_PATH = args.save_path+'_'+args.model_name.lower()+'_best.pth'

for epoch in range(0, args.epochs):
    model.train()

    logger.info("Trianing Epoch: {}/{}".format(epoch, int(args.epochs)))
    for step,batch in enumerate(tqdm(train_dataloader)):
        # if step>2:break
        batch = tuple(t.to(args.device) for t in batch) 
        
        source_inut_ids, source_attention_mask, target_input_ids, decoder_input_ids  = batch
    
        optimizer.zero_grad()
        
        output = model(source_inut_ids, source_attention_mask, target_input_ids, decoder_input_ids)

        loss = output.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            print(f"Epoch {epoch}, Step {step}: Loss = {loss.item()}")




    model.eval()

    ##############################  eval  ##############################
    preds_list, candicate_list = [], []
    path = './output/result' + str(epoch) + '.json'
    f_result = open(path,'w')
    for step,batch in enumerate(tqdm(eval_dataloader)):
        batch = tuple(t.to(args.device) for t in batch)
        source_input_ids, source_attention_mask, target_input_ids, decoder_input_ids = batch
        with torch.no_grad():
            view_generated_tokens = model.model.generate(source_input_ids,max_length=300,num_beams=5).cpu().numpy()

        label_tokens = target_input_ids.cpu().numpy()   


        view_decoded_preds = process.tokenizer.batch_decode(view_generated_tokens, skip_special_tokens=True)
        decoded_labels = process.tokenizer.batch_decode(label_tokens, skip_special_tokens=True)
        preds_list += [pred.strip() for pred in view_decoded_preds]
        candicate_list += [label.strip() for label in decoded_labels]
    
    for p_, c_ in zip(preds_list, candicate_list):
        result = {}
        result['preds'] = p_
        result['candicate'] = c_
        json_str = json.dumps(result, ensure_ascii=False)
        f_result.write(json_str + "\n")
    f_result.close()


    bleu_score1, bleu_score2, bleu_score3,bleu_score4,rouge_score1, rouge_score2, rouge_scorel = caluate_matrix(preds_list, candicate_list)

    PATH = args.save_path+args.model_name.lower()+'_'+str(epoch) + '.pth'
    torch.save(model.state_dict(), PATH)
    
    if bleu_score1 + bleu_score2 + bleu_score3  + bleu_score4 + rouge_score1 + rouge_score2 + rouge_scorel > val_max_blue: 
        torch.save(model.state_dict(), BEST_MODEL_PATH)



# 加载最佳模型
model.load_state_dict(torch.load(BEST_MODEL_PATH))

test_dataset = process.process_data('test', model.model, args.debug)
print(len(test_dataset))
test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, num_workers=0, drop_last=False)


# 在测试集上进行预测
with torch.no_grad():
    preds_list, candicate_list = [], []
    path = './output/result_test.json'
    f_result = open(path,'w')
    for step,batch in enumerate(tqdm(test_dataloader)):
        batch = tuple(t.to(args.device) for t in batch)
        source_input_ids, source_attention_mask, target_input_ids, decoder_input_ids = batch
        with torch.no_grad():
            view_generated_tokens = model.model.generate(source_input_ids,max_length=300,num_beams=5).cpu().numpy()

        label_tokens = target_input_ids.cpu().numpy()   


        view_decoded_preds = process.tokenizer.batch_decode(view_generated_tokens, skip_special_tokens=True)
        decoded_labels = process.tokenizer.batch_decode(label_tokens, skip_special_tokens=True)
        preds_list += [pred.strip() for pred in view_decoded_preds]
        candicate_list += [label.strip() for label in decoded_labels]
    
    for p_, c_ in zip(preds_list, candicate_list):
        result = {}
        result['preds'] = p_
        result['candicate'] = c_
        json_str = json.dumps(result, ensure_ascii=False)
        f_result.write(json_str + "\n")
    f_result.close()

    bleu_score1, bleu_score2, bleu_score3,bleu_score4,rouge_score1, rouge_score2, rouge_scorel = caluate_matrix(preds_list, candicate_list)





