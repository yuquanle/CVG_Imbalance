import logging
import os
import argparse
import random
import time
from tqdm import tqdm, trange
import csv
import pickle
import json
import numpy as np
import torch
import os
from transformers import AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import warnings
# Disable all warnings
warnings.filterwarnings("ignore")
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from  utils import *
import sys
# 设置最大递归深度，避免计算rouge-L报错
sys.setrecursionlimit(2100 * 210 + 10)


from bert_score import BERTScorer


def truncate_and_convert_back(text, tokenizer, max_length):
    # 使用tokenizer对文本进行编码和截断
    encoded = tokenizer.encode(text, max_length=max_length, truncation=True, return_tensors='pt')
    
    # 将截断后的编码转换回文本
    converted_text = tokenizer.decode(encoded[0], skip_special_tokens=True)
    
    return converted_text

def check_gt_court_view(data_path, references_list, tokenizer):
    '''核对预测文件中的reference_view是否与测试集中的ground truth一致'''
    f = open(data_path,'r',encoding='utf8')
    for line, ref_court_view in zip(f, references_list):
        line = json.loads(line)
        test_ref_court_view = line['court_view'].replace(' ', '')
        
        # 特别注意：由于mengzi-t5-base, mt5base, Randeng-T5-784M会把中、英文逗号(，,)统一表示成英文逗号（,），因此通过编解码统一表示成英文逗号（,）
        rebuild_test_ref_court_view = tokenizer.decode(tokenizer.encode(test_ref_court_view, padding=True, truncation=False), skip_special_tokens=True).replace(' ', '')
    
        assert rebuild_test_ref_court_view == ref_court_view
    print(f"check reference's court view finish.")

def load_data(data_path):
    predictions_list = []
    references_list = []
    f = open(data_path,'r',encoding='utf8')
    for line in f:
        line = json.loads(line) 
        predictions_list.append(line['prediction_view'].replace(' ', ''))  
        references_list.append(line['reference_view'].replace(' ', ''))  

    return references_list, predictions_list

def calculate_rouge(
    predictions_list=None, 
    references_list=None,
    rouge=None
):
    # 计算rouge和blue时: 不同模型的token粒度要保持一致，这样才公平，因此之间在字之间加空格；
    preds_tokenizer_list = [" ".join(list(pred.replace(' ', ''))) for pred in predictions_list]
    reference_tokenizer_list = [" ".join(list(reference.replace(' ', ''))) for reference in references_list]
    logging.info("calculate rouge metrics.")
    logging.info(f"predictions example: {predictions_list[0]}")
    logging.info(f"references example: {references_list[0]}")
    logging.info(f"predictions tokenizer example: {preds_tokenizer_list[0]}")
    logging.info(f"reference tokenizer example: {reference_tokenizer_list[0]}")
    
    rouge_1_score = []
    rouge_2_score = []
    rouge_L_score = []

    for pred, reference in zip(preds_tokenizer_list, reference_tokenizer_list):
        # hyps, refs
        rouge_score = rouge.get_scores(hyps=pred, refs=reference)
        rouge_1_score.append(rouge_score[0]["rouge-1"]['f'])
        rouge_2_score.append(rouge_score[0]["rouge-2"]['f'])
        rouge_L_score.append(rouge_score[0]["rouge-l"]['f'])
        
    return rouge_1_score, rouge_2_score, rouge_L_score


def calculate_bleu(
    predictions_list=None, 
    references_list=None, 
):
    # 计算rouge和blue时: 不同模型的token粒度要保持一致，这样才公平，因此之间在字之间加空格；
    preds_tokenizer_list = [" ".join(list(pred.replace(' ', ''))) for pred in predictions_list]
    reference_tokenizer_list = [" ".join(list(reference.replace(' ', ''))) for reference in references_list]
    logging.info("calculate bleu metrics.")
    logging.info(f"predictions example: {predictions_list[0]}")
    logging.info(f"references example: {references_list[0]}")
    logging.info(f"predictions tokenizer example: {preds_tokenizer_list[0]}")
    logging.info(f"reference tokenizer example: {reference_tokenizer_list[0]}")

    bleu_1_score = []
    bleu_2_score = []
    bleu_3_score = []
    bleu_4_score = []
    for pred, reference in zip(preds_tokenizer_list, reference_tokenizer_list):
        # 计算BLEU分数时候，reference格式[['a', 'b', 'c']], pred格式['a', 'b', 'c']
        reference = [reference.split(' ')]
        pred = pred.split(' ')
        bleu_1_score.append(sentence_bleu(references=reference, hypothesis=pred, weights=(1, 0, 0, 0)))
        bleu_2_score.append(sentence_bleu(references=reference, hypothesis=pred, weights=(1/2, 1/2, 0, 0)))
        bleu_3_score.append(sentence_bleu(references=reference, hypothesis=pred, weights=(1/3, 1/3, 1/3, 0)))
        bleu_4_score.append(sentence_bleu(references=reference, hypothesis=pred, weights=(1/4, 1/4, 1/4, 1/4)))

    return bleu_1_score, bleu_2_score, bleu_3_score, bleu_4_score


def calculate_bert_score(
    predictions_list=None, 
    references_list=None, 
    bert_scorer=None,
    bert_tokenerizer=None,
    max_length=512
):
    # 计算BertScore，token之间不需要空格
    # 使用bert时，需要注意长度小于512
    # 截断操作
    predictions_truncation_lists = [truncate_and_convert_back(text, bert_tokenerizer, max_length) for text in predictions_list]
    references_truncation_lists = [truncate_and_convert_back(text, bert_tokenerizer, max_length) for text in references_list]
    
    logging.info("calculate bert score metrics.")
    logging.info(f"predictions_truncation example: {predictions_truncation_lists[0]}")
    logging.info(f"references_truncation example: {references_truncation_lists[0]}")
    
    bertscore_P, bertscore_R, bertscore_F1 = bert_scorer.score(refs=references_truncation_lists, cands=predictions_truncation_lists, batch_size=128, verbose=True)
    return bertscore_P, bertscore_R, bertscore_F1
    
    
def calculate_all_metrics(
    predictions_list=None, 
    references_list=None, 
    rouge=None, 
    bert_scorer=None,
    bert_tokenerizer=None,
    metric_save_path=None,
    each_sample_metric_save_path=None,
):
    
    print('calculate rouge metric...')
    rouge_1_score, rouge_2_score, rouge_L_score = calculate_rouge(
                                predictions_list=predictions_list, 
                                references_list=references_list,
                                rouge=rouge
    )
    print('calculate bleu metric...')
    bleu_1_score, bleu_2_score, bleu_3_score, bleu_4_score = calculate_bleu(
                                        predictions_list=predictions_list, 
                                        references_list=references_list
    )
    print('calculate bertscore metric...')
    bertscore_P, bertscore_R, bertscore_F1 = calculate_bert_score(
                                predictions_list=predictions_list, 
                                references_list=references_list, 
                                bert_scorer=bert_scorer,
                                bert_tokenerizer=bert_tokenerizer,
                                max_length=512
    )

    # average all sample 
    avg_rouge_1_score = round(np.mean(rouge_1_score) * 100, 2)
    avg_rouge_2_score = round(np.mean(rouge_2_score) * 100, 2)
    avg_rouge_L_score = round(np.mean(rouge_L_score) * 100, 2)
    
    avg_bleu_1_score = round(np.mean(bleu_1_score) * 100, 2)
    avg_bleu_2_score = round(np.mean(bleu_2_score) * 100, 2)
    avg_bleu_3_score = round(np.mean(bleu_3_score) * 100, 2)
    avg_bleu_4_score = round(np.mean(bleu_4_score) * 100, 2)
    avg_bleu_n_score = round(np.mean([avg_bleu_1_score, avg_bleu_2_score,avg_bleu_3_score, avg_bleu_4_score]), 2)
    # bert score
    avg_bertscore_P = round(bertscore_P.mean().item() * 100, 2)
    avg_bertscore_R = round(bertscore_R.mean().item() * 100, 2)
    avg_bertscore_F1 = round(bertscore_F1.mean().item() * 100, 2)
    
    metrics_dict = {
        "rouge_1_score": rouge_1_score,
        "rouge_2_score": rouge_2_score,
        "rouge_L_score": rouge_L_score,
        "bleu_1_score": bleu_1_score,
        "bleu_2_score": bleu_2_score,
        "bleu_3_score": bleu_3_score,
        "bleu_4_score": bleu_4_score,
        "bertscore_P": bertscore_P,
        "bertscore_R": bertscore_R,
        "bertscore_F1": bertscore_F1
    }
   
    avg_metrics_dict = {
        "avg_rouge_1_score": avg_rouge_1_score,
        "avg_rouge_2_score": avg_rouge_2_score,
        "avg_rouge_L_score": avg_rouge_L_score,
        "avg_bleu_1_score": avg_bleu_1_score,
        "avg_bleu_2_score": avg_bleu_2_score,
        "avg_bleu_3_score": avg_bleu_3_score,
        "avg_bleu_4_score": avg_bleu_4_score,
        "avg_bleu_n_score": avg_bleu_n_score, 
        "avg_bertscore_P": avg_bertscore_P,
        "avg_bertscore_R": avg_bertscore_R,
        "avg_bertscore_F1": avg_bertscore_F1
    }
   
    # 列名的顺序
    fieldnames = ['avg_rouge_1_score', 
                  'avg_rouge_2_score', 
                  'avg_rouge_L_score', 
                  'avg_bleu_1_score',
                  'avg_bleu_2_score',
                  'avg_bleu_3_score',
                  'avg_bleu_4_score',
                  'avg_bleu_n_score',
                  'avg_bertscore_P',
                  'avg_bertscore_R',
                  'avg_bertscore_F1'
    ]

    # 保存总体指标
    with open(metric_save_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # 写入列名
        writer.writeheader()
        # 写入数据行
        writer.writerow(avg_metrics_dict)        
                
    # 保存每一条样本的指标值
    # 保存数据到文件
    with open(each_sample_metric_save_path, 'wb') as fw:
        pickle.dump(metrics_dict, fw)


if __name__ == "__main__":        
    parser = argparse.ArgumentParser(description='evaluate_metrics for CCVG') 
    parser.add_argument('--input_pred_result_path', type=str, default='', help='input_pred_result_path')
    parser.add_argument('--test_path', type=str, default='', help='test_path')
    parser.add_argument('--backbone_model_name', type=str, default='', help='backbone_model_name')
    parser.add_argument('--metric_save_path', type=str, default='', help='metric_save_path')
    parser.add_argument('--each_sample_metric_save_path', type=str, default='', help='each_sample_metric_save_path')

    args = parser.parse_args()

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO,
                        )
    logger = logging.getLogger(__name__)

    print("PyTorch is using GPU:", torch.cuda.current_device())
    args.device = torch.device("cuda:" + str(torch.cuda.current_device()))

    print(args.device)

    for k, v in vars(args).items():
        logger.info("{:20} : {:10}".format(k, str(v)))

    # load prediction file
    input_pred_result_path = args.input_pred_result_path
    test_path = args.test_path
    # load evaluate model
    evaluate_model_base_path = "/home/leyuquan/projects/LLMs/PLM_Backbones/"
    bert_model_path = os.path.join(evaluate_model_base_path, "bert-base-chinese/snapshots/84b432f646e4047ce1b5db001d43a348cd3f6bd0/")

    # rouge metrics
    rouge = Rouge()

    # BertScorer metrics (using all 12 layer of bert-base-chinese)
    bert_scorer = BERTScorer(
        model_type=bert_model_path,
        num_layers=12,
        device=args.device
    )
    bert_tokenerizer = AutoTokenizer.from_pretrained(bert_model_path)

    references_list, predictions_list = load_data(input_pred_result_path)
    backbone_model_path = os.path.join(evaluate_model_base_path, args.backbone_model_name)
    backbone_tokenizer = AutoTokenizer.from_pretrained(backbone_model_path)
  
    logger.info(f'reference example: {references_list[0]}.')
    logger.info(f'prediction example: {predictions_list[0]}.')

    # check references whether align with test data's references
    check_gt_court_view(
        data_path=args.test_path, 
        references_list=references_list, 
        tokenizer=backbone_tokenizer
    )

    # 记录程序开始时间
    start_time = time.time()

    # calculate all metrics for test dataset.
    calculate_all_metrics(
        predictions_list=predictions_list, 
        references_list=references_list, 
        rouge=rouge, 
        bert_scorer=bert_scorer,
        bert_tokenerizer=bert_tokenerizer,
        metric_save_path=args.metric_save_path,
        each_sample_metric_save_path=args.each_sample_metric_save_path
    )

    # 记录程序结束时间
    end_time = time.time()
    # 计算程序运行时长（秒）
    duration_seconds = end_time - start_time
    # 将总秒数转换为小时、分钟、秒
    hours, rem = divmod(duration_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    # 打印程序运行时长格式化字符串
    print(f"程序运行时长为: {int(hours)} 小时 {int(minutes)} 分钟 {seconds:.2f} 秒")