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

def load_test_label(data_path, references_list, tokenizer):
    references_labels_list = []
    f = open(data_path,'r',encoding='utf8')
    for ref_view, line in zip(references_list, f):
        line = json.loads(line) 
        # ref和charge位置对应检测，有一些模型会把符号统一成英文，因此在check内容是否一致时要统一替换下
        court_view = tokenizer.decode(tokenizer.encode(line['court_view']), skip_special_tokens=True)
        # BART分字会加空格
        # check是否是同一条数据，对齐charge和court view
        assert ref_view == court_view.replace(' ', '') 
        references_labels_list.append(line['charge'])  

    return references_labels_list

def calculate_macro_rouge(
    references_list=None,
    predictions_list=None, 
    references_labels_list=None,
    rouge=None
):
    """
    Calculate macro rouge-1/2/L score between references and predictions, which consider each type (e.g. charge) of text with equal importance. 

    Parameters
    ----------
    references_list : 1d array-like. Ground truth (correct) target values.
    predictions_list : 1d array-like. Estimated targets as returned by a generater model.
    references_labels_list : 1d array-like. List of labels name (e.g. charge) for references text.
    rouge : rouge object for calculate rouge score.

    Returns
    -------
    report : dict
        Text summary of the rouge-1/2/L score for each class.
        Output_dict has the following structure:

            {'label 1': {'rouge-1': 0.5,
                         'rouge-2': 0.4,
                         'rouge-L': 0.3,
                         'support': 11},
             'label 2': { ... },
              ...
            }

        The reported averages include macro average (averaging the unweighted mean per label).
    """
    # 计算rouge和blue时: 不同模型的token粒度要保持一致，这样才公平，因此之间在字之间加空格；
    predictions_tokenizer_list = [" ".join(list(pred.replace(' ', ''))) for pred in predictions_list]
    references_tokenizer_list = [" ".join(list(reference.replace(' ', ''))) for reference in references_list]
    logging.info("calculate macro-rouge metrics.")
    logging.info(f"predictions example: {predictions_list[0]}")
    logging.info(f"references example: {references_list[0]}")
    logging.info(f"predictions tokenizer example: {predictions_tokenizer_list[0]}")
    logging.info(f"reference tokenizer example: {references_tokenizer_list[0]}")
    
    def calculate_rouge(
        current_label_refs_list=None, 
        current_label_preds_list=None, 
        rouge=None
    ):
        rouge_1_score = []
        rouge_2_score = []
        rouge_L_score = []

        for pred, ref in zip(current_label_preds_list, current_label_refs_list):
            # hyps, refs
            rouge_score = rouge.get_scores(hyps=pred, refs=ref)
            rouge_1_score.append(rouge_score[0]["rouge-1"]['f'])
            rouge_2_score.append(rouge_score[0]["rouge-2"]['f'])
            rouge_L_score.append(rouge_score[0]["rouge-l"]['f'])
        avg_rouge_1_score = np.mean(rouge_1_score)  
        avg_rouge_2_score = np.mean(rouge_2_score)
        avg_rouge_L_score = np.mean(rouge_L_score)  
        return avg_rouge_1_score, avg_rouge_2_score, avg_rouge_L_score
  
    unique_label_list = list(set(references_labels_list))
    logging.info(f'unique_label_num: {len(unique_label_list)}')
    
    output_dict = {}
    for current_label in unique_label_list:
        current_label_refs_list = []
        current_label_preds_list = []
        # 从ref和pred中获取当前类的所有样本
        for label, ref, pred in zip(references_labels_list, references_tokenizer_list, predictions_tokenizer_list):
            if current_label == label:
                current_label_refs_list.append(ref)
                current_label_preds_list.append(pred)
    
        # refs包括当前标签的样本数量
        current_label_support_num = len(current_label_refs_list)
        # 计算当前标签下所有样本的平均rouge分数
        rouge_1, rouge_2, rouge_L = calculate_rouge(
            current_label_refs_list=current_label_refs_list, 
            current_label_preds_list=current_label_preds_list,
            rouge=rouge
        )
        output_dict[current_label] = {
            "rouge_1": rouge_1, 
            "rouge_2": rouge_2, 
            "rouge_L": rouge_L, 
            "support": current_label_support_num
        }
    
    # 计算macro-level rouge：类平均    
    # macro_rouge_1 = sum_rouge_1 / len(unique_label_list)
    sum_rouge_1 = sum([value["rouge_1"] for value in output_dict.values()])
    sum_rouge_2 = sum([value["rouge_2"] for value in output_dict.values()])
    sum_rouge_L = sum([value["rouge_L"] for value in output_dict.values()])
    macro_rouge_1 = sum_rouge_1 / len(output_dict)
    macro_rouge_2 = sum_rouge_2 / len(output_dict)
    macro_rouge_L = sum_rouge_L / len(output_dict)
    
    return output_dict, macro_rouge_1, macro_rouge_2, macro_rouge_L

def calculate_macro_bleu(
    references_list=None,
    predictions_list=None, 
    references_labels_list=None
):
    """
    Calculate macro bleu-1/2/3/4/n score between references and predictions, which consider each type (e.g. charge) of text with equal importance. 

    Parameters
    ----------
    references_list : 1d array-like. Ground truth (correct) target values.
    predictions_list : 1d array-like. Estimated targets as returned by a generater model.
    references_labels_list : 1d array-like. List of labels name (e.g. charge) for references text.
    Returns
    -------
    report : dict
        Text summary of the bleu-1/2/3/4/n score for each class.
        Output_dict has the
        following structure::

            {'label 1': {'bleu-1': 0.4,
                         'bleu-2': 0.4,
                         'bleu-3': 0.3,
                         'bleu-4': 0.3,
                         'bleu-n': 0.35,
                         'support': 11},
             'label 2': { ... },
              ...
            }

    The reported averages include macro average (averaging the unweighted mean per label).
    """
    # 计算rouge和blue时: 不同模型的token粒度要保持一致，这样才公平，因此之间在字之间加空格；
    predictions_tokenizer_list = [" ".join(list(pred.replace(' ', ''))) for pred in predictions_list]
    references_tokenizer_list = [" ".join(list(reference.replace(' ', ''))) for reference in references_list]
    logging.info("calculate macro-rouge metrics.")
    logging.info(f"predictions example: {predictions_list[0]}")
    logging.info(f"references example: {references_list[0]}")
    logging.info(f"predictions tokenizer example: {predictions_tokenizer_list[0]}")
    logging.info(f"reference tokenizer example: {references_tokenizer_list[0]}")
        
    def calculate_bleu(
        current_label_refs_list=None, 
        current_label_preds_list=None, 
    ):
        bleu_1_score = []
        bleu_2_score = []
        bleu_3_score = []
        bleu_4_score = []
        for pred, ref in zip(current_label_preds_list, current_label_refs_list):
            # 计算BLEU分数时候，reference格式[['a', 'b', 'c']], pred格式['a', 'b', 'c']
            ref = [ref.split(' ')]
            pred = pred.split(' ')
            bleu_1_score.append(sentence_bleu(references=ref, hypothesis=pred, weights=(1, 0, 0, 0)))
            bleu_2_score.append(sentence_bleu(references=ref, hypothesis=pred, weights=(1/2, 1/2, 0, 0)))
            bleu_3_score.append(sentence_bleu(references=ref, hypothesis=pred, weights=(1/3, 1/3, 1/3, 0)))
            bleu_4_score.append(sentence_bleu(references=ref, hypothesis=pred, weights=(1/4, 1/4, 1/4, 1/4)))

        avg_bleu_1_score = np.mean(bleu_1_score)  
        avg_bleu_2_score= np.mean(bleu_2_score)
        avg_bleu_3_score = np.mean(bleu_3_score)  
        avg_bleu_4_score = np.mean(bleu_4_score)  
        avg_bleu_n_score = np.mean([avg_bleu_1_score, avg_bleu_2_score,avg_bleu_3_score, avg_bleu_4_score])

        return avg_bleu_1_score, avg_bleu_2_score, avg_bleu_3_score, avg_bleu_4_score, avg_bleu_n_score

  
    unique_label_list = list(set(references_labels_list))
    logging.info(f'unique_label_num: {len(unique_label_list)}')
    
    output_dict = {}
    for current_label in unique_label_list:
        current_label_refs_list = []
        current_label_preds_list = []
        # 从ref和pred中获取当前类的所有样本
        for label, ref, pred in zip(references_labels_list, references_tokenizer_list, predictions_tokenizer_list):
            if current_label == label:
                current_label_refs_list.append(ref)
                current_label_preds_list.append(pred)
    
        # refs包括当前标签的样本数量
        current_label_support_num = len(current_label_refs_list)
        # 计算当前标签下所有样本的平均rouge分数
        bleu_1, bleu_2, bleu_3, bleu_4, bleu_n = calculate_bleu(
            current_label_refs_list=current_label_refs_list, 
            current_label_preds_list=current_label_preds_list,
        )
        output_dict[current_label] = {
            "bleu_1": bleu_1, 
            "bleu_2": bleu_2, 
            "bleu_3": bleu_3, 
            "bleu_4": bleu_4, 
            "bleu_n": bleu_n, 
            "support": current_label_support_num
        }
    
    # 计算macro-level bleu：类平均    
    # macro_bleu_1 = sum_bleu_1 / len(unique_label_list)
    sum_bleu_1 = sum([value["bleu_1"] for value in output_dict.values()])
    sum_bleu_2 = sum([value["bleu_2"] for value in output_dict.values()])
    sum_bleu_3 = sum([value["bleu_3"] for value in output_dict.values()])
    sum_bleu_4 = sum([value["bleu_4"] for value in output_dict.values()])
    sum_bleu_n = sum([value["bleu_n"] for value in output_dict.values()])

    macro_bleu_1 = sum_bleu_1 / len(output_dict)
    macro_bleu_2 = sum_bleu_2 / len(output_dict)
    macro_bleu_3 = sum_bleu_3 / len(output_dict)
    macro_bleu_4 = sum_bleu_4 / len(output_dict)
    macro_bleu_n = sum_bleu_n / len(output_dict)
    
    return output_dict, macro_bleu_1, macro_bleu_2, macro_bleu_3, macro_bleu_4, macro_bleu_n

def calculate_macro_bertscore(
    references_list=None,
    predictions_list=None, 
    references_labels_list=None,
    bert_scorer=None,
    bert_tokenerizer=None,
    max_length=None
):
    """
    Calculate macro bertscore-P/R/F1 score between references and predictions, which consider each type (e.g. charge) of text with equal importance. 

    Parameters
    ----------
    references_list : 1d array-like. Ground truth (correct) target values.
    predictions_list : 1d array-like. Estimated targets as returned by a generater model.
    references_labels_list : 1d array-like. List of labels name (e.g. charge) for references text.
    Returns
    -------
    report : dict
        Text summary of the bertscore-P/R/F1 score for each class.
        Output_dict has the
        following structure::

            {'label 1': {'bertscore-P': 0.1,
                         'bertscore-R': 0.1,
                         'bertscore-F1': 0.1,
                         'support': 11},
             'label 2': { ... },
              ...
            }

    The reported averages include macro average (averaging the unweighted mean per label).
    """
    logging.info("calculate bertscore metrics.")
    
    # 计算BertScore，token之间不需要空格
    # 使用bert时，需要注意长度小于512
    # 截断操作
    predictions_truncation_lists = [truncate_and_convert_back(text, bert_tokenerizer, max_length) for text in predictions_list]
    references_truncation_lists = [truncate_and_convert_back(text, bert_tokenerizer, max_length) for text in references_list]
    
    logging.info("calculate bert score metrics.")
    logging.info(f"predictions_truncation example: {predictions_truncation_lists[0]}")
    logging.info(f"references_truncation example: {references_truncation_lists[0]}")
    
    def calculate_bert_score(
        current_label_refs_list=None, 
        current_label_preds_list=None, 
        bert_scorer=None
    ):
        bertscore_P, bertscore_R, bertscore_F1 = bert_scorer.score(
            refs=current_label_refs_list, 
            cands=current_label_preds_list, 
            batch_size=12, 
            verbose=True
        )
        
        avg_bertscore_P = bertscore_P.mean().item()
        avg_bertscore_R = bertscore_R.mean().item()
        avg_bertscore_F1 = bertscore_F1.mean().item() 
        return avg_bertscore_P, avg_bertscore_R, avg_bertscore_F1
        
         
    unique_label_list = list(set(references_labels_list))
    logging.info(f'unique_label_num: {len(unique_label_list)}')
    
    output_dict = {}
    for current_label in unique_label_list:
        current_label_refs_list = []
        current_label_preds_list = []
        # 从ref和pred中获取当前类的所有样本
        for label, ref, pred in zip(references_labels_list, references_truncation_lists, predictions_truncation_lists):
            if current_label == label:
                current_label_refs_list.append(ref)
                current_label_preds_list.append(pred)
    
        # refs包括当前标签的样本数量
        current_label_support_num = len(current_label_refs_list)
        # 计算当前标签下所有样本的平均rouge分数
        bertscore_P, bertscore_R, bertscore_F1 = calculate_bert_score(
            current_label_refs_list=current_label_refs_list, 
            current_label_preds_list=current_label_preds_list,
            bert_scorer=bert_scorer
        )
        output_dict[current_label] = {
            "bertscore_P": bertscore_P, 
            "bertscore_R": bertscore_R, 
            "bertscore_F1": bertscore_F1, 
            "support": current_label_support_num
        }
    
    # 计算macro-level bertscore：类平均    
    # macro_bertscore_F1 = sum_bertscore_F1 / len(unique_label_list)
    sum_bertscore_P = sum([value["bertscore_P"] for value in output_dict.values()])
    sum_bertscore_R = sum([value["bertscore_R"] for value in output_dict.values()])
    sum_bertscore_F1 = sum([value["bertscore_F1"] for value in output_dict.values()])
    
    assert len(output_dict) == len(unique_label_list)
    macro_bertscore_P = sum_bertscore_P / len(output_dict)
    macro_bertscore_R = sum_bertscore_R / len(output_dict)
    macro_bertscore_F1 = sum_bertscore_F1 / len(output_dict)
    
    return output_dict, macro_bertscore_P, macro_bertscore_R, macro_bertscore_F1

def calculate_all_macro_metrics(
    predictions_list=None, 
    references_list=None, 
    references_labels_list=None,
    rouge=None, 
    bert_scorer=None,
    bert_tokenerizer=None,
    metric_save_path=None,
    each_label_metric_save_path=None,
):
    
    print('calculate rouge metric...')
    each_class_macro_rouge_dict, macro_rouge_1, macro_rouge_2, macro_rouge_L = calculate_macro_rouge(
                                                predictions_list=predictions_list, 
                                                references_list=references_list,
                                                references_labels_list=references_labels_list,
                                                rouge=rouge
    )
    print('calculate bleu metric...')
    each_class_macro_bleu_dict, macro_bleu_1, macro_bleu_2, macro_bleu_3, macro_bleu_4, macro_bleu_n = calculate_macro_bleu(
                                                            predictions_list=predictions_list, 
                                                            references_list=references_list,
                                                            references_labels_list=references_labels_list,
    )
    print('calculate bertscore metric...')
    each_class_macro_bertscore_dict, macro_bertscore_P, macro_bertscore_R, macro_bertscore_F1 = calculate_macro_bertscore(
                                                predictions_list=predictions_list, 
                                                references_list=references_list, 
                                                references_labels_list=references_labels_list,
                                                bert_scorer=bert_scorer,
                                                bert_tokenerizer=bert_tokenerizer,
                                                max_length=512
    )

    macro_metrics_dict = {
        "macro_rouge_1": round(macro_rouge_1 * 100, 2),
        "macro_rouge_2": round(macro_rouge_2 * 100, 2),
        "macro_rouge_L": round(macro_rouge_L * 100, 2),
        "macro_bleu_1": round(macro_bleu_1 * 100, 2),
        "macro_bleu_2": round(macro_bleu_2 * 100, 2),
        "macro_bleu_3": round(macro_bleu_3 * 100, 2),
        "macro_bleu_4": round(macro_bleu_4 * 100, 2),
        "macro_bleu_n": round(macro_bleu_n * 100, 2),
        "macro_bertscore_P": round(macro_bertscore_P * 100, 2),
        "macro_bertscore_R": round(macro_bertscore_R * 100, 2),
        "macro_bertscore_F1": round(macro_bertscore_F1 * 100, 2),
    }
       
    # 保存macro总体指标
    with open(metric_save_path, 'w', newline='') as csvfile:
        fieldnames = list(macro_metrics_dict.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # 写入列名
        writer.writeheader()
        # 写入数据行
        writer.writerow(macro_metrics_dict)  
                
    # 保存每一个类的指标值
    # 保存数据到文件
    with open(each_label_metric_save_path, 'w', newline='') as each_label_csvfile:
        fieldnames = ["label", "rouge_1", "rouge_2", "rouge_L", "bleu_1", "bleu_2", "bleu_3", "bleu_4", "bleu_n", "bertscore_P", "bertscore_R", "bertscore_F1", 'support']
        each_label_writer = csv.DictWriter(each_label_csvfile, fieldnames=fieldnames)
        # 写入列名
        each_label_writer.writeheader()
        
        all_label_lists = list(each_class_macro_rouge_dict.keys())
        for label in all_label_lists:
            label_rouge_1 = each_class_macro_rouge_dict[label]['rouge_1']
            label_rouge_2 = each_class_macro_rouge_dict[label]['rouge_2']
            label_rouge_L = each_class_macro_rouge_dict[label]['rouge_L']
            label_bleu_1 = each_class_macro_bleu_dict[label]['bleu_1']
            label_bleu_2 = each_class_macro_bleu_dict[label]['bleu_2']
            label_bleu_3 = each_class_macro_bleu_dict[label]['bleu_3']
            label_bleu_4 = each_class_macro_bleu_dict[label]['bleu_4']
            label_bleu_n = each_class_macro_bleu_dict[label]['bleu_n']
            label_bertscore_P = each_class_macro_bertscore_dict[label]['bertscore_P']
            label_bertscore_R = each_class_macro_bertscore_dict[label]['bertscore_R']
            label_bertscore_F1 = each_class_macro_bertscore_dict[label]['bertscore_F1']
            label_support_num = each_class_macro_bertscore_dict[label]['support']
            
            assert each_class_macro_bertscore_dict[label]['support'] == each_class_macro_bleu_dict[label]['support'] == each_class_macro_rouge_dict[label]['support']
            
            each_label_metric = [
                label,
                round(label_rouge_1 * 100, 2),
                round(label_rouge_2 * 100, 2),
                round(label_rouge_L * 100, 2),
                round(label_bleu_1 * 100, 2),
                round(label_bleu_2 * 100, 2),
                round(label_bleu_3 * 100, 2),
                round(label_bleu_4 * 100, 2),
                round(label_bleu_n * 100, 2),
                round(label_bertscore_P * 100, 2),
                round(label_bertscore_R * 100, 2),
                round(label_bertscore_F1 * 100, 2),
                label_support_num
            ]
            
            each_label_metric_dict = {k:v for k,v in zip(fieldnames, each_label_metric)}
            
            # 写入数据行
            each_label_writer.writerow(each_label_metric_dict)
    

if __name__ == "__main__":     
    parser = argparse.ArgumentParser(description='Macro-level evaluate metric for CCVG')
    parser.add_argument('--test_path', type=str, default='', help='')
    parser.add_argument('--test_pred_result_path', type=str, default='', help='')
    parser.add_argument('--backbone_model_name', type=str, default='', help='backbone_model_name')
    parser.add_argument('--metric_save_path', type=str, default='', help='metric_save_path')
    parser.add_argument('--each_label_metric_save_path', type=str, default='', help='each_label_metric_save_path')

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

    # load evaluate model
    evaluate_model_base_path = "/home/leyuquan/projects/LLMs/PLM_Backbones/"
    bert_model_path = os.path.join(evaluate_model_base_path, "bert-base-chinese/snapshots/84b432f646e4047ce1b5db001d43a348cd3f6bd0/")

    # rouge metrics
    rouge = Rouge()

    # BertScorer metrics (using all 12 layer of bert-base-chinese)
    bert_scorer = BERTScorer(model_type=bert_model_path,
                            num_layers=12,
                            device=args.device
    )
    bert_tokenerizer = AutoTokenizer.from_pretrained(bert_model_path)


    references_list, predictions_list = load_data(args.test_pred_result_path)        
    
    backbone_model_path = os.path.join("/home/leyuquan/projects/LLMs/PLM_Backbones", args.backbone_model_name)
    backbone_tokenizer = AutoTokenizer.from_pretrained(backbone_model_path)
    references_labels_list = load_test_label(args.test_path, references_list, backbone_tokenizer)
    logger.info(f'reference example: {references_list[0]}')
    logger.info(f'prediction example: {predictions_list[0]}')
    logger.info(f'references labels example: {references_labels_list[0]}')

    # check references whether align with test data's references
    check_gt_court_view(
        data_path=args.test_path, 
        references_list=references_list, 
        tokenizer=backbone_tokenizer
    )

    # 记录程序开始时间
    start_time = time.time()

    # calculate all macro matrix for test dataset.
    calculate_all_macro_metrics(
        predictions_list=predictions_list, 
        references_list=references_list, 
        references_labels_list=references_labels_list,
        rouge=rouge, 
        bert_scorer=bert_scorer,
        bert_tokenerizer=bert_tokenerizer,
        metric_save_path=args.metric_save_path,
        each_label_metric_save_path=args.each_label_metric_save_path
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