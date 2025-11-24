import json
import torch
import re
import logging
import random
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import numpy as np
import bert_score
from bert_score import BERTScorer
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


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

def calculate_metrics(references_list, predictions_list, bert_scorer):
    preds_tokenizer_list = [" ".join(list(pred.replace(' ', ''))) for pred in predictions_list]
    reference_tokenizer_list = [" ".join(list(reference.replace(' ', ''))) for reference in references_list]
    logging.info("calculate rouge/bleu/bertscore metrics.")
    logging.info(f"predictions example: {predictions_list[0]}")
    logging.info(f"references example: {references_list[0]}")
    logging.info(f"predictions tokenizer example: {preds_tokenizer_list[0]}")
    logging.info(f"reference tokenizer example: {reference_tokenizer_list[0]}")
    
    rouge_score1_ = []
    rouge_score2_ = []
    rouge_scorel_ = []

    bleu_score1_ = []
    bleu_score2_ = []
    bleu_score3_ = []
    bleu_score4_ = []
    rouge = Rouge()
    for pred, ref in zip(preds_tokenizer_list, reference_tokenizer_list):
        rouge_score = rouge.get_scores(pred, ref)
        rouge_score1_.append(rouge_score[0]["rouge-1"]['f'])
        rouge_score2_.append(rouge_score[0]["rouge-2"]['f'])
        rouge_scorel_.append(rouge_score[0]["rouge-l"]['f'])

        # 计算BLEU分数时候，reference格式[['a', 'b', 'c']], pred格式['a', 'b', 'c']
        ref = [ref.split(' ')]
        pred = pred.split(' ')

        bleu_score1_.append(sentence_bleu(references=ref, hypothesis=pred, weights=(1, 0, 0, 0)))
        bleu_score2_.append(sentence_bleu(references=ref, hypothesis=pred, weights=(1/2, 1/2, 0, 0)))
        bleu_score3_.append(sentence_bleu(references=ref, hypothesis=pred, weights=(1/3, 1/3, 1/3, 0)))
        bleu_score4_.append(sentence_bleu(references=ref, hypothesis=pred, weights=(1/4, 1/4, 1/4, 1/4)))

    bleu_score1 = np.mean(bleu_score1_)
    bleu_score2 = np.mean(bleu_score2_)
    bleu_score3 = np.mean(bleu_score3_)
    bleu_score4 = np.mean(bleu_score4_)
    bleu_scoren = np.mean([bleu_score1,bleu_score2,bleu_score3,bleu_score4])
    rouge_score1 = np.mean(rouge_score1_)
    rouge_score2 = np.mean(rouge_score2_)
    rouge_scorel = np.mean(rouge_scorel_)
  
    # 计算BertScore，token之间不需要空格
    bertscore_P, bertscore_R, bertscore_F1 = bert_scorer.score(cands=predictions_list, refs=references_list, batch_size=16, verbose=True)
    
    metrics_dict = {
        "bleu_score1": round(bleu_score1 * 100, 2),
        "bleu_score2": round(bleu_score2 * 100, 2),
        "bleu_score3": round(bleu_score3 * 100, 2),
        "bleu_score4": round(bleu_score4 * 100, 2),
        "bleu_scoren": round(bleu_scoren * 100, 2),
        "rouge_score1": round(rouge_score1 * 100, 2),
        "rouge_score2": round(rouge_score2 * 100, 2),
        "rouge_scoreL": round(rouge_scorel * 100, 2),
        "bertscore_P": round(bertscore_P.mean().item() * 100, 2),
        "bertscore_R": round(bertscore_R.mean().item() * 100, 2),
        "bertscore_F1": round(bertscore_F1.mean().item() * 100, 2)
    }
    return metrics_dict
def eval_test_prediction(model=None, dataloader=None, file_path =None, params = None, args=None, tokenizer=None):
    predictions_list, references_list = [], []
    f_result = open(file_path,'w')
    for step,batch in enumerate(tqdm(dataloader)):
        batch = tuple(v.to(args.device) for k,v in batch.items())
        
        # if args.model_name == 'vanilla':
        #     source_input_ids, source_attention_mask, target_input_ids = batch
        # elif args.model_name == 'Intent':
        #     source_input_ids, source_attention_mask, target_input_ids, claim_intent_id, claim_intent_attention_mask = batch
        # else:
        #     raise NotImplementedError  
        source_input_ids, source_attention_mask, target_input_ids = batch
        
        #source_input_ids, source_attention_mask, target_input_ids, claim_intent_id, claim_intent_attention_mask, allIntent_fact_claim_id, allIntent_fact_claim_attention_mask, claim_intent_label = batch
      
        with torch.no_grad():
            generated_tokens = model.model.generate(source_input_ids, **params)

        decoded_preds = tokenizer.batch_decode(generated_tokens.cpu().numpy(), skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(target_input_ids.cpu().numpy(), skip_special_tokens=True)
        predictions_list += [pred.strip() for pred in decoded_preds]
        references_list += [label.strip() for label in decoded_labels]

        if step == 0:
            print(decoded_preds[0])
            print(references_list[0])

        if args.debug:
            if step > 5:
                break

    for p_, r_ in zip(predictions_list, references_list):
        result = {}
        result['prediction_view'] = p_
        result['reference_view'] = r_
        json_str = json.dumps(result, ensure_ascii=False)
        f_result.write(json_str + "\n")
    f_result.close()
    return predictions_list, references_list