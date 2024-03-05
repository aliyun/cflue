from evaluator.evaluator import load_models_tokenizer, load_llama_models_tokenizer
from utils.dataset import load_dataset
from utils.format_example import format_one_example, format_multi_example
from utils.extract_choice import extract_one_choice, extract_multi_choice
from utils.compute_score import *
from tqdm import tqdm
import pandas as pd
import jieba
import argparse


def eval_ability(args):
    # load model & tokenizer
    if 'llama' in args.model_name:
        model, tokenizer = load_llama_models_tokenizer(args)
    else:
        model, tokenizer = load_models_tokenizer(args)

    # 载入评测集
    dataset = load_dataset(args.eval_type)

    # 大模型推理回答&记录答案
    responses = []
    y_true, y_pred = [], []
    references, candidates = [], []
    for _, record in tqdm(dataset.iterrows()):
        if record['task'] == '多项选择题':
            prompt = format_multi_example(record)
            model_response, _ = model.chat(
                tokenizer,
                prompt,
                history=None,
            )
            if len(model_response.split('\n')) == 2:
                pred = extract_multi_choice(model_response.split('\n')[0], [choice for choice in record["choices"]])
            else:
                pred = extract_multi_choice(model_response, [choice for choice in record["choices"]])
        else:
            prompt = format_one_example(record)
            model_response, _ = model.chat(
                tokenizer,
                prompt,
                history=None,
            )
            if len(model_response.split('\n')) == 2:
                pred = extract_one_choice(model_response.split('\n')[0], [choice for choice in record["choices"]])
            else:
                pred = extract_one_choice(model_response, [choice for choice in record["choices"]])

        responses.append(model_response)
        y_pred.append(pred)
        y_true.append(record['answer'])
        if 'analysis' in record and not pd.isna(record['analysis']):
            references.append(" ".join(jieba.lcut(record['analysis'])))
            if len(model_response.strip().split('\n')) == 2:
                candidates.append(" ".join(jieba.lcut(model_response.strip().split('\n')[1].replace("解析：", ""))))
            else:
                candidates.append(" ".join(jieba.lcut(model_response)))

    # 计算分数
    score_acc = acc_score(y_true, y_pred)
    score_weighted_f1 = f1_score(y_true, y_pred)
    bleu_1, bleu_4 = bleu_score(references, candidates)
    rouge_1, rouge_2, rouge_l = rouge_score(references, candidates)

    result_path = os.path.join(args.save_result_dir, f"{args.model_name}_ability_result.json")
    if args.save_result_dir:
        dataset["model_response"] = responses
        dataset["model_pred"] = y_pred
        dataset["acc"] = score_acc
        dataset["weighted_f1"] = score_weighted_f1
        dataset["bleu_1"] = bleu_1
        dataset["bleu_4"] = bleu_4
        dataset["rouge_1"] = rouge_1
        dataset["rouge_2"] = rouge_2
        dataset["rouge_L"] = rouge_l
        os.makedirs(args.save_result_dir, exist_ok=True)
        dataset.to_json(result_path, orient='records', force_ascii=False)
