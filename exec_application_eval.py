from evaluator.evaluator import load_models_tokenizer, load_llama_models_tokenizer
from utils.dataset import load_dataset
from utils.compute_score import *
from tqdm import tqdm
import argparse


def eval_application(args):
    # load model & tokenizer
    if 'llama' in args.model_name:
        model, tokenizer = load_llama_models_tokenizer(args)
    else:
        model, tokenizer = load_models_tokenizer(args)

    # 载入评测集
    dataset = load_dataset(args.eval_type)

    # 大模型推理回答&记录答案
    responses = []
    for _, record in tqdm(dataset.iterrows()):
        prompt = record['instruction']
        model_response, _ = model.chat(
            tokenizer,
            prompt,
            history=None,
        )
        responses.append(model_response)
    result_path = os.path.join(args.save_result_dir, f"{args.model_name}_application_result.json")
    if args.save_result_dir:
        dataset["model_response"] = responses
        os.makedirs(args.save_result_dir, exist_ok=True)
        dataset.to_json(result_path, orient='records', force_ascii=False)

    # 计算应用评分
    get_application_score(args)


def get_application_score(args):
    _path = args.save_result_dir
    file_path = f'{_path}/{args.model_name}_application_result.json'

    result = {}
    print('Model: %s' % args.model_name)
    # QA
    rouge_l, qa_bert = compute_finqa(file_path)
    result['QA'] = {'rouge-L': rouge_l,
                    'Bert': qa_bert}
    # TG
    rouge_l_tg, _, tg_bert, _ = compute_text_generation(file_path)
    result['TG'] = {'rouge-L': rouge_l_tg,
                    'Bert': tg_bert}
    # MT-e2zh
    bleu, comet = compute_nmt_en2zh(file_path)
    result['MT-e2zh'] = {'BLEU': bleu,
                         'COMET': comet}
    # MT-zh2e
    bleu, comet = compute_nmt_zh2en(file_path)
    result['MT-zh2e'] = {'BLEU': bleu,
                         'COMET': comet}
    # TC
    acc, _ = compute_text_classification(file_path)
    result['TC'] = {'ACC': acc}
    # RE
    f1, _ = compute_extraction(file_path)
    result['RE'] = {'F1-score': f1}
