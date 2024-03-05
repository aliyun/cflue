import json
import os
import numpy as np
import re
import sys
import pandas as pd
import jieba
import nltk
from sklearn.metrics import f1_score as f1
from sklearn.metrics import accuracy_score
from rouge import Rouge
from sacrebleu.metrics import BLEU
from comet import load_from_checkpoint
from bert_score import score


sys.setrecursionlimit(100000)  # 例如这里设置为十万


def tokenizer_en(sent):
    return nltk.word_tokenize(sent)


def acc_score(y_true, y_pred):
    res = accuracy_score(y_true, y_pred)
    print("ACC:", res)
    return res


def f1_score(y_true, y_pred):
    res = f1(y_true, y_pred, average='weighted')
    print("Weighted F1:", res)
    return res


def bleu_score(reference, candidate):
    bleu = BLEU()
    bleu_score = bleu.corpus_score(candidate, [reference])
    bleu_1 = bleu_score.precisions[0]
    bleu_4 = bleu_score.score
    print("BLEU-1:", bleu_1)
    print("BLEU-4:", bleu_4)
    return bleu_1, bleu_4


def bert_score(reference, candidate):
    P, R, F1 = score(candidate, reference, model_type="bert-base-chinese", lang="zh", verbose=True)
    print('Bert Score: %s' % F1.mean())
    return F1.mean()


def rouge_score(reference, candidate):
    rouge_calculator = Rouge()
    # Rouge-1 [r, p, f]
    rouge_1 = rouge_calculator.get_scores(candidate, reference, avg=True)['rouge-1']['f']

    # Rouge-2
    rouge_2 = rouge_calculator.get_scores(candidate, reference, avg=True)['rouge-2']['f']

    # Rouge-L
    rouge_l = rouge_calculator.get_scores(candidate, reference, avg=True)['rouge-l']['f']

    print("Rouge-1:", rouge_1 * 100)
    print("Rouge-2:", rouge_2 * 100)
    print("Rouge-L:", rouge_l * 100)
    return rouge_1, rouge_2, rouge_l


def compute_nmt_zh2en(file_path, model_path):
    references = []
    candidates = []
    comet_data = []
    print("金融中英翻译")
    # model_path = download_model("Unbabel/XCOMET-XL")
    model = load_from_checkpoint(model_path)  # XCOMET-XL/checkpoints/model.ckpt
    """
    data = [
        {
            "src": "10 到 15 分钟可以送到吗",
            "mt": "Can I receive my food in 10 to 15 minutes?",
            "ref": "Can it be delivered between 10 to 15 minutes?"
        },...
    ]
    """
    with open(file_path, 'r') as input_file:
        print("Reading\t" + file_path)
        sample_list = json.load(input_file)

        for line in sample_list:
            if line['sub_task'] == '金融中英翻译':
                references.append(" ".join(tokenizer_en(line['output'])))
                candidates.append(" ".join(tokenizer_en(line['predict'])))
                src = line['instruction'].replace("你是一个金融行业专家，请将下面金融领域的中文内容翻译成准确、专业的英文。\n中文：", "").replace("英文：",
                                                                                                           "").strip()
                comet_data.append({
                    "src": src,
                    "mt": line['predict'],
                    "ref": line['output']
                })
        _, bleu4 = bleu_score(references, candidates)
        model_output = model.predict(comet_data, batch_size=32, gpus=1)
        print(model_output.system_score)  # system-level score
        print('\n')
    return bleu4, model_output.system_score


def compute_nmt_en2zh(file_path, model_path):
    references = []
    candidates = []
    comet_data = []

    # model_path = download_model("Unbabel/XCOMET-XL")
    model = load_from_checkpoint(model_path)  # XCOMET-XL/checkpoints/model.ckpt
    print("金融英中翻译")
    with open(file_path, 'r') as input_file:
        print("Reading\t" + file_path)
        sample_list = json.load(input_file)

        for line in sample_list:
            if line['sub_task'] == '金融英中翻译':
                references.append(" ".join(jieba.lcut(line['output'])))
                candidates.append(" ".join(jieba.lcut(line['predict'])))
                src = line['instruction'].replace(
                    "你是一个金融行业专家，请将下面金融领域的英文内容翻译成准确、专业的中文。\n英文：", "").replace("中文：", "").strip()
                comet_data.append({
                    "src": src,
                    "mt": line['predict'],
                    "ref": line['output']
                })
        _, bleu4 = bleu_score(references, candidates)
        model_output = model.predict(comet_data, batch_size=32, gpus=1)
        print(model_output.system_score)  # system-level score
        print('\n')
    return bleu4, model_output.system_score


def compute_text_generation(file_path):
    references = []
    candidates = []

    sub_task_references = {}
    sub_task_candidates = {}

    with open(file_path, 'r', encoding='utf-8') as input_file:
        print("Reading\t" + file_path)
        sample_list = json.load(input_file)
        sample_list = pd.json_normalize(sample_list, 'outputs')
        for _, line in sample_list.iterrows():
            if line['__raw.task'] == '金融文本生成':
                # compute all score
                if not line['response']:
                    line['response'] = 'None'
                references.append(" ".join(jieba.lcut(line['__raw.output'])))
                candidates.append(" ".join(jieba.lcut(line['response'])))

                # compute sub_task score
                sub_task = line['__raw.sub_task']
                if sub_task not in sub_task_references:
                    sub_task_references[sub_task] = []
                    sub_task_candidates[sub_task] = []
                sub_task_references[sub_task].append(" ".join(jieba.lcut(line['__raw.output'])))
                sub_task_candidates[sub_task].append(" ".join(jieba.lcut(line['response'])))

        # result
        print(f"task: 金融文本生成")
        _, _, rougel = rouge_score(references, candidates)
        bert = bert_score(references, candidates)
        print('\n')

        sub_task_tg, sub_task_tg_bert = {}, {}
        for sub_task, refs in sub_task_references.items():
            print(f"Sub-task: {sub_task}")
            candidates = sub_task_candidates[sub_task]
            _, _, rouge_l = rouge_score(refs, candidates)
            bert_sub_task = bert_score(refs, candidates)
            sub_task_tg[sub_task] = rouge_l
            sub_task_tg_bert[sub_task] = bert_sub_task
            print('\n')
    return rougel, sub_task_tg, bert, sub_task_tg_bert


def compute_finqa(file_path):
    references = []
    candidates = []

    sub_task_references = {}
    sub_task_candidates = {}

    with open(file_path, 'r', encoding='utf-8') as input_file:
        print("Reading\t" + file_path)
        sample_list = json.load(input_file)
        samples = pd.json_normalize(sample_list, 'outputs')
        for _, line in samples.iterrows():
            if line['__raw.task'] == '金融咨询':
                # compute all score
                if not line['response']:
                    line['response'] = 'None'
                references.append(" ".join(jieba.lcut(line['__raw.output'])))
                candidates.append(" ".join(jieba.lcut(line['response'])))

                # compute sub_task score
                sub_task = line['__raw.sub_task']
                if sub_task not in sub_task_references:
                    sub_task_references[sub_task] = []
                    sub_task_candidates[sub_task] = []
                sub_task_references[sub_task].append(" ".join(jieba.lcut(line['__raw.output'])))
                sub_task_candidates[sub_task].append(" ".join(jieba.lcut(line['response'])))

        # result
        print(f"task: 金融咨询")
        _, _, rougel = rouge_score(references, candidates)
        bert = bert_score(references, candidates)
        print('\n')

        sub_task_qa, sub_task_bert = {}, {}
        for sub_task, refs in sub_task_references.items():
            print(f"Sub-task: {sub_task}")
            candidates = sub_task_candidates[sub_task]
            _, _, rouge_l = rouge_score(refs, candidates)
            print('\n')
    return rougel, bert


def compute_text_classification(file_path):
    with open(file_path, 'r', encoding='utf-8') as input_file:
        print("Reading\t" + file_path)
        sample_list = json.load(input_file)
        samples = pd.json_normalize(sample_list, 'outputs')
    samples = samples[samples['__raw.task'] == '金融文本分类']
    pattern = r"'([^']*)'"  # 提取单引号之间字符串
    puncs = [',', '，', '、']
    acc_list = []
    for _, row in samples.iterrows():
        label = row['__raw.output']
        if row['__raw.sub_task'] == 'ESG情感分类':
            if label[0] in row['response']:  # '正', '中', '负'
                acc = 1
            elif label == '负向' and row['response'] == 'Negative' or label == '正向' and row['response'] == 'Positive':
                acc = 1
            else:
                acc = 0
        elif row['__raw.sub_task'] == '合规政策审核':  # for baichuan2
            if any(s in row['response'] for s in ['不合规', '不符合']):
                if row['__raw.output'] == '否':
                    acc = 1
                else:
                    acc = 0
            else:
                if row['__raw.output'] == '是':
                    acc = 1
                else:
                    acc = 0
        else:
            pred = row['response'].strip('\n')
            if re.findall(pattern, pred):  # "'物料'", " ['经济绩效', '非直接经济影响']"
                pred = re.findall(pattern, pred)
                if len(pred) == 1 and label in pred:
                    acc = 1
                else:
                    acc = 0
            elif any(p in pred for p in puncs):  # "反腐败行为, 非虚假营销, 依法合规纳税, 反不正当竞争, 安全管理实践, 能源, 市场占有率, 排放"
                acc = 0
            else:  # " 依法合规纳税"
                if label in pred:
                    acc = 1
                else:
                    acc = 0
        acc_list.append(acc)
    return np.mean(acc_list)


def get_num_infos(target_str, target_char=None):
    '''统计target_char在target_str里面出现的个数'''
    if target_char is None:
        target_char = ['、', ':', ';', ',', '\n']
    num_infos = []
    for sep in target_char:
        num = target_str.count(sep)
        if num > 0:
            num_info = {'sep': sep, 'num': num}
            num_infos.append(num_info)
    num_infos = sorted(num_infos, key=lambda x: x['num'])
    return num_infos


def pad_list(org_list, target_length, pad_val, left=False):
    '''对list进行padding操作'''
    if left:
        new_list = [pad_val] * (target_length - len(org_list)) + org_list
    else:
        new_list = org_list + [pad_val] * (target_length - len(org_list))
    return new_list


def decode_re_content(content):
    '''把映射的内容转换成map'''
    num_infos = get_num_infos(content)
    class_info_dict = None
    if len(num_infos) > 0:
        try:
            sep = num_infos[0]['sep']
            ind_info_list = [strip(x) for x in content.split(sep)]
            if len(num_infos) > 1:
                sep = num_infos[1]['sep']
                class_info_dict = dict([tuple(x.split(sep)) for x in ind_info_list])
            else:
                if len(ind_info_list) % 2 != 0:
                    ind_info_list = ind_info_list[1:]
                class_info_dict = dict([tuple(ind_info_list)])
        except:
            ## 出现异常，使用第二个分隔符作为第一分隔符
            try:
                sep = num_infos[1]['sep']
                ind_info_list = content.split(sep)
                sep = num_infos[0]['sep']
                class_info_dict = dict([tuple(x.split(sep)) for x in ind_info_list if sep in x])
            except:
                content = [strip(x) for x in content.split('\n') if len(x) > 0]
                content = '\n'.join(content)
                reg_exp = '([\u4e00-\u9fa5]+)[:\s]+([正|负|中|无])[\u4e00-\u9fa5]+'
                find_res = re.findall(reg_exp, content)
                class_info_dict = {x[0]: x[1] for x in find_res}
    return class_info_dict


def extract_re_expr(class_info_list, pattern, default_val=None):
    '''正则匹配'''
    find_results = []
    for class_info in class_info_list:
        find_result = re.findall(pattern, class_info, re.DOTALL)
        if len(find_result) > 0:
            find_results.append(find_result[0])
        else:
            find_results.append(default_val)
    return find_results


def extract_re_exprs(class_info_list, patterns, default_val=None):
    '''多个正则匹配'''
    for pattern in patterns:
        find_result = extract_re_expr(class_info_list, pattern, default_val)
        if None not in find_result:
            break
    return find_result


def strip(target_str, strip_lst='\n.;'):
    '''进行字符串的strip'''
    new_str = target_str.strip(strip_lst)
    new_str = re.sub('^\s+', '', new_str)
    new_str = re.sub('\s+$', '', new_str)
    return new_str


def financial_extract_result_process(sub_df):
    '''处理模型输出的结果'''
    contents_adj = []
    for i in range(sub_df.shape[0]):
        content = sub_df.iloc[i].predict.lstrip().rstrip()
        content = content.replace('：', ':').replace('，', ',').replace(' ', '').replace('；', ';').replace('。',
                                                                                                         '.').replace(
            ',\n', '\n')
        result = {}
        if 'value' in content:
            find_list = re.findall('(key\d):(.*)\s*(value\d?):?\s*(.*)', content)
            bad_res = False
            for find_res in find_list:
                if len(find_res) != 4:
                    bad_res = True
                    break
                result[strip(find_res[1])] = strip(find_res[3])
            if bad_res:
                result = {}
        else:
            find_list = re.findall('(.*):(.*)', content)
            for find_res in find_list:
                if len(find_res) != 2:
                    break
                if 'key' in find_res[0]:
                    break
                result[strip(find_res[0])] = strip(find_res[1])
        contents_adj.append(result)
    return contents_adj


def cal_f1(label, pred):
    label_set = set(['|'.join(x) for x in label.items()])
    pred_set = set(['|'.join(x) for x in pred.items()])
    comment_set = label_set & pred_set
    precision = len(comment_set) / (len(pred_set) + 1e-12)
    recall = len(comment_set) / (len(label_set) + 1e-12)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall) * 100
    return f1


def output_adj(x):
    '''由于样本中gpt的数据比较异常，是字符串，这里对其进行转换'''
    if isinstance(x, str):
        x = eval(x)
    return x


def industry_classification_result_process_single(content):
    content = content.replace('：', ':').replace('，', ',').replace(' ', '').replace('；', ';').replace('。', '.').replace(
        ',\n', '\n')
    match_result = re.match('行业:(.*)情感分类:(.*)', content, re.DOTALL)
    class_info_dict = {}
    if match_result is not None:
        ind_info, class_info = match_result.groups()
        ind_info = ind_info.strip('\n.;')
        class_info = class_info.strip('\n.;')
        num_infos = get_num_infos(class_info)
        ## 情感分类为多个项目
        if len(num_infos) > 0:
            sep = num_infos[0]['sep']
            class_info_list = class_info.split(sep)
            ## 情感分类为二阶
            try:
                if len(num_infos) > 1:
                    sec_sep = num_infos[1]['sep']
                    class_info_dict = dict([tuple(x.split(':')) for x in class_info_list])
                else:
                    ## 情感分类为一阶
                    num_infos = get_num_infos(ind_info)
                    if len(num_infos) > 0:
                        sep = num_infos[0]['sep']
                        ind_info_list = ind_info.split(sep)
                    class_info_list_adj = extract_re_exprs(class_info_list, ['.*[为是](.*)', '.*（(.*)）'])
                    if None not in class_info_list_adj:
                        class_info_dict = dict(zip(ind_info_list, class_info_list_adj))
                    else:
                        max_len = max(len(ind_info_list), len(class_info_list))
                        class_info_list = pad_list(class_info_list, max_len, '无')
                        class_info_dict = dict(zip(ind_info_list, class_info_list))
            except:
                find_res = re.findall('(\w+)[:\s]+([正|负|中|无])\w+', class_info)
                class_info_dict = {x[0]: x[1] for x in find_res}
        else:
            num_infos = get_num_infos(ind_info)
            if len(num_infos) > 0:
                sep = num_infos[0]['sep']
                ind_info_list = ind_info.split(sep)
                class_info_dict = dict(zip(ind_info_list, [class_info] * len(ind_info_list)))
            else:
                class_info_dict = {ind_info: class_info}
            pass
    else:
        if '抽取结果' in content:
            match_result = re.match('抽取结果.*?:(.*)', content, re.DOTALL)
            if match_result is not None:
                match_result = strip(match_result.groups()[0])
                match_result1 = re.match('.*情感分类结果:(.*)', match_result, re.DOTALL)
                if match_result1 is not None:
                    class_info = match_result1.groups()[0]
                    # num_infos = [x for x in sorted(get_num_infos(ind_info), key=lambda x: x['num']) if x['num'] > 0]
                    num_infos = get_num_infos(class_info)
                    if len(num_infos) > 0:
                        sep = num_infos[0]['sep']
                        class_info_list = class_info.split(sep)
                    match_result1 = re.match('(.*)情感分类结果', match_result, re.DOTALL)
                    if match_result1 is not None:
                        ind_info = match_result1.groups()[0].strip()
                        # num_infos = [x for x in sorted(get_num_infos(ind_info), key=lambda x: x['num']) if x['num'] > 0]
                        num_infos = get_num_infos(class_info)
                        if len(num_infos) > 0:
                            sep = num_infos[0]['sep']
                            ind_info_list = ind_info.split(sep)
                            if len(class_info_list) == len(ind_info_list):
                                class_info_dict = dict(zip(ind_info_list, class_info_list))
                            else:
                                max_len = max(len(ind_info_list), len(class_info_list))
                                ind_info_list = pad_list(ind_info_list, max_len, '无')
                                class_info_list = pad_list(class_info_list, max_len, '无')
                                class_info_dict = dict(zip(ind_info_list, class_info_list))
                else:
                    try:
                        num_infos = get_num_infos(match_result)
                        if len(num_infos) > 0:
                            sep = num_infos[0]['sep']
                            ind_info_list = match_result.split(sep)
                            if len(num_infos) > 1:
                                sep = num_infos[1]['sep']
                                class_info_dict = dict([tuple(x.split(sep)) for x in ind_info_list])
                            else:
                                class_info_dict = dict(zip(ind_info_list, ['无'] * len(ind_info_list)))
                        else:
                            class_info_dict = {match_result: '无'}
                    except:
                        find_res = re.findall('(\w+)[:\s]+([正|负|中|无])\w+', match_result)
                        class_info_dict = {x[0]: x[1] for x in find_res}
            else:
                find_res = re.findall('(\w+)[:\s]+([正|负|中|无])\w+', content)
                class_info_dict = {x[0]: x[1] for x in find_res}
        else:
            find_result = [(strip(x[0]), strip(x[1])) for x in re.findall('^行业:(.*)情感分类:(.*)', content, re.DOTALL)]
            if len(find_result) > 0:
                ind_info = find_result[0][0]
                class_info = find_result[0][1]
                num_infos = get_num_infos(ind_info)
                if len(num_infos) > 0:
                    sep = num_infos[0]['sep']
                    ind_info_list = ind_info.split(sep)
                    num_infos = get_num_infos(class_info)
                    if num_infos > 0:
                        sep = num_infos[0]['sep']
                        class_info_list = class_info.split(sep)
                    else:
                        class_info_list = [class_info] * len(ind_info_list)
            else:
                ## 预测结果如下:xxx:xxx,xxx:xxx
                find_result = [strip(x) for x in re.findall('^.*结果.*?:(.*)', content, re.DOTALL)]
                if len(find_result) > 0:
                    class_info_dict = decode_re_content(find_result[0])
                else:
                    class_info_dict = decode_re_content(content)
    if class_info_dict is None or len(class_info_dict) <= 0:
        find_res = re.findall('(\w+)[:\s]*([正|负|中|无])\w+', content)
        class_info_dict = {x[0]: x[1] for x in find_res}
    return class_info_dict


def financial_extract_result_process_single(content):
    content = content.replace('：', ':').replace('，', ',').replace(' ', '').replace('；', ';').replace('。', '.').replace(
        ',\n', '\n')
    result = {}
    if 'value' in content:
        find_list = re.findall('(key\d):(.*)\s*(value\d?):?\s*(.*)', content)
        bad_res = False
        for find_res in find_list:
            if len(find_res) != 4:
                bad_res = True
                break
            result[strip(find_res[1])] = strip(find_res[3])
            pass
        if bad_res:
            result = {}
    else:
        find_list = re.findall('(.*):(.*)', content)
        bad_res = False
        for find_res in find_list:
            if len(find_res) != 2:
                bad_res = True
                break
            if 'key' in find_res[0]:
                bad_res = True
                break
            result[strip(find_res[0])] = strip(find_res[1])
    return result


def industry_classification_result_process(sub_df):
    '''行业情感信息抽取的数据处理'''
    contents_adj = []
    for i in range(sub_df.shape[0]):
        print(f'=============={i}==============')
        content = sub_df.iloc[i].predict.lstrip().rstrip()
        content = content.replace('：', ':').replace('，', ',').replace(' ', '').replace('；', ';').replace('。',
                                                                                                         '.').replace(
            ',\n', '\n')
        match_result = re.match('行业:(.*)情感分类:(.*)', content, re.DOTALL)
        class_info_dict = {}
        if match_result is not None:
            ind_info, class_info = match_result.groups()
            ind_info = ind_info.strip('\n.;')
            class_info = class_info.strip('\n.;')
            num_infos = get_num_infos(class_info)
            ## 情感分类为多个项目
            if len(num_infos) > 0:
                sep = num_infos[0]['sep']
                class_info_list = class_info.split(sep)
                ## 情感分类为二阶
                try:
                    if len(num_infos) > 1:
                        sec_sep = num_infos[1]['sep']
                        class_info_dict = dict([tuple(x.split(':')) for x in class_info_list])
                    else:
                        ## 情感分类为一阶
                        num_infos = get_num_infos(ind_info)
                        if len(num_infos) > 0:
                            sep = num_infos[0]['sep']
                            ind_info_list = ind_info.split(sep)
                        class_info_list_adj = extract_re_exprs(class_info_list, ['.*[为是](.*)', '.*（(.*)）'])
                        if None not in class_info_list_adj:
                            class_info_dict = dict(zip(ind_info_list, class_info_list_adj))
                        else:
                            max_len = max(len(ind_info_list), len(class_info_list))
                            class_info_list = pad_list(class_info_list, max_len, '无')
                            class_info_dict = dict(zip(ind_info_list, class_info_list))
                except:
                    find_res = re.findall('(\w+)[:\s]+([正|负|中|无])\w+', class_info)
                    class_info_dict = {x[0]: x[1] for x in find_res}
                    pass
            else:
                num_infos = get_num_infos(ind_info)
                if len(num_infos) > 0:
                    sep = num_infos[0]['sep']
                    ind_info_list = ind_info.split(sep)
                    class_info_dict = dict(zip(ind_info_list, [class_info] * len(ind_info_list)))
                else:
                    class_info_dict = {ind_info: class_info}
                pass
        else:
            if '抽取结果' in content:
                match_result = re.match('抽取结果.*?:(.*)', content, re.DOTALL)
                if match_result is not None:
                    match_result = strip(match_result.groups()[0])
                    match_result1 = re.match('.*情感分类结果:(.*)', match_result, re.DOTALL)
                    if match_result1 is not None:
                        class_info = match_result1.groups()[0]
                        # num_infos = [x for x in sorted(get_num_infos(ind_info), key=lambda x: x['num']) if x['num'] > 0]
                        num_infos = get_num_infos(class_info)
                        if len(num_infos) > 0:
                            sep = num_infos[0]['sep']
                            class_info_list = class_info.split(sep)
                        match_result1 = re.match('(.*)情感分类结果', match_result, re.DOTALL)
                        if match_result1 is not None:
                            ind_info = match_result1.groups()[0].strip()
                            # num_infos = [x for x in sorted(get_num_infos(ind_info), key=lambda x: x['num']) if x['num'] > 0]
                            num_infos = get_num_infos(class_info)
                            if len(num_infos) > 0:
                                sep = num_infos[0]['sep']
                                ind_info_list = ind_info.split(sep)
                                if len(class_info_list) == len(ind_info_list):
                                    class_info_dict = dict(zip(ind_info_list, class_info_list))
                                else:
                                    max_len = max(len(ind_info_list), len(class_info_list))
                                    ind_info_list = pad_list(ind_info_list, max_len, '无')
                                    class_info_list = pad_list(class_info_list, max_len, '无')
                                    class_info_dict = dict(zip(ind_info_list, class_info_list))
                    else:
                        try:
                            num_infos = get_num_infos(match_result)
                            if len(num_infos) > 0:
                                sep = num_infos[0]['sep']
                                ind_info_list = match_result.split(sep)
                                if len(num_infos) > 1:
                                    sep = num_infos[1]['sep']
                                    class_info_dict = dict([tuple(x.split(sep)) for x in ind_info_list])
                                else:
                                    class_info_dict = dict(zip(ind_info_list, ['无'] * len(ind_info_list)))
                                    pass
                            else:
                                class_info_dict = {match_result: '无'}
                                pass
                        except:
                            find_res = re.findall('(\w+)[:\s]+([正|负|中|无])\w+', match_result)
                            class_info_dict = {x[0]: x[1] for x in find_res}
                            pass
                        pass
                    pass
                else:
                    find_res = re.findall('(\w+)[:\s]+([正|负|中|无])\w+', content)
                    class_info_dict = {x[0]: x[1] for x in find_res}
                    pass
            else:
                find_result = [(strip(x[0]), strip(x[1])) for x in re.findall('^行业:(.*)情感分类:(.*)', content, re.DOTALL)]
                if len(find_result) > 0:
                    ind_info = find_result[0][0]
                    class_info = find_result[0][1]
                    num_infos = get_num_infos(ind_info)
                    if len(num_infos) > 0:
                        sep = num_infos[0]['sep']
                        ind_info_list = ind_info.split(sep)
                        num_infos = get_num_infos(class_info)
                        if num_infos > 0:
                            sep = num_infos[0]['sep']
                            class_info_list = class_info.split(sep)
                        else:
                            class_info_list = [class_info] * len(ind_info_list)
                else:
                    ## 预测结果如下:xxx:xxx,xxx:xxx
                    find_result = [strip(x) for x in re.findall('^.*结果.*?:(.*)', content, re.DOTALL)]
                    if len(find_result) > 0:
                        class_info_dict = decode_re_content(find_result[0])
                        pass
                    else:
                        class_info_dict = decode_re_content(content)
                pass
        if class_info_dict is None or len(class_info_dict) <= 0:
            find_res = re.findall('(\w+)[:\s]*([正|负|中|无])\w+', content)
            class_info_dict = {x[0]: x[1] for x in find_res}
            if len(class_info_dict) <= 0:
                print(content)
        contents_adj.append(class_info_dict)
    return contents_adj


def process_industry_classification_output(output_adj):
    '''处理行业情感信息抽取的标签，以便进行匹配计算f1'''
    new_output = {}
    for k, v in output_adj.items():
        find_res = re.findall('(正|负|中|无).*', v)
        if len(find_res) == 0:
            continue
        new_output[k] = find_res[0]
    return new_output


def industry_classification_result_process1(sub_df):
    contents_adj = []
    for i in range(sub_df.shape[0]):
        content = sub_df.iloc[i].predict.lstrip().rstrip()
        content = [strip(x) for x in content.split('\n') if len(x) > 0]
        content = '\n'.join(content)
        content = content.replace('：', ':').replace('，', ',').replace('；', ';'). \
            replace('。', '.').replace(',\n', '\n')
        find_res1 = re.findall('(\S+)[:|\s]+(\S+)', content)
        find_res2 = re.findall('(\S+)\s+(\S+)', content)
        if len(find_res1) > len(find_res2):
            find_res = find_res1
        else:
            find_res = find_res2
        result = {x[0]: x[1] for x in find_res if len(re.findall('正|负|中|无', x[1])) > 0}
        contents_adj.append(result)
    return contents_adj


def cal_financial_extract_score(data):
    '''金融事件抽取得分'''
    content = data['response']
    output = data['__raw.output']
    content_adj = financial_extract_result_process_single(content)
    output = output_adj(output)
    output = (lambda a: {x['role']: x['argument'] for x in a})(output)
    f1 = cal_f1(output, content_adj)
    return f1


def cal_industry_classification_score(data):
    '''金融行业情感分类打分'''
    content = data['response']
    output = data['__raw.output']
    contents_adj = industry_classification_result_process_single(content)
    contents_adj = process_industry_classification_output(contents_adj)
    output = (lambda x: process_industry_classification_output(eval(x)))(output)
    f1 = cal_f1(output, contents_adj)
    return f1


def compute_extraction(file_path):
    with open(file_path, 'r', encoding='utf-8') as input_file:
        print("Reading\t" + file_path)
        sample_list = json.load(input_file)
        samples = pd.json_normalize(sample_list, 'outputs')
    samples = samples[samples['__raw.task'] == '金融文本抽取']
    f1_list = []
    for _, row in samples.iterrows():
        label = row['__raw.output']
        pred = row['response']
        if row['__raw.sub_task'] == '金融事件主体抽取':
            if label in pred:
                acc = 1
            else:
                acc = 0

            f1 = 2 * acc * 1 / (acc + 1)
        elif row['__raw.sub_task'] == '金融事件因果关系抽取':
            entities = {'原因类型': 'reason_type',
                        '原因产品': 'reason_product',
                        '原因地区': 'reason_region',
                        '原因行业': 'reason_industry',
                        '结果类型': 'result_type',
                        '结果产品': 'result_product',
                        '结果地区': 'result_region',
                        '结果行业': 'result_industry'}
            not_mentioned = r'未提及|无|无结果|无明确|None'
            acc, recall = 0, 0
            for ent in entities:
                pattern = f'{ent}(.*?)(\n|$)'
                # pattern = f'(?<={ent}\nvalue[1-8]: ).*'  # for gpt4
                try:
                    _label = label[0][entities[ent]]  # 实体对应答案
                except TypeError:
                    _label = eval(label)[0][entities[ent]]
                if re.findall(pattern, pred):  # 有对应抽取结果
                    _pred = ''.join(s for s in re.findall(pattern, pred)[0] if s not in [',', '，', '。', '.', '、', ';',
                                                                                         '；', ':', '：', ' '])
                    recall += 1
                    # print(_pred)
                    if re.findall(not_mentioned, _pred):  # 模型回答为空
                        if _label == '':
                            acc += 1
                        elif _label in _pred:
                            acc += 1
                        else:
                            pass
                    else:
                        if _pred == _label:
                            acc += 1
                        else:
                            pass
                else:  # 无抽取结果，acc、recall均不得分
                    pass

            acc /= 8
            recall /= 8
            if acc + recall == 0:
                f1 = 0
            else:
                f1 = 2 * acc * recall / (acc + recall)
        elif row['__raw.sub_task'] == '金融事件抽取':
            f1 = cal_financial_extract_score(row) / 100
        elif row['__raw.sub_task'] == '行业情感信息抽取':
            f1 = cal_industry_classification_score(row) / 100
        else:
            f1 = 0
        f1_list.append(f1)
    samples['f1'] = f1_list
    return samples['f1'].mean(), samples[['__raw.sub_task', 'f1']].groupby('__raw.sub_task').mean().to_dict()['f1']


def main(model, path):
    files = os.listdir(f'{path}')

    qa_bert_list = []
    tg_bert_list, sub_task_tg_bert_list = [], []
    qa_list, tg_list, e2z_bleu_list, e2z_comet_list, z2e_bleu_list, z2e_comet_list, comet_list, acc_list, f1_list = [], [], [], [], [], [], [], [], []
    sub_task_qa_list, sub_task_tg_list, sub_task_tc_list, sub_task_re_list = [], [], [], []
    for f in files:
        if model in f:
            print('Model: %s' % model)
            file_path = f'{path}/{f}'
            # QA
            rouge_l, qa_bert = compute_finqa(file_path)
            qa_list.append(rouge_l)
            qa_bert_list.append(qa_bert)
            # TG
            rouge_l_tg, sub_task_tg, tg_bert, tg_sub_task_bert = compute_text_generation(file_path)
            tg_list.append(rouge_l_tg)
            sub_task_tg_list.append(sub_task_tg)
            tg_bert_list.append(tg_bert)
            sub_task_tg_bert_list.append(tg_sub_task_bert)
            # MT-e2zh
            bleu, comet = compute_nmt_en2zh(file_path)
            e2z_bleu_list.append(bleu)
            e2z_comet_list.append(comet)
            # MT-zh2e
            bleu, comet = compute_nmt_zh2en(file_path)
            z2e_bleu_list.append(bleu)
            z2e_comet_list.append(comet)
            # TC
            acc, sub_task_acc = compute_text_classification(file_path)
            acc_list.append(acc)
            sub_task_tc_list.append(sub_task_acc)
            # RE
            f1, sub_task_re = compute_extraction(file_path)
            f1_list.append(f1)
            sub_task_re_list.append(sub_task_re)

    # 总分类
    print('QA mean: %s' % np.mean(qa_list), '\n')
    print('QA Std: %s' % np.std(qa_list), '\n')
    print('QA bert mean: %s' % np.mean(qa_bert_list), '\n')
    print('QA bert Std: %s' % np.std(qa_bert_list), '\n')
    print('TG mean: %s' % np.mean(tg_list), '\n')
    print('TG Std: %s' % np.std(tg_list), '\n')
    print('TG bert mean: %s' % np.mean(tg_bert_list), '\n')
    print('TG bert Std: %s' % np.std(tg_bert_list), '\n')
    print('EN2CH mean: %s' % np.mean(e2z_bleu_list), '\n')
    print('EN2CH Std: %s' % np.std(e2z_bleu_list), '\n')
    print('EN2CH comet mean: %s' % np.mean(e2z_comet_list), '\n')
    print('EN2CH comet Std: %s' % np.std(e2z_comet_list), '\n')
    print('CH2EN mean: %s' % np.mean(z2e_bleu_list), '\n')
    print('CH2EN Std: %s' % np.std(z2e_bleu_list), '\n')
    print('CH2EN comet mean: %s' % np.mean(z2e_comet_list), '\n')
    print('CH2EN comet Std: %s' % np.std(z2e_comet_list), '\n')
    print('TG mean: %s' % np.mean(tg_list), '\n')
    print('TG Std: %s' % np.std(tg_list), '\n')

    ## 子任务分类
    # TG
    tg_df = pd.DataFrame(sub_task_tg_list)
    print('TG sub task mean: %s' % tg_df.mean())
    print('TG sub task std: %s' % tg_df.std())
    tg_bert_df = pd.DataFrame(sub_task_tg_bert_list)
    print('TG sub task bert mean: %s' % tg_bert_df.mean())
    print('TG sub task bert std: %s' % tg_bert_df.std())
    # TC
    tc_df = pd.DataFrame(sub_task_tc_list)
    print('TG sub task mean: %s' % tc_df.mean())
    print('TG sub task std: %s' % tc_df.std())
    # RE
    re_df = pd.DataFrame(sub_task_re_list)
    print('TG sub task mean: %s' % re_df.mean())
    print('TG sub task std: %s' % re_df.std())
