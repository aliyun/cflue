import re
import random


def extract_one_choice(gen, choice_list):
    if "Analysis" in gen:
        gen = gen.split("Analysis")[0]
    else:
        gen = gen.split("解析")[0]
    # 答案是A | 选项是A | 应该选A选项
    res = re.search(
        r"(?:(?:选|选择|选定)[：:]?\s*|(?:(?:答案|选项)(?![^ABCDEF]{0,10}?(?:不|非)[^ABCDEF]{0,10}?(?:是|选|为|：|:|】))[^ABCDEF]{0,10}?(?:是|选|为|：|:|】))[^ABCDEF]{0,10}?)(A|B|C|D|E|F)(?:选项)?(?:\)|。|\.|，|,|．|、|A|B|C|D|E|F|$|：|:|\)|）)",
        gen,
    )

    # A选项正确 | A选项符合题意
    if res is None:
        res = re.search(
            r"(A|B|C|D|E|F)(?:选?项)?(?![^ABCDEF]{0,4}?(?:不|非)[^ABCDEF]{0,4}?(?:正确|对[的，。：]|符合))[^ABCDEF]{0,4}?(?:正确|对[的，。：]|符合)",
            gen,
        )

    # 直接输出 A
    if res is None:
        res = re.search(r"^[\(（]?(A|B|C|D|E|F)(?:。|\)|）|\.|，|,|．|：|:|$)", gen)

    # 获取第一个出现的字母
    if res is None:
        res = re.search(r"(?<![a-zA-Z])(A|B|C|D|E|F)(?![a-zA-Z=])", gen)

    if res is None:
        return random.choice(choice_list)

    return res.group(1)


def extract_multi_choice(gen, choice_list):
    if "Analysis" in gen:
        gen = gen.split("Analysis")[0]
    else:
        gen = gen.split("解析")[0]
    pattern = r"[A-F]"
    choices = re.findall(pattern, gen)
    if choices is None:
        return random.choice(choice_list)
    res = sorted(list(set(choices)))
    return ''.join(res)
