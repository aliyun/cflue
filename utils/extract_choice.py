import re
import random

def extract_one_choice(gen, choice_list):
    return extract_multi_choice(gen, choice_list)


def extract_multi_choice(gen, choice_list):
    gen = gen.strip()
    gen = gen.replace("、", "").replace(",", "").replace("，", "").replace(" ", "")

    # 严格遵循指令的情况
    res = re.search(
        r"答案[是：:]+([ABCDEF]+)",
        gen,
    )

    # 未严格遵循指令，直接输出答案的情况
    if res is None:
        res = re.search(
            r"(^[ABCDEF]+)",
            gen,
        )

    if res is None:
        return "X"

    return res.group(1)