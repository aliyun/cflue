def format_one_example(line):
    template = """假设你是一位金融行业专家，请回答下列问题。
注意：题目是单选题，只需要返回一个最合适的选项，若有多个合适的答案，只返回最准确的即可。
注意：结果只输出两行，第一行只需要返回答案的英文选项(注意只需要返回一个最合适的答案)，第二行进行简要的解析，输出格式限制为：“答案：”，“解析：”。

{question}
{choices}"""

    example = template.format(
        question=line["question"],
        choices='\n'.join([f'{choice}. {content}' for choice, content in eval(line["choices"]).items()])
    )
    return example


def format_multi_example(line):
    template = """假设你是一位金融行业专家，请回答下列问题。
注意：题目是多选题，可能存在多个正确的答案。
注意：结果只输出两行，第一行只需要返回答案的英文选项，第二行进行简要的解释。输出格式限制为：“答案：”，“解析：”。

{question}
{choices}"""

    example = template.format(
        question=line["question"],
        choices='\n'.join([f'{choice}. {content}' for choice, content in eval(line["choices"]).items()])
    )
    return example
