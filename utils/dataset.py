import pandas as pd


def load_knowledge():
    dataset = pd.read_json('../data/knowledge/knowledge.json')
    return dataset


def load_application():
    dataset = pd.read_json('../data/application/application.json')
    return dataset


def load_dataset(eval_type):
    if eval_type == 'knowledge':
        return load_knowledge()
    elif eval_type == 'application':
        return load_application()
    else:
        print(f'{eval_type} not supported. "eval_type" has to be "knowledge" or "application".')
