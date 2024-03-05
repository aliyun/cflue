import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from modelscope import Model
from modelscope.models.nlp.llama2 import Llama2Tokenizer


def load_models_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path,
                                              use_fast=False,
                                              trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint_path,
                                                 device_map="auto",
                                                 torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True).eval()
    model.generation_config = GenerationConfig.from_pretrained(args.checkpoint_path)

    model.generation_config.do_sample = False  # use greedy decoding
    model.generation_config.repetition_penalty = 1.0  # disable repetition penalty
    return model, tokenizer


def load_llama_models_tokenizer(args):
    tokenizer = Llama2Tokenizer.from_pretrained(args.checkpoint_path)
    model = Model.from_pretrained(
        args.checkpoint_path,
        device_map=f'cuda:{args.gpu}')
    model.generation_config.do_sample = False  # use greedy decoding
    model.generation_config.repetition_penalty = 1.0  # disable repetition penalty
    return model, tokenizer

