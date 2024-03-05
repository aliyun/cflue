import argparse
from exec_knowledge_eval import eval_ability
from exec_application_eval import eval_application

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--checkpoint-path", required=True, type=str)
parser.add_argument('--model_name', required=True, type=str)
parser.add_argument('--save_result_dir', required=True, type=str)
parser.add_argument('--eval_type', required=True, type=str)
args = parser.parse_args()


if args.eval_type == 'knowledge':
    eval_ability(args)
elif args.eval_type == 'application':
    eval_application(args)
