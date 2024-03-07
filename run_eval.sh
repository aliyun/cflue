#!/bin/bash

model_name=""  # 模型名称
checkpoint_path=""  # 指定模型checkpoint路径
eval_type="knowledge"  # 指定金融知识评测或金融应用评测
save_result_dir="../results"

python cflue_main.py \
    --model_name ${model_name} \
    --checkpoint_path ${checkpoint_path} \
    --eval_type ${eval_type} \
    --save_result_dir ${save_result_dir}