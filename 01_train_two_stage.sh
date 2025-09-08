#!/bin/bash
#### Qwen2.5-VL Rocks class and cap LoRA训练

# "3B" "7B" "32B"
model_types=("32B")

for model in "${model_types[@]}"; do
    echo "Running with model_type=$model"
    python ./01_train_rockset_early.py --do_train --task_type class --train_epoch 3 --model_type $model --lora_rank 32 --data_train rocks13_random_tv_augu_224_train.json --data_val rocks13_random_tv_augu_224_val.json
    python ./01_train_rockset_early.py --task_type class --model_type $model --lora_rank 32 --data_val rocks13_random_tv_augu_224_val.json
    python ./01_train_rockset_early.py --do_train --task_type cap --train_epoch 14 --model_type $model --lora_rank 32 --data_train pic_label_vl_train_two_linux.json --data_val pic_label_vl_val_two_linux.json
    python ./01_train_rockset_early.py --task_type cap --model_type $model --lora_rank 32 --data_val pic_label_vl_val_two_linux.json
    python ./01_train_rockset_evaluator.py --task_type cap --model_type $model --lora_rank 32
    echo ""
done

python ./smsDemo.py --message "岩石图像多任务多适配器训练32B-r32结束！"


