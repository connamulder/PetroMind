"""
    @Project: PetraMind
    @File   : 04_train.py
    @Author : mulder
    @E-mail : c_mulder@163.com
    @Date   : 2025-04-07
    @Info   : 基于 transformers、peft 等框架，使用 Qwen2.5-VL-7B-Instruct 模型在COCO2014图像描述 上进行LoRA微调训练，
              同时使用 SwanLab 监控训练过程与评估模型效果。
    @Date   : 2025-06-06
    @Info   : 增加代码对Linux系统的适配
    @Ref    : https://blog.csdn.net/SoulmateY/article/details/143807035
    @Date   : 2025-07-22
    @Info   : 增加早停机制
    @Date   : 2025-07-28
    @Info   : LoRA两阶段微调，发现第二次可训练参数全为零，LoRA微调无法执行，改为任务适配器合并。
"""


import os
import argparse
import json
from dotenv import load_dotenv
import sys
import pickle
import matplotlib.pyplot as plt
import contextlib

import torch
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
# from modelscope import Qwen2VLForConditionalGeneration
# from modelscope import LlamaForCausalLM
from qwen_vl_utils import process_vision_info

from peft import LoraConfig, TaskType, get_peft_model, PeftModel
import swanlab
from swanlab.integration.transformers import SwanLabCallback

from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

from datasets import Dataset, disable_caching

import cal_acc_draw_confusion_matrix
from utils.utils_result import print_result_to_excel
from classes.earlystopping import EarlyStopping


# 是否为编码阶段的测试，编码阶段加载少量数据
is_coding_test = False

parser = argparse.ArgumentParser()
parser.add_argument("--do_train", action='store_true', help="is do training")
parser.add_argument("--lora_rank", type=int, default=64, help="16 | 32 | 64 | 96| 128")
parser.add_argument("--task_type", type=str, default='class', help=" class | cap | multi | merge")
parser.add_argument("--epoch", type=int, default=1, help="")
parser.add_argument("--train_epoch", type=int, default=3, help="")
parser.add_argument("--batch_size", type=int, default=8, help="16 | 64不行")
parser.add_argument("--img_size", type=int, default=224, help="size of image")
parser.add_argument("--model_type", type=str, default='7B', help="3B | 7B | 32B-耗尽 | 72B")
parser.add_argument("--data_train", type=str, default='rocks13_random_tv_augu_224_train_imbalance.json', help="training data")
parser.add_argument("--data_val", type=str, default='rocks13_random_tv_augu_224_val.json', help="testing data")
parser.add_argument("--swanlab_online", action='store_true', help="is use online swanlab")
opt = parser.parse_args()
print(opt)

if is_coding_test:
    opt.do_train = True

task_list = ["class", "cap", "multi", "merge"]
dataset_list = ["rock13_class", "rock13_caption", "rock13_merge"]

device = ''
cuda_list = ""
num_gpus = torch.cuda.device_count()
if num_gpus == 1:
    cuda_list = "0"
    device = "cuda:0"
elif num_gpus == 2:
    cuda_list = "0, 1"
    device = "cuda:1"
elif num_gpus >= 3:
    cuda_list = ", ".join([f"{i}" for i in range(num_gpus)])
print("供使用的GPU列表：%s" % cuda_list)

image_size = opt.img_size
model_name = 'Qwen/Qwen2.5-VL-%s-Instruct' % opt.model_type

dataset_name = ""
task_type = opt.task_type
num_patience = 3
if "merged" in opt.data_train:
    dataset_name = dataset_list[2]
    num_patience = 1
elif "label_vl" in opt.data_train:
    dataset_name = dataset_list[1]
    num_patience = 3
elif "tv_augu" in opt.data_train:
    dataset_name = dataset_list[0]
    num_patience = 1

project_name = "Qwen-VL-%s_finetune" % dataset_name

# 使用Transformers加载模型权重
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_name)


def process_func(example):
    """
    将数据集进行预处理
    """
    MAX_LENGTH = 8192
    input_ids, attention_mask, labels = [], [], []
    conversation = example["conversations"]
    input_content = conversation[0]["value"]
    output_content = conversation[1]["value"]
    str_prompt_temp = input_content.split("<|vision_start|>")[0]
    file_path = input_content.split("<|vision_start|>")[1].split("<|vision_end|>")[0]  # 获取图像路径
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"{file_path}",
                    "resized_height": image_size,
                    "resized_width": image_size,
                },
                {"type": "text", "text": {str_prompt_temp}},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )  # 获取文本
    image_inputs, video_inputs = process_vision_info(messages)  # 获取数据数据（预处理过）
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.tolist() for key, value in inputs.items()}  # tensor -> list,为了方便拼接
    instruction = inputs

    response = tokenizer(f"{output_content}", add_special_tokens=False)

    input_ids = (
            instruction["input_ids"][0] + response["input_ids"] + [tokenizer.pad_token_id]
    )

    attention_mask = instruction["attention_mask"][0] + response["attention_mask"] + [1]
    labels = (
            [-100] * len(instruction["input_ids"][0])
            + response["input_ids"]
            + [tokenizer.pad_token_id]
    )
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    labels = torch.tensor(labels)
    inputs['pixel_values'] = torch.tensor(inputs['pixel_values'])
    inputs['image_grid_thw'] = torch.tensor(inputs['image_grid_thw']).squeeze(0)  # 由（1,h,w)变换为（h,w）
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels,
            "pixel_values": inputs['pixel_values'], "image_grid_thw": inputs['image_grid_thw']}


def predict(messages, model):
    # 准备推理
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # return_tensors='tf'返回TensorFlow张量；'pt'返回PyTorch张量；'np'返回Numpy数组。
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # 生成输出
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]


model = None
if num_gpus > 1:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map='auto',  # auto, balanced_low_0
    )
else:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,  # auto, balanced_low_0
    )
if opt.do_train:
    model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法
    model.config.use_cache = False

# 配置LoRA
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=opt.lora_rank,  # Lora 秩
    lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.05,  # Dropout 比例
    bias="none",
)

# 配置测试参数
val_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=True,  # 训练模式
    r=opt.lora_rank,  # Lora 秩
    lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.05,  # Dropout 比例
    bias="none",
)

# -------------------------------  设置checkpoint和log输出路径 --------------------------------- #
output_type_name = '%s-%d' % (opt.model_type, opt.lora_rank)

# checkpoint输出路径
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir_str = ''
if sys.platform == 'linux':
    output_dir = '/home/u2021800205/VQA/output'
    output_dir_str = os.path.join(output_dir, output_type_name)
elif sys.platform == 'win32':
    output_dir = 'output'
    output_dir_str = os.path.join(current_dir, output_dir, output_type_name)
if not os.path.exists(output_dir_str):
    os.makedirs(output_dir_str)
    print("创建output_dir_str: %s" % output_dir_str)
else:
    print("output_dir_str: %s已存在" % output_dir_str)

# 离线看板
log_dir = ''
if sys.platform == 'linux':
    log_dir = '/home/u2021800205/VQA/logs'
elif sys.platform == 'win32':
    log_dir = './logs'
log_dir = os.path.join(log_dir, output_type_name)

if opt.swanlab_online:
    # 在线看板
    env_path = './swanlab.env'
    load_dotenv(dotenv_path=env_path, verbose=True)
    SWANLAB_API_KEY = os.getenv("SWANLAB_API_KEY", default="")
    swanlab.login(api_key=SWANLAB_API_KEY)


def Check_valuate(check_dir, tag, data_name, check_dataset, check_swanlab=None):

    if tag == "merge":
        class_check_dir = os.path.join(check_dir, task_list[0])
        merged_model = PeftModel.from_pretrained(model, model_id=class_check_dir, adapter_name=task_list[0])

        cap_check_dir = os.path.join(check_dir, task_list[1])
        _ = merged_model.load_adapter(model_id=cap_check_dir, adapter_name=task_list[1])

        adapters = [task_list[0], task_list[1]]
        weights = [1.0, 1.0]
        adapter_name = task_list[3]
        density = 0.2
        merged_model.add_weighted_adapter(adapters, weights, adapter_name, combination_type="ties", density=density)
        merged_model.set_adapter(adapter_name)
    else:
        check_dir = os.path.join(check_dir, tag)
        val_peft_model = PeftModel.from_pretrained(model, model_id=check_dir, config=val_config)
        merged_model = val_peft_model.merge_and_unload()

    test_image_list = []
    val_index = 0
    y_pred_labels = []
    y_true_labels = []
    caption_list = []
    for item in check_dataset:
        """
        if val_index > 100:
            break
        """

        input_image_prompt = item["conversations"][0]["value"]
        input_caption = item["conversations"][1]["value"]
        # 去掉前后的<|vision_start|>和<|vision_end|>
        str_prompt_temp = input_image_prompt.split("<|vision_start|>")[0]
        # str_prompt_temp = "Rock lithological description Yes: "
        origin_image_path = input_image_prompt.split("<|vision_start|>")[1].split("<|vision_end|>")[0]

        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": origin_image_path
                },
                {
                    "type": "text",
                    "text": str_prompt_temp
                }
            ]}]

        response = predict(messages, merged_model)
        if "Rock image Yes" in str_prompt_temp:
            y_pred_labels.append(response)
            y_true_labels.append(input_caption)

            messages.append({"role": "assistant", "content": f"{response}"})
            print("%s--%s" % (messages[-1], input_caption))

            test_image_list.append(check_swanlab.Image(origin_image_path, caption=response))
            val_index += 1
        elif "Rock lithological description Yes" in str_prompt_temp:
            caption_list.append({'image': origin_image_path,  'caption': response, 'ref_caption': input_caption, 'type': ''})

    if len(y_pred_labels) > 0 and len(y_true_labels) > 0:
        preds_pkl_name = 'val_preds_%s_%s_%s.pkl' % (opt.model_type, tag, data_name)
        preds_pkl_dir = os.path.join(output_dir_str, preds_pkl_name)
        # 保存List对象到文件
        with open(preds_pkl_dir, 'wb') as file:
            pickle.dump(y_pred_labels, file)

        trues_pkl_name = 'val_trues_%s_%s_%s.pkl' % (opt.model_type, tag, data_name)
        trues_pkl_dir = os.path.join(output_dir_str, trues_pkl_name)
        with open(trues_pkl_dir, 'wb') as file:
            pickle.dump(y_true_labels, file)

        acc_class, f1_macro = cal_acc_draw_confusion_matrix.cal_acc_draw_cf(preds_pkl_dir, trues_pkl_dir, res_dir=output_dir_str)
        if check_swanlab is not None:
            check_swanlab.log({"rock image classification acc": acc_class}, 0)
            check_swanlab.log({"rock image classification f1_score_macro": f1_macro}, 0)

    if len(caption_list) > 0:
        evaluate_file_name = '%s_Qwen_VL_%s_%s.xlsx' % ('result', opt.model_type, task_type)
        evaluate_path = os.path.join(output_dir_str, evaluate_file_name)
        print_result_to_excel(txt_filepath=evaluate_path, result=caption_list)
        if swanlab is not None:
            check_swanlab.log({"rock image caption acc": 1}, 0)


if opt.do_train:
    train_json_path = opt.data_train
    train_json_path = os.path.join('data', train_json_path)

    # 在全局范围内禁用 Datasets 缓存
    # disable_caching()
    # 处理数据集：读取json文件
    if os.path.exists(train_json_path):
        train_ds = Dataset.from_json(train_json_path)
        if is_coding_test:
            train_ds = train_ds.select(range(100))
    else:
        print("训练文件%s不存在" % train_json_path)
        exit(0)

    eval_json_path = opt.data_val
    eval_json_path = os.path.join('data', eval_json_path)
    # 处理数据集：读取json文件
    if os.path.exists(eval_json_path):
        eval_ds = Dataset.from_json(eval_json_path)
        if is_coding_test:
            eval_ds = eval_ds.select(range(20))
    else:
        print("验证文件%s不存在" % eval_json_path)
        exit(0)

    # 配置训练参数
    args = TrainingArguments(
        output_dir=output_dir_str,
        per_device_train_batch_size=opt.batch_size,
        gradient_accumulation_steps=4,
        logging_steps=10,
        logging_first_step=1,
        num_train_epochs=opt.train_epoch,
        eval_strategy="no",
        save_strategy="no",
        learning_rate=1e-4,
        gradient_checkpointing=True,
        report_to="none",
    )
    # eval_strategy="epoch",
    # save_strategy="epoch",

    # 设置SwanLab回调
    experiment_name = 'qwen-vl-%s-%s_%s_%d' % (opt.model_type, dataset_name, task_type, opt.lora_rank)
    if opt.swanlab_online:
        swanlab_callback = SwanLabCallback(
            project=project_name,
            experiment_name=experiment_name,
            config={
                "model": model_name,
                "dataset": dataset_name,
                "github": "",
                "prompt": "",
                "train_data_number": len(train_ds),
                "lora_rank": opt.lora_rank,
                "lora_alpha": 16,
                "lora_dropout": 0.1,
            },
        )
    else:
        swanlab_callback = SwanLabCallback(
            logdir=log_dir,
            mode="local",
            project=project_name,
            experiment_name=experiment_name,
            config={
                "model": model_name,
                "dataset": dataset_name,
                "github": "",
                "prompt": "",
                "train_data_number": len(train_ds),
                "lora_rank": opt.lora_rank,
                "lora_alpha": 16,
                "lora_dropout": 0.1,
            },
        )

    train_dataset = train_ds.map(process_func, load_from_cache_file=True)
    eval_dataset = eval_ds.map(process_func, load_from_cache_file=True)

    # 获取LoRA模型
    peft_model = get_peft_model(model, config)
    peft_model.print_trainable_parameters()

    # 现在保存到文件
    output_file = "trainable_params_output.txt"
    output_file_path = os.path.join(output_dir_str, output_file)
    with open(output_file_path, "w", encoding="utf-8") as f:
        with contextlib.redirect_stdout(f):
            peft_model.print_trainable_parameters()

    # 配置Trainer
    trainer = Trainer(
        model=peft_model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        callbacks=[swanlab_callback],
    )

    save_dir_str = os.path.join(output_dir_str, task_type)
    early_stopper = EarlyStopping(patience=num_patience, mode='min', save_path=save_dir_str)

    loss_history = []
    loss_val_history = []

    for epoch in range(opt.epoch):
        train_results = trainer.train()  # 训练一个epoch
        eval_results = trainer.evaluate()  # 评估验证集
        loss = train_results.metrics['train_loss']
        loss_history.append(loss)
        val_loss = eval_results["eval_loss"]  # 获取验证集损失
        loss_val_history.append(val_loss)
        print(
            "[Epoch %d/%d] [loss: %f] [eval_loss: %f]"
            % (epoch, opt.epoch, loss, val_loss)
        )

        early_stopper(val_loss, peft_model)  # 检查是否触发早停
        if early_stopper.early_stop:
            print(f"Early stopping triggered at epoch {epoch}!")
            break

    # --------------------------- Display performance --------------------------- #
    # plot loss
    plt.plot(loss_history, label='loss')
    plt.plot(loss_val_history, label='val_loss')
    plt.legend()
    plt.savefig('%s/train_plot_%s.png' % (output_dir_str, task_type))
    plt.savefig('%s/train_plot_%s.svg' % (output_dir_str, task_type))
    plt.clf()

    # --------------------------- Save performance ------------------------------ #
    loss_train_fname = "{}/train_loss_{}.csv".format(output_dir_str, task_type)
    with open(loss_train_fname, 'w') as train_loss_csvfile:
        for item in loss_history:
            train_loss_csvfile.write("%s\n" % item)
    loss_val_fname = "{}/val_loss_{}.csv".format(output_dir_str, task_type)
    with open(loss_val_fname, 'w') as train_val_csvfile:
        for item in loss_val_history:
            train_val_csvfile.write("%s\n" % item)

else:
    if "merged" in opt.data_val:
        dataset_name = dataset_list[2]
    elif "label_vl" in opt.data_val:
        dataset_name = dataset_list[1]
    elif "tv_augu" in opt.data_val:
        dataset_name = dataset_list[0]

    # 读取测试数据
    test_json_path = opt.data_val
    test_json_path = os.path.join('data', test_json_path)
    if os.path.exists(test_json_path):
        with open(test_json_path, "r") as f:
            test_dataset = json.load(f)
            if is_coding_test:
                test_dataset = test_dataset[:2]
    else:
        print("测试文件%s不存在" % test_json_path)
        exit(0)

    experiment_name = 'qwen-vl-%s-%s_val_%s_%d' % (opt.model_type, dataset_name, task_type, opt.lora_rank)
    swanlab.init(
        logdir=log_dir,
        mode="local",
        project=project_name,
        experiment_name=experiment_name,
        config={
            "model": model_name,
            "dataset": dataset_name,
            "github": "",
            "prompt": "",
            "train_data_number": len(test_dataset),
            "lora_rank": opt.lora_rank,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
        },
    )

    Check_valuate(output_dir_str, task_type, dataset_name, test_dataset, swanlab)
