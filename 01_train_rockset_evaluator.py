"""
    @Project: PetraMind
    @File   : 01_train_rockset_evaluator.py
    @Author : mulder
    @E-mail : c_mulder@163.com
    @Date   : 2025-06-26
    @Info   : 训练结果评估
    @Date   : 2025-07-01
    @Info   : 采用段代码，执行颜色、结构、构造和矿物成分信息分离后。准确度大模型自动评估
"""

import os
import pandas as pd
from tqdm import tqdm
import re
from pathlib import Path

from langchain.evaluation import load_evaluator
from langchain.evaluation.schema import EvaluatorType, LLMEvalChain, StringEvaluator
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
# from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
from langchain_community.llms import Tongyi
import dashscope
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--lora_rank", type=int, default=32, help="16 | 32 | 64 | 96| 128")
parser.add_argument("--model_type", type=str, default='7B', help="3B | 7B | 32B | 72B")
parser.add_argument("--task_type", type=str, default='cap', help=" class | cap | multi | merge")
parser.add_argument("--model_online", action='store_true', help="using online model")
parser.add_argument("--criteria_type", type=str, default='sep', help="all: 汇总评价；sep：分别评价")
opt = parser.parse_args()
opt.model_online = False
print(opt)


# output_type_name = 'Qwen-VL-TWO-%s' % opt.model_type
output_type_name = '%s-%d' % (opt.model_type, opt.lora_rank)
root_dir = os.path.abspath('.')
root_dir = os.path.join(root_dir, 'output')

llm = None
if opt.model_online:
    str_key = 'sk-46f1ca28fbe2477cbdb0d6397e0bc846'
    dashscope.api_key = str_key
    # sk-46f1ca28fbe2477cbdb0d6397e0bc846  原sk-02aebf11c8144232821eec5c90fc7d9e
    llm = Tongyi(model_name='qwen-max', dashscope_api_key=str_key)
    # llm = ChatOpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", model='qwen-max')
else:
    llm = OllamaLLM(model="qwen2.5:72b")


# llm = Tongyi(temperature=1)
split_prompt = """
                您是一位专业的地质学家，需要严格按照以下规则将岩石描述文本进行分类：
                【分类规则】
                    1. 颜色：仅包含岩石整体颜色描述（如灰色、浅红色）和风化面/新鲜面颜色差异。
                    - 示例：灰黑色、浅红色、风化面呈褐黄色
                    - 注意：单个矿物的颜色不属于此类
                    - 注意：岩石风化较弱、岩石较新鲜、岩石发生蚀变等岩石表面状态的描述也属于此类
                    2. 结构：描述矿物颗粒的相互关系或排列方式
                    - 包括：粒度（粗/中/细粒）、结晶程度、颗粒形状（等粒/不等粒）等
                    - 示例：粒状结构、花岗结构、斑状结构
                    3. 构造：严格限定于以下类型：
                    - 块状/带状/斑杂/流纹/枕状/气孔及杏仁/原生片麻构造
                    - 示例：发育气孔构造、具带状构造
                    4. 矿物成分：包含
                    - 矿物种类（角闪石、钾长石等）及其比例
                    - 矿物特征（自形/半自形/它形）
                    - 矿物特定颜色（如"角闪石呈灰黑色"）
                    - 矿物分布关系
                    - 示例：含30%石英（它形粒状）、黑云母（片状，含量约15%）
                    【特殊规则】
                    - 分类截取文字，不能改变语序，所有文字都要分到上述4种类别中，不能有遗漏
                    - 所有矿物专属颜色描述必须归入"矿物成分"
                    - "岩石组成"相关内容全部属于"矿物成分"
                    - 当出现"呈xx色"时，根据主语判断：
                      * "岩石呈xx色"→颜色类
                      * "[矿物名称]呈xx色"→矿物成分类
                    【输出格式要求
                    必须严格按照以下格式，保留原文所有文字：
                        【颜色】[内容]
                        【结构】[内容] 
                        【构造】[内容]
                        【矿物成分】[内容]
                    待分类文本：
                        {text} 
                    """

accuracy_criteria_dict = {}

if opt.criteria_type == "all":
    accuracy_criteria_all = \
    {
        "accuracy": """
            分数 0: 答案中未出现任何与颜色、结构、构造和矿物成分有关的描述。
            分数 3: 答案中出现与颜色、结构、构造和矿物成分相关的描述，但在描述的细节上与参考文献均不一致。
            分数 5: 答案中出现与颜色、结构、构造和矿物成分相关的描述，但在描述的细节上与参考文献存在一致，也存在不一致的地方。
            分数 7: 答案中出现与颜色、结构、构造和矿物成分相关的描述，在描述的细节上与参考文献大部分一致，存在少量不一致的地方。
            分数 9: 答案中出现与颜色、结构、构造和矿物成分相关的描述，在描述的细节上与参考文献基本一致。
            分数 10: 答案中出现与颜色、结构、构造和矿物成分相关的描述，在描述的细节上也与参考文献完全一致。
        """
    }
    accuracy_criteria_dict["汇总"] = accuracy_criteria_all
elif opt.criteria_type == "sep":
    accuracy_criteria_color = \
    {
        "accuracy": """
        分数 0: 答案中未出现任何与新鲜面、新鲜色、风化面、风化色有关的颜色描述。
        分数 3: 答案中出现与颜色相关的描述，颜色描述与参考文献中的颜色描述完全不一致。
        分数 5: 答案中出现与颜色相关的描述，所描述的颜色与参考文献中的颜色色系差异较大。
        分数 7: 答案中出现与颜色相关的描述，所描述的颜色与参考文献中的颜色色系相近。
        分数 10: 答案中出现与颜色相关的描述，颜色描述与参考文献完全一致。
    """
    }
    accuracy_criteria_dict["颜色"] = accuracy_criteria_color

    accuracy_criteria_texture = \
    {
        "accuracy": """
        分数 0: 答案中未出现任何与结构有关的描述。
        分数 3: 答案中出现与结构相关的描述，结构描述与参考文献完全不相关。
        分数 5: 答案中出现与结构相关的描述，结构描述与参考文献部分相关。
        分数 7: 答案中出现与结构相关的描述，结构描述的核心特征与参考答案基本一致。
        分数 10: 答案中出现与结构相关的描述，结构描述与参考文献完全一致，存在微小差异。
        注意：答案仅需要与参考文献进行比对，比较两者差异，只考虑正确性，不要求进一步细化描述。
    """
    }
    accuracy_criteria_dict["结构"] = accuracy_criteria_texture

    accuracy_criteria_struture = \
    {
        "accuracy": """
        分数 0: 答案中未出现任何与构造有关的描述。
        分数 3: 答案中出现与构造相关的描述，构造描述与参考文献完全不相关。
        分数 5: 答案中出现与构造相关的描述，构造描述与参考文献部分相关。
        分数 7: 答案中出现与构造相关的描述，构造描述的核心特征与参考答案基本一致，存在微小差异。
        分数 10: 答案中出现与构造相关的描述，构造描述与参考文献完全一致。
        注意：答案仅需要与参考文献进行比对，比较两者差异，只考虑正确性，不要求进一步细化描述。
    """
    }
    accuracy_criteria_dict["构造"] = accuracy_criteria_struture

    # 存在一些重要的遗漏和不完全准确之处
    accuracy_criteria_minerals = \
    {
        "accuracy": """
        分数 0: 答案中未出现任何与主要矿物、次要矿物、副矿物、特征矿物等有关的矿物成分描述。
        分数 3: 答案中出现与矿物成分相关的描述，但在矿物成分描述细节上与参考文献均不一致。
        分数 5: 答案中出现与矿物成分相关的描述，在矿物成分描述细节上与参考文献有相关性，但存在不准确之处。
        分数 7: 答案中出现与矿物成分相关的描述，在矿物成分描述细节上大部分与参考文献一致，但有一些小错误或遗漏。
        分数 9: 答案中出现与矿物成分相关的描述，在矿物成分描述细节上也与参考文献基本一致。
        分数 10: 答案中出现与矿物成分相关的描述，在矿物成分描述细节上也与参考文献完全一致。
    """
    }
    accuracy_criteria_dict["矿物成分"] = accuracy_criteria_minerals


def read_qa_result_list_from_excel(excel_path):
    data_excel = pd.read_excel(excel_path)
    qa_list = []

    for i in range(0, len(data_excel)):
        question = data_excel.iloc[i]['image']
        answer = data_excel.iloc[i]['caption']
        ref_answer = data_excel.iloc[i]['ref_caption']
        qa_list.append({'image': question, 'caption': answer, 'ref_caption': ref_answer})

    return qa_list


def print_result_to_excel(txt_filepath, result: list):
    keys = [key for key in result[0].keys()]
    records = {}

    for key in keys:
        records[key] = []
    for record in result:
        for key in keys:
            records[key].append(record[key])

    pf_obj = pd.DataFrame(records)
    pf_obj.to_excel(txt_filepath)

    print('保存结果文件至{}。'.format(txt_filepath))


def split_by_category_regex(text):
    # text = text.content
    pattern = re.compile(r'【(.*?)】(.*?)(?=\n【|$)', re.DOTALL)
    matches = pattern.findall(text)

    return {category.strip(): content.strip() for category, content in matches}


def print_dict_result_to_excel(txt_filepath, result: dict):
    with pd.ExcelWriter(txt_filepath) as writer:
        for criteria_type in result:
            results_type = result[criteria_type]
            keys = [key for key in results_type[0].keys()]
            records = {}

            for key in keys:
                records[key] = []
            for record in results_type:
                for key in keys:
                    records[key].append(record[key])
            for key in keys:
                if len(records[key]) > 0:
                    pf_obj = pd.DataFrame(records)
                    pf_obj.to_excel(writer, sheet_name=criteria_type, index=False)
                    print('保存{}结果到文件{}。'.format(criteria_type, txt_filepath))
                    break
    print('保存结果到文件{}。'.format(txt_filepath))


def print_dict_list_to_excel(txt_filepath, results: list, txt_origin):
    output_obj = {'origin': [], '颜色': [], '结构': [], '构造': [], '矿物成分': []}
    for result_item, txt_origin_item in zip(results, txt_origin):
        output_obj['origin'].append(txt_origin_item)
        output_obj['颜色'].append(result_item['颜色'])
        output_obj['结构'].append(result_item['结构'])
        output_obj['构造'].append(result_item['构造'])
        output_obj['矿物成分'].append(result_item['矿物成分'])
    pf_obj = pd.DataFrame(output_obj)
    pf_obj.to_excel(txt_filepath)


if __name__ == '__main__':
    result_dir = os.path.join(root_dir, output_type_name)

    result_file_name = '%s_Qwen_VL_%s_%s.xlsx' % ('result', opt.model_type, opt.task_type)
    result_path = os.path.join(root_dir, output_type_name, result_file_name)
    qa_result_list = read_qa_result_list_from_excel(excel_path=result_path)
    # qa_result_list = qa_result_list[:2]

    if opt.criteria_type == "all":
        result_dict = {}
        for criteria_type in accuracy_criteria_dict:
            print('{} Starting...'.format(criteria_type))
            result_list = []
            accuracy_criteria = accuracy_criteria_dict[criteria_type]
            evaluator_accuracy = load_evaluator(EvaluatorType.LABELED_SCORE_STRING, criteria=accuracy_criteria, llm=llm)

            for qa in tqdm(qa_result_list):
                qa_input = qa['image']
                ref_ans = qa['ref_caption']
                ref_ans = ref_ans.strip()  # 去除两端空白
                qa_ans = qa['caption']
                qa_ans = qa_ans.strip()  # 去除两端空白
                qa_num = 0
                if len(ref_ans) > 0 and len(qa_ans) > 0:
                    try:
                        eval_result = evaluator_accuracy.evaluate_strings(
                            prediction=qa_ans,
                            reference=ref_ans,
                            input=qa_input,
                        )
                        result_list.append({'image': qa_input, 'ref_ans': ref_ans, 'answer': qa_ans,
                                            'score': eval_result['score'], 'evaluate_reasoning': eval_result['reasoning']})
                    except BaseException as e:
                        print(e)
                        result_list.append({'image': qa_input, 'ref_ans': ref_ans, 'answer': qa_ans,
                                            'score': 0, 'evaluate_reasoning': str(e)})
                elif len(ref_ans) == 0 and len(qa_ans) == 0:
                    result_list.append({'image': qa_input, 'ref_ans': ref_ans, 'answer': qa_ans,
                                        'score': 10, 'evaluate_reasoning': '全为空值'})
                else:
                    result_list.append({'image': qa_input, 'ref_ans': ref_ans, 'answer': qa_ans,
                                        'score': 0, 'evaluate_reasoning': '存在空值'})

            result_dict[criteria_type] = result_list

        evaluate_file_name = ""
        if opt.model_online:
            evaluate_file_name = '%s_Qwen_VL_%s_qwen_max_%s.xlsx' % ('evaluate', opt.model_type, opt.criteria_type)
        else:
            evaluate_file_name = '%s_Qwen_VL_%s_qwen2.5_72B_%s.xlsx' % ('evaluate', opt.model_type, opt.criteria_type)
        evaluate_path = os.path.join(root_dir, output_type_name, evaluate_file_name)
        print_dict_result_to_excel(txt_filepath=evaluate_path, result=result_dict)

    else:
        prompt = ChatPromptTemplate.from_template(split_prompt)
        chain = prompt | llm
        categories = ["颜色", "结构", "构造", "矿物成分"]
        answer_list = []
        ref_ans_list = []
        image_list = []

        ref_ans_list_origin = []
        answer_list_origin = []

        for qa_item in tqdm(qa_result_list):
            answer = qa_item['caption']
            ref_ans = qa_item['ref_caption']
            image = qa_item['image']

            result_a = chain.invoke({"text": answer})
            result_a = split_by_category_regex(result_a)
            print('answer:', result_a)
            result_b = chain.invoke({"text": ref_ans})
            result_b = split_by_category_regex(result_b)
            print('ref_ans:', result_b)
            flag = True
            for key in categories:
                if key not in result_a:
                    print(image, result_a)
                    flag = False
                    break
                if key not in result_b:
                    print(image, result_b)
                    flag = False
                    break
            if flag:
                image_list.append(image)

                answer_list.append(result_a)
                answer_list_origin.append(answer)
                ref_ans_list.append(result_b)
                ref_ans_list_origin.append(ref_ans)

        answer_list_file = ""
        ref_answer_list_file = ""
        if opt.model_online:
            answer_list_file = '%s_Qwen_VL_%s_%s_qwen_max.xlsx' % ('answer_split', opt.model_type, opt.task_type)
            ref_answer_list_file = '%s_Qwen_VL_%s_%s_qwen_max.xlsx' % ('ref_ans_split', opt.model_type, opt.task_type)
        else:
            answer_list_file = '%s_Qwen_VL_%s_%s_qwen2.5_72B.xlsx' % ('answer_split', opt.model_type, opt.task_type)
            ref_answer_list_file = '%s_Qwen_VL_%s_%s_qwen2.5_72B.xlsx' % ('ref_ans_split', opt.model_type, opt.task_type)

        answer_list_path = os.path.join(root_dir, output_type_name, answer_list_file)
        print_dict_list_to_excel(answer_list_path, answer_list, answer_list_origin)

        ref_answer_list_path = os.path.join(root_dir, output_type_name, ref_answer_list_file)
        print_dict_list_to_excel(ref_answer_list_path, ref_ans_list, ref_ans_list_origin)

        result_dict = {}
        for criteria_type in accuracy_criteria_dict:
            print('{} Starting...'.format(criteria_type))
            result_list = []
            accuracy_criteria = accuracy_criteria_dict[criteria_type]
            evaluator_accuracy = load_evaluator(EvaluatorType.LABELED_SCORE_STRING, criteria=accuracy_criteria, llm=llm)
    
            for img_item, ans_item, ref_ans_item in tqdm(zip(image_list, answer_list, ref_ans_list)):
                ans_input = ans_item[criteria_type]
                ans_input = ans_input.strip()  # 去除两端空白
                ref_input = ref_ans_item[criteria_type]
                ref_input = ref_input.strip()  # 去除两端空白
                rock_sample_obj = Path(img_item).stem
                qa_input = f'对岩石样本{rock_sample_obj}的{criteria_type}方面进行描述'

                if len(ref_input) > 0 and len(ans_input) > 0:
                    try:
                        eval_result = evaluator_accuracy.evaluate_strings(
                            prediction=ans_input,
                            reference=ref_input,
                            input=qa_input,
                        )
                        result_list.append({'image': qa_input, 'ref_ans': ref_input, 'answer': ans_input,
                                            'score': eval_result['score'], 'evaluate_reasoning': eval_result['reasoning']})
                    except BaseException as e:
                        result_list.append({'image': qa_input, 'ref_ans': ref_input, 'answer': ans_input,
                                            'score': 0, 'evaluate_reasoning': str(e)})
                elif len(ref_input) == 0 and len(ans_input) == 0:
                    result_list.append({'image': qa_input, 'ref_ans': ref_input, 'answer': ans_input,
                                        'score': 10, 'evaluate_reasoning': '全为空值'})
                else:
                    result_list.append({'image': qa_input, 'ref_ans': ref_input, 'answer': ans_input,
                                        'score': 0, 'evaluate_reasoning': '存在空值'})
            result_dict[criteria_type] = result_list
    
        evaluate_file_name = ""
        if opt.model_online:
            evaluate_file_name = '%s_Qwen_VL_%s_%s_qwen_max_%s.xlsx' % ('evaluate', opt.model_type, opt.criteria_type, opt.task_type)
        else:
            evaluate_file_name = '%s_Qwen_VL_%s_%s_qwen2.5_72B_%s.xlsx' % ('evaluate', opt.model_type, opt.criteria_type, opt.task_type)
        evaluate_path = os.path.join(root_dir, output_type_name, evaluate_file_name)
        print_dict_result_to_excel(txt_filepath=evaluate_path, result=result_dict)


