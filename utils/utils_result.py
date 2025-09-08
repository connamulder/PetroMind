"""
    @Project: PetraMind
    @File   : utils_result.py
    @Author : mulder
    @E-mail : c_mulder@163.com
    @Date   : 2025-09-08
    @info   : 通用工具
"""

import os
import re
import pandas as pd


def count_model_parameters(model):
    """
    统计模型的总参数量和可训练参数量
    Args:
        model: PyTorch 模型实例

    Returns:
        total_params: 总参数量
        trainable_params: 可训练参数量
    """
    # 计算总参数量
    total_params = sum(p.numel() for p in model.parameters())

    # 计算可训练参数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 格式化输出（单位：百万）
    total_m = total_params / 1e6
    trainable_m = trainable_params / 1e6

    print(f"模型参数统计:")
    print(f"总参数量: {total_m:.2f}M")
    print(f"可训练参数量: {trainable_m:.2f}M")

    return total_params, trainable_params


class AverageMeter(object):
    """ Computes ans stores the average and current value"""
    def __init__(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def find_max_number_folder(root_dir):
    """
    在指定目录中查找包含最大数字的文件夹

    参数：
    root_dir (str): 要搜索的根目录路径

    返回：
    tuple: (包含最大数字的文件夹名称, 最大数字) 或 (None, -1)
    """
    max_num = -1
    target_folder = None

    try:
        for item in os.listdir(root_dir):
            item_path = os.path.join(root_dir, item)

            if os.path.isdir(item_path):
                # 提取文件夹名称中的所有数字序列
                numbers = re.findall(r'\d+', item)

                if numbers:
                    # 转换为整数并找到最大值
                    current_max = max(map(int, numbers))

                    if current_max > max_num:
                        max_num = current_max
                        target_folder = item
    except PermissionError:
        print("警告：部分文件夹因权限问题无法访问")
    except Exception as e:
        print(f"发生未知错误：{str(e)}")

    return target_folder, max_num


def contains_rock_description(text):
    # 定义要查找的关键词
    keyword = "Rock lithological description"

    # 检查关键词是否存在于文本中
    if keyword in text:
        return True
    else:
        return False


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


def remove_last_sentence_if_no_rock_name(text):
    """
    删除文本的最后一句，但需要判断最后一句是否含有"岩石定名"
    如果最后一句包含"岩石定名"，则删除；否则保留不删除最后一句

    参数:
        text: 输入的文本字符串

    返回:
        处理后的文本字符串
    """
    # 首先按句号分句（简单处理，实际可能需要更复杂的分句逻辑）
    sentences = [s.strip() for s in text.split('。') if s.strip()]

    if not sentences:
        return text

    # 检查最后一句是否包含"岩石定名"
    last_sentence = sentences[-1]
    if "岩石定名" in last_sentence:
        # 不包含则删除最后一句
        if len(sentences) == 1:
            return ""
        result = '。'.join(sentences[:-1]) + '。'
        return result
    else:
        # 包含则保留所有句子
        return text


# 使用示例
if __name__ == "__main__":
    test_text1 = "这是一个测试句子。这是另一个句子。岩石定名为花岗闪长岩。"
    print("测试1（应删除最后一句）:")
    print(remove_last_sentence_if_no_rock_name(test_text1))

    """
    # 示例用法
    sample_text = "This is a sample text containing Rock lithological description of the area."
    result = contains_rock_description(sample_text)
    print(f"文本中包含'Rock lithological description'吗？ {result}")
    """

    """
    # target_dir = input("请输入要搜索的目录路径：").strip()
    target_dir = r"F:\Pytorch_learning\VQA\output\Qwen-VL-3B"
    if os.path.isdir(target_dir):
        folder, number = find_max_number_folder(target_dir)
        if folder:
            print(f"找到最大数字文件夹：{folder}（包含数字：{number}）")
        else:
            print("未找到包含数字的文件夹")
    else:
        print("输入的路径无效或不存在")
    """