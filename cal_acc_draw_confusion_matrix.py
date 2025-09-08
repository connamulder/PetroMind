"""
    @Project: PetraMind
    @File   : cal_acc_draw_confusion_matrix.py
    @Author : mulder
    @E-mail : c_mulder@163.com
    @Date   : 2025-09-08
    @info   : 混淆矩阵计算与输出
"""

import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import pandas as pd
import os


# 13: '斜长花岗岩',
labels_name = {0: '橄榄岩', 1: '辉石-角闪石岩', 2: '辉长岩', 3: '辉绿岩', 4: '斜长岩', 5: '闪长岩',
              6: '正长岩', 7: '二长岩', 8: '正长花岗岩', 9: '二长花岗岩', 10: '花岗闪长岩',
              11: '碱长花岗岩', 12: '英云闪长岩',  13: '其它'}


def draw_confusion_matrix(cm, label_class_num, res_dir, step=0):
    normalize = True
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # 绘制混淆矩阵
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.yticks(range(label_class_num))
    plt.xticks(range(label_class_num))
    # plt.yticks(range(opt.n_classes), label_name)
    # plt.xticks(range(opt.n_classes), label_name, rotation=45)

    # plt.tight_layout()
    plt.colorbar()

    for i in range(label_class_num):
        for j in range(label_class_num):
            color = (1, 1, 1) if i == j else (0, 0, 0)  # 对角线字体白色，其他黑色
            value = float(format('%.2f' % cm[j, i]))
            # value = float(format('%.1f' % (cm[j, i] / label_counts[j])))
            plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color)

    if matplotlib.__version__ < '3.8':
        figmanager = plt.get_current_fig_manager()
        figmanager.window.state('zoomed')  # 窗口最大化
    else:
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()  # 窗口最大化

    figure = plt.gcf()
    figure.set_size_inches(10, 8)

    plt.savefig('%s/confusion_matrix_%d.png' % (res_dir, step), dpi=300)
    plt.savefig('%s/confusion_matrix_%d.svg' % (res_dir, step), dpi=300)
    # plt.show()


def cal_acc_draw_cf(preds_pkl_dir, trues_pkl_dir, step=0, res_dir='output'):
    y_pred_labels = []
    y_true_labels = []
    y_preds = []
    y_trues = []

    id2label = dict(zip(labels_name.keys(), labels_name.values()))
    print(id2label)
    label2id = dict(zip(labels_name.values(), labels_name.keys()))
    print(label2id)

    # preds_pkl_dir = os.path.join(res_dir, preds_pkl_dir)
    # trues_pkl_dir = os.path.join(res_dir, trues_pkl_dir)

    # 从文件中读取List对象
    with open(preds_pkl_dir, 'rb') as file:
        y_pred_labels = pickle.load(file)
    with open(trues_pkl_dir, 'rb') as file:
        y_true_labels = pickle.load(file)

    unique_labels_trues = np.unique(y_true_labels)  # 返回唯一值数组
    unique_labels_preds = np.unique(y_pred_labels)
    print("unique_labels_preds: %s" % unique_labels_preds)

    for iter in range(len(y_pred_labels)):
        label = y_pred_labels[iter]
        if label in label2id.keys():
            id = label2id[label]
        else:
            id = len(labels_name) - 1
        y_preds.append(id)

    for iter in range(len(y_true_labels)):
        label = y_true_labels[iter]
        id = label2id[label]
        y_trues.append(id)

    unique_classes_trues = np.unique(y_trues)  # 返回唯一值数组
    unique_classes_preds = np.unique(y_preds)
    label_name = []
    if len(unique_classes_trues) == 14:
        label_name = ['010_olivinite', '021_pyroxene_hornblendite', '022_gabbro', '023_diabase', '024_anorthosite',
                      '031_diorite', '041_syenite', '044_monzonite', '061_Syenogranite', '062_monzonitic_granite',
                      '063_granodiorite', '064_alkali_feldspar_granite', '064_tonalite', '065_plagioclase_granite', 'Others']
    elif len(unique_classes_trues) == 13:
        label_name = ['010_olivinite', '021_pyroxene_hornblendite', '022_gabbro', '023_diabase', '024_anorthosite',
                      '031_diorite', '041_syenite', '044_monzonite', '061_Syenogranite', '062_monzonitic_granite',
                      '063_granodiorite', '064_alkali_feldspar_granite', '064_tonalite', 'Others']

    assert len(y_preds) == len(y_trues)

    y_preds = torch.Tensor(y_preds)
    y_preds = y_preds.type(torch.int32)
    y_trues = torch.Tensor(y_trues)
    y_trues = y_trues.type(torch.int32)


    # 计算每个类别的样本数量
    label_counts = torch.bincount(y_trues)

    # 计算正确预测的数量
    correct = (y_preds == y_trues).sum().item()
    # 计算准确率
    accuracy = correct / y_trues.size(0)
    print("准确率为：%.3f" % (accuracy*100))

    f1_value = f1_score(y_trues, y_preds, average='macro')
    print("F1值宏平均为：%.3f" % (f1_value * 100))

    str_report = classification_report(y_trues, y_preds)
    print(str_report)

    report = classification_report(y_trues, y_preds, output_dict=True)
    report_file = "%s/classification_report.csv" % (res_dir)
    df = pd.DataFrame(report).transpose()
    df.to_csv(report_file, index=True)

    if len(label_counts) == len(label_name):
        # 计算混淆矩阵
        cm = confusion_matrix(y_trues.cpu().numpy(), y_preds.cpu().numpy())
        draw_confusion_matrix(cm, len(label_name), res_dir, step)
    else:
        print("label_counts < len(label_name)")

    return accuracy, f1_value


# 使用示例
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='32B', help="3B | 7B | 32B-耗尽 | 72B")
    parser.add_argument("--task_type", type=str, default='class', help=" class | cap | multi | merge")
    opt = parser.parse_args()
    print(opt)

    output_dir = './output_hfut/32B-32'
    step = 7310
    preds_pkl_name = 'val_preds_%s_%s_rock13_class.pkl' % (opt.model_type, opt.task_type)
    preds_pkl_dir = os.path.join(output_dir, preds_pkl_name)

    trues_pkl_name = 'val_trues_%s_%s_rock13_class.pkl' % (opt.model_type, opt.task_type)
    trues_pkl_dir = os.path.join(output_dir, trues_pkl_name)

    acc = cal_acc_draw_cf(preds_pkl_dir, trues_pkl_dir, res_dir=output_dir)