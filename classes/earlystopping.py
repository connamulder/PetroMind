"""
    @Project: PetraMind
    @File   : earlystopping.py
    @Author : mulder
    @E-mail : c_mulder@163.com
    @Date   : 2025-04-07
    @Info   : 元宝（hunyuan）“LoRA早停”生成代码。
"""

import torch
import numpy as np


class EarlyStopping:
    def __init__(self, patience=3, delta=0, mode='min', restore_best_weights=True, save_path=''):
        """
        Args:
            patience (int): 容忍指标不提升的epoch数，默认3。
            delta (float): 指标提升的最小幅度，默认0（任何提升都算改进）。
            mode (str): 'min'表示监控指标越小越好（如val_loss），'max'表示越大越好（如val_acc）。
            restore_best_weights (bool): 是否恢复最佳权重，默认True。
        """
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.best_metric = np.inf if mode == 'min' else -np.inf
        self.counter = 0  # 记录连续不提升的epoch数
        self.best_weights = None  # 保存最佳模型权重
        self.best_model = None  # 保存最佳模型
        self.early_stop = False  # 是否触发早停标志
        self.save_path = save_path    # 模型保存路径

    def __call__(self, current_metric, model):
        """
        Args:
            current_metric (float): 当前epoch的验证集指标值（如val_loss或val_acc）。
            model (torch.nn.Module): 待监控的模型。
        """
        if self.mode == 'min':
            improved = (current_metric < self.best_metric - self.delta)
        else:  # mode == 'max'
            improved = (current_metric > self.best_metric + self.delta)

        if improved:
            self.best_metric = current_metric
            self.counter = 0
            model.save_pretrained(self.save_path)
            if self.restore_best_weights:
                self.best_weights = model.state_dict()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True  # 触发早停

    def load_best_weights(self, model):
        """加载最佳权重"""
        if self.restore_best_weights and self.best_weights is not None:
            model.load_state_dict(self.best_weights)