"""
    @Project: PetraMind
    @File   : dataset_rocks_list.py
    @Author : mulder
    @E-mail : c_mulder@163.com
    @Date   : 2024-05-24
    @info   : 读取岩石图像数据.
"""

import torch
from torchvision import transforms
from PIL import Image
import torch.utils.data as data
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
import os


class ROCKS(data.Dataset):
    def __init__(self, txt_path, image_size=224, transform=None, convert=True, is_path=False):
        self.image_size = image_size
        self.transform = transform
        self.imgs_info = self.get_images(txt_path)
        self.convert = convert
        self.is_path = is_path

    def get_images(self, txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            imgs_info = f.readlines()
            imgs_info = list(map(lambda x: x.strip().split(' '), imgs_info))
        return imgs_info

    def __getitem__(self, index):
        img_file_fullpath, label = self.imgs_info[index]
        img_fg = None
        if self.is_path:
            img_fg = img_file_fullpath
        else:
            img = Image.open(img_file_fullpath)
            if self.convert:
                img = img.convert("RGB")
                img_fg = img.resize([self.image_size, self.image_size])
                if self.transform is not None:
                    img_fg = self.transform(img_fg)
            else:
                img_fg = img

        label = int(label)

        return img_fg, label

    def __len__(self):
        return len(self.imgs_info)

    def get_labels(self):
        return [x[1] for x in self.imgs_info]


# 计算输入图像的 mean 和std
def cal_std_mean(images):
    # 创建3维的空列表
    channel_mean = torch.zeros(3)
    channel_std = torch.zeros(3)
    print(images.shape)

    N, C, H, W = images.shape[:4]
    # 将w,h维度的数据展平，为batch，channel,data,然后对三个维度上的数分别求和和标准差
    images = images.view(N, C, -1)
    print(images.shape)
    # 展平后，w,h属于第二维度，对他们求平均，sum(0)为将同一纬度的数据累加
    channel_mean += images.mean(2).sum(0)
    # 展平后，w,h属于第二维度，对他们求标准差，sum(0)为将同一纬度的数据累加
    channel_std += images.std(2).sum(0)

    return channel_mean, channel_std, N


if __name__ == "__main__":
    input_size = 224
    imbalance_loader = True
    txt_path = r"F:\11_CV_Datasets\rockclass\rocks14_tv_augu_224_train.txt"

    # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    transform = transforms.Compose([transforms.Resize((input_size, input_size)),
                                    transforms.ToTensor()])
    my_rocks = ROCKS(txt_path, input_size, transform=transform)

    labels = my_rocks.get_labels()
    labels = [int(x) for x in labels]
    labels = torch.tensor(labels)
    # 计算每个类别的样本数量
    label_counts = torch.bincount(labels)
    # weight = [ ] 里面每一项代表该样本种类占总样本的倒数。
    class_weights = 1.0 / label_counts.float()
    sample_weights = class_weights[labels]

    # 创建采样器
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    dataloader = data.DataLoader(my_rocks, batch_size=128, sampler=sampler)

    """    
    from torchsampler.imbalanced import ImbalancedDatasetSampler

    if imbalance_loader:
        dataloader = data.DataLoader(my_rocks, sampler=ImbalancedDatasetSampler(my_rocks), batch_size=128)
    else:
        dataloader = data.DataLoader(my_rocks, batch_size=128, shuffle=True)
    """
    print(my_rocks.__len__())

    mean = torch.zeros(3)
    std = torch.zeros(3)
    nb_samples = 0
    for i, (input_data, target) in enumerate(dataloader):
        temp_mean = torch.zeros(3)
        temp_std = torch.zeros(3)
        print('input_data%d=' % i, input_data)
        print('input_data_shape%d/%d=' % (i, len(dataloader)), input_data.shape)
        print('target%d=' % i, target)
        temp_mean, temp_std, temp_num = cal_std_mean(input_data)
        mean += temp_mean
        std += temp_std
        nb_samples += temp_num
    # 获取同一batch的均值和标准差
    mean /= nb_samples
    std /= nb_samples

    # rocks7
    # 224: tensor([0.5617, 0.5398, 0.5221]) tensor([0.1609, 0.1573, 0.1584])
    # 299: tensor([0.5617, 0.5398, 0.5221]) tensor([0.1593, 0.1556, 0.1567])

    # rocks32
    # 224: tensor([0.5521, 0.5317, 0.5116]) tensor([0.1469, 0.1462, 0.1475])
    # 299: tensor([0.5521, 0.5317, 0.5116]) tensor([0.1452, 0.1445, 0.1458])
    print(mean, std)