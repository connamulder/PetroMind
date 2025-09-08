"""
    @Project: PetraMind
    @File   : dataset_rocks_list.py
    @Author : mulder
    @E-mail : c_mulder@163.com
    @Date   : 2024-05-24
    @info   : 按CLIP格式要求读取岩石图像数据.
"""

import torch.utils.data as data


class_labels = ['olivinite', 'pyroxene or hornblendite', 'gabbro', 'diabase', 'anorthosite', 'diorite',
                'syenite', 'monzonite', 'syenogranite', 'monzonitic granite', 'granodiorite',
                'alkali feldspar granite', 'tonalite', 'plagioclase granite']


class ROCKS_CLIP(data.Dataset):
    def __init__(self, txt_path, processor, image_size=224):
        self.image_size = image_size
        self.imgs_info = self.get_images(txt_path)

    def get_images(self, txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            imgs_info = f.readlines()
            imgs_info = list(map(lambda x: x.strip().split(' '), imgs_info))
        return imgs_info

    def __getitem__(self, index):
        img_file_fullpath, label = self.imgs_info[index]
        label = int(label)

        return img_file_fullpath, label

    def __len__(self):
        return len(self.imgs_info)

    def get_labels(self):
        return [x[1] for x in self.imgs_info]