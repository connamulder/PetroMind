"""
    @Project: PetraMind
    @File   : 02_cnn_train.py
    @Author : mulder
    @E-mail : c_mulder@163.com
    @Date   : 2025-09-05
    @Info   : 基于 transformers、peft 等框架，使用 Qwen2.5-VL-7B-Instruct 模型在COCO2014图像描述 上进行LoRA微调训练，
              同时使用 SwanLab 监控训练过程与评估模型效果。
"""

import os
import argparse
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.models import inception_v3, vgg16, resnet50
from torchvision.models import densenet121, DenseNet121_Weights, efficientnet_b7, EfficientNet_B7_Weights
from modelscope import ViTImageProcessor, ViTForImageClassification
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from torchsummary import summary
from torch.utils.data.sampler import WeightedRandomSampler
from thop import profile

from dataset_rocks_list import ROCKS
from dataset_rocks_clip import ROCKS_CLIP
from utils import utils_result


# 是否为编码阶段的测试，编码阶段加载少量数据
is_coding_test = True

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=2, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="8 | 128")
parser.add_argument("--n_stop", type=int, default=20, help="number of epochs of training")
parser.add_argument("--do_train", action='store_true', help="is do training")
parser.add_argument("--imbalance_loader", action='store_true', help="use imbalance loader")
parser.add_argument("--lr", type=float, default=0.00005, help="adam: learning rate")
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument("--dataset_name", type=str, default='rocks13', help="dataset, mnist  cifar10 | rocks7 | rocks32")
parser.add_argument("--dataset_file", type=str, default='rocks13_random_tv_augu_224_train.txt', help="the file name of the dataset")
parser.add_argument("--val_file", type=str, default='rocks13_random_tv_augu_224_val.txt', help="the file name of the val")
parser.add_argument("--n_classes", type=int, default=13, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=224, help="size of each image dimension: 224 | 299")
parser.add_argument("--data_dist", type=str, default='norm', help="data distribution.norm: mean=std=[0.5, 0.5, 0.5]")
parser.add_argument("--model_type", type=str, default='ViT', help="pre-trained model: VGG16 | INCEPTIONV3 | resnet50 | desnet | effinet | ViT | CLIP")
parser.add_argument("--is_parall", action='store_true', help="is data paralled")
parser.add_argument("--is_pretrained", action='store_true', help="is model pretrained")
opt = parser.parse_args()
print(opt)

if is_coding_test:
    opt.do_train = True
    opt.is_pretrained = True

"""
opt.is_parall = True
opt.is_pretrained = True
"""

class_labels = ['olivinite', 'pyroxene or hornblendite', 'gabbro', 'diabase', 'anorthosite', 'diorite',
                'syenite', 'monzonite', 'syenogranite', 'monzonitic granite', 'granodiorite',
                'alkali feldspar granite', 'tonalite']

labels_name = {0: '橄榄岩', 1: '辉石-角闪石岩', 2: '辉长岩', 3: '辉绿岩', 4: '斜长岩', 5: '闪长岩',
               6: '正长岩', 7: '二长岩', 8: '正长花岗岩', 9: '二长花岗岩', 10: '花岗闪长岩',
               11: '碱长花岗岩', 12: '英云闪长岩'}


class RockModel(torch.nn.Module):
    def __init__(self, class_num=10, model_type='VGG16', pretrained=True):
        super(RockModel, self).__init__()
        self.class_num = class_num
        self.model_type = model_type
        self.pretrained = pretrained

        if self.model_type == "VGG16":
            if self.pretrained:
                self.model = vgg16(pretrained=True)
            else:
                self.model = vgg16(pretrained=False)
                self.model.classifier[6] = torch.nn.Linear(4096, self.class_num)
        elif self.model_type == "INCEPTIONV3":
            if self.pretrained:
                self.model = inception_v3(pretrained=True)
            else:
                self.model = inception_v3(pretrained=False)
                # self.model.fc = torch.nn.Linear(self.model.fc.in_features, self.class_num)
            self.model.aux_logits = False
        elif self.model_type == "resnet50":
            if self.pretrained:
                self.model = resnet50(pretrained=True)
            else:
                self.model = resnet50(pretrained=False)
                num_ftrs = self.model.fc.in_features
                self.model.fc = torch.nn.Linear(num_ftrs, self.class_num)
        elif self.model_type == "desnet":
            if self.pretrained:
                # self.model = densenet121(pretrained=True)
                self.model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
            else:
                self.model = densenet121(pretrained=False)
                num_ftrs = self.model.classifier.in_features
                self.model.classifier = torch.nn.Linear(num_ftrs, self.class_num)
        elif self.model_type == "effinet":
            if self.pretrained:
                self.model = efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1)
            else:
                self.model = efficientnet_b7(weights=None)
                num_ftrs = self.model.classifier[1].in_features
                self.model.classifier[1] = torch.nn.Linear(num_ftrs, self.class_num)
        elif self.model_type == "ViT":
            model_name = "google/vit-base-patch16-224"

            id2label = dict(zip(labels_name.keys(), labels_name.values()))
            print(id2label)
            label2id = dict(zip(labels_name.values(), labels_name.keys()))
            print(label2id)
            self.model = ViTForImageClassification.from_pretrained(model_name, num_labels=len(id2label),
                                                                   ignore_mismatched_sizes=True,
                                                                   id2label=id2label, label2id=label2id)
            print(self.model.classifier)
        elif self.model_type == "CLIP":
            model_name = "openai/clip-vit-base-patch16"
            self.model = CLIPModel.from_pretrained(model_name)
            # self.model.text_model.training = False
            # self.model.vision_model.training = False
            # 修改文本编码器输出层（适应14个类别）
            # self.model.text_projection = torch.nn.Linear(self.model.text_projection.in_features, self.class_num)
            # self.model.visual_projection = torch.nn.Linear(self.model.visual_projection.in_features, self.class_num)

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(1000, self.class_num),
            torch.nn.Sigmoid(),
        )

    def forward(self, img):
        if self.model_type == "CLIP":
            x = self.model(**img)
        else:
            x = self.model(img)
            if self.pretrained and self.model_type != "ViT":
                x = self.fc(x)
            if self.model_type == "ViT":
                x = x.logits

        return x


def main():
    model_type = opt.model_type
    os.makedirs(opt.dataset_name, exist_ok=True)
    res_dir = '%s/%s' % (opt.dataset_name, model_type)
    os.makedirs(res_dir, exist_ok=True)

    if opt.imbalance_loader:
        if opt.is_pretrained:
            res_dir = "./{}/res_{}_imbalance_loader_pretrained_epochs_{}_lr_{:f}".format(
                res_dir, opt.dataset_name, opt.n_epochs, opt.lr)
        else:
            res_dir = "./{}/res_{}_imbalance_loader_scratch_epochs_{}_lr_{:f}".format(
                res_dir, opt.dataset_name, opt.n_epochs, opt.lr)
    else:
        if opt.is_pretrained:
            res_dir = "./{}/res_{}_pretrained_epochs_{}_lr_{:f}".format(
                res_dir, opt.dataset_name, opt.n_epochs, opt.lr
            )
        else:
            res_dir = "./{}/res_scratch_{}_epochs_{}_lr_{:f}".format(
                res_dir, opt.dataset_name, opt.n_epochs, opt.lr
            )

    if opt.data_dist == 'norm':
        channel_mean = [0.5, 0.5, 0.5]
        channel_std = [0.5, 0.5, 0.5]

    # 这是反归一化的 mean 和std
    MEAN = [-mean / std for mean, std in zip(channel_mean, channel_std)]
    STD = [1 / std for std in channel_std]

    if not os.path.exists(res_dir):
        os.makedirs(res_dir, exist_ok=True)
    print("训练结果输出目录: {}".format(res_dir))

    cuda = True if torch.cuda.is_available() else False
    print("cuda: {}".format(cuda))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: {}".format(device))

    # Configure data loader
    data_path = 'data/%s' % opt.dataset_name
    os.makedirs(data_path, exist_ok=True)
    if opt.dataset_name == 'cifar10':
        class_to_prune = 2
        data_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                data_path,
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
                ),
            ),
            batch_size=opt.batch_size,
            shuffle=True,
        )
    elif 'rocks' in opt.dataset_name:
        # tensor([0.5617, 0.5398, 0.5221]) tensor([0.1609, 0.1573, 0.1584])
        train_tf = transforms.Compose([transforms.Resize(opt.img_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize(channel_mean, channel_std)])
        # 预读取图像并存储到pkl文件中，可启动GPU训练
        dataset_file = os.path.join(r"F:\11_CV_Datasets\rockclass", opt.dataset_file)
        val_file = os.path.join(r"F:\11_CV_Datasets\rockclass", opt.val_file)
        if opt.model_type == "CLIP":
            train_data = ROCKS_CLIP(dataset_file, opt.img_size)
            val_data = ROCKS_CLIP(val_file, opt.img_size)
        else:
            train_data = ROCKS(dataset_file, opt.img_size, transform=train_tf)
            val_data = ROCKS(val_file, opt.img_size, transform=train_tf)

        if opt.imbalance_loader:
            labels = train_data.get_labels()
            labels = [int(x) for x in labels]
            labels = torch.tensor(labels)
            # 计算每个类别的样本数量
            label_counts = torch.bincount(labels)
            # weight = [ ] 里面每一项代表该样本种类占总样本的倒数。
            class_weights = 1.0 / label_counts.float()
            sample_weights = class_weights[labels]

            # 创建采样器
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
            data_loader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size, sampler=sampler)
        else:
            data_loader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size, pin_memory=True,
                                                      shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=opt.batch_size, pin_memory=True, shuffle=True)

    data = data_loader.__iter__().__next__()[0]
    if opt.model_type != "CLIP":
        img_shape = (data.shape[1], opt.img_size, opt.img_size)
        print("image shape: {}".format(img_shape))

    # 加载预训练的Inception-v3模型
    model = RockModel(class_num=opt.n_classes, model_type=model_type, pretrained=opt.is_pretrained)
    processor = None
    if opt.model_type == "CLIP":
        model_name = "openai/clip-vit-base-patch16"
        processor = CLIPProcessor.from_pretrained(model_name)
        print(model.model.config)

    input = torch.randn(opt.batch_size, 3, opt.img_size, opt.img_size)  # 假设输入是 1 张 3x224x224 的图片
    flops, params = profile(model, inputs=(input,))

    total_params, trainable_params = utils_result.count_model_parameters(model)
    params_file_path = '%s/params.txt' % res_dir
    with open(params_file_path, 'w') as params_file:
        params_file.write("thop模型参数统计:\n")
        params_file.write("FLOPs: %.6fG\n" % (flops / 1e9))
        params_file.write("Params: %.6fM\n" % (params / 1e6))
        params_file.write("模型参数统计:\n")
        params_file.write("总参数量: %.6fM\n" % (total_params / 1e6))
        params_file.write("可训练参数量: %.6fM\n" % (trainable_params / 1e6))

    if (opt.model_type == "ViT") or (opt.model_type == "CLIP") or (opt.model_type == "desnet"):
        print("不输出summary")
    else:
        summary(model, input_size=(3, opt.img_size, opt.img_size), device='cpu')

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = None
    if opt.model_type == "CLIP":
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98),
                                     eps=1e-6, weight_decay=0.2)
    else:
        optimizer = torch.optim.SGD(model.parameters(), opt.lr,
                                    momentum=opt.momentum,
                                    weight_decay=opt.weight_decay)

    if cuda:
        if opt.is_parall:
            model = torch.nn.DataParallel(model.cuda())
        else:
            model = model.cuda()
        criterion = criterion.cuda()

    if opt.do_train:
        model.train()

        losses = utils_result.AverageMeter()
        top1 = utils_result.AverageMeter()
        losses_val = utils_result.AverageMeter()
        top1_val = utils_result.AverageMeter()
        loss_history = []
        pred_history = []
        loss_val_history = []
        pred_val_history = []
        best_loss = 10.0
        train_step = 0
        start_time = time.time()

        for epoch in range(opt.n_epochs):
            for iter, (x_, y_) in enumerate(data_loader):
                if is_coding_test:
                    if iter > 3:
                        break

                if iter == data_loader.dataset.__len__() // opt.batch_size:
                    break
                if opt.model_type == "CLIP":
                    images = [Image.open(p) for p in x_]
                    # text_descriptions = [f"a photo of a {class_labels[cls]}" for cls in y_]
                    text_descriptions = [f"a photo of a {label}" for label in class_labels]
                    inputs = processor(text=text_descriptions, images=images, return_tensors="pt", padding=True)
                    if cuda:
                        y_ = y_.cuda()
                        inputs['input_ids'] = inputs['input_ids'].cuda()
                        inputs['attention_mask'] = inputs['attention_mask'].cuda()
                        inputs['pixel_values'] = inputs['pixel_values'].cuda()
                    outputs = model(inputs)
                    y_pred = outputs.logits_per_image
                    # loss = F.cross_entropy(logits_per_image, y_)
                    loss = criterion(y_pred, y_)
                else:
                    if cuda:
                        x_, y_ = x_.cuda(), y_.cuda()
                    # update D network
                    optimizer.zero_grad()
                    y_pred = model(x_)
                    loss = criterion(y_pred, y_)

                losses.update(loss.item())
                prec1 = accuracy(y_pred.data, y_)[0]
                top1.update(prec1.item())

                loss.backward()
                optimizer.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [loss: %f] [pred: %f]"
                    % (epoch, opt.n_epochs, iter, len(data_loader), loss.item(), prec1.item())
                )

            loss_history.append(losses.avg)
            pred_history.append(top1.avg)

            if losses.avg < best_loss:
                best_loss = losses.avg
                weight_file = '%s/%s_%s_model.pth' % (res_dir, opt.dataset_name, model_type)
                torch.save(model.state_dict(), weight_file)
                train_step = 0
            else:
                train_step += 1
            if train_step > opt.n_stop:
                break

            losses.reset()
            top1.reset()

            # val
            model.eval()
            for iter, (x_, y_) in enumerate(val_loader):
                if iter == val_loader.dataset.__len__() // opt.batch_size:
                    break

                if is_coding_test:
                    if iter > 3:
                        break

                if opt.model_type == "CLIP":
                    images = [Image.open(p) for p in x_]
                    text_descriptions = [f"a photo of a {label}" for label in class_labels]
                    inputs = processor(text=text_descriptions, images=images, return_tensors="pt", padding=True)
                    if cuda:
                        y_ = y_.cuda()
                        inputs['input_ids'] = inputs['input_ids'].cuda()
                        inputs['attention_mask'] = inputs['attention_mask'].cuda()
                        inputs['pixel_values'] = inputs['pixel_values'].cuda()
                    outputs = model(inputs)
                    y_pred = outputs.logits_per_image
                    v_loss = criterion(y_pred, y_)
                else:
                    if cuda:
                        x_, y_ = x_.cuda(), y_.cuda()
                    with torch.no_grad():
                        y_pred = model(x_)
                        v_loss = criterion(y_pred, y_)
                prec_val = accuracy(y_pred.data, y_)[0]
                losses_val.update(v_loss.item())
                top1_val.update(prec_val.item())

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [val loss: %f] [val pred: %f]"
                    % (epoch, opt.n_epochs, iter, len(val_loader), v_loss.item(), prec_val.item())
                )
            loss_val_history.append(losses_val.avg)
            pred_val_history.append(top1_val.avg)

            losses_val.reset()
            top1_val.reset()

        end_time = time.time()
        # --------------------------- Saving train time --------------------------- #
        train_time = end_time - start_time
        runtime_file_path = '%s/runtime_train.txt' % res_dir
        with open(runtime_file_path, 'w') as runtime_file:
            runtime_file.write("start time: %s\n" % time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
            runtime_file.write("end time: %s\n" % time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))

            hours, remainder = divmod(train_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            time_format = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

            runtime_file.write("train time: %s\n" % time_format)

        # --------------------------- Display performance --------------------------- #
        # plot loss
        plt.plot(loss_history, label='loss')
        plt.plot(pred_history, label='pred')
        plt.plot(loss_val_history, label='val_loss')
        plt.plot(pred_val_history, label='val_pred')
        plt.legend()
        plt.savefig('%s/%s_%s_train_plot.png' % (res_dir, opt.dataset_name, model_type))
        plt.savefig('%s/%s_%s_train_plot.svg' % (res_dir, opt.dataset_name, model_type))
        plt.clf()

        # --------------------------- Save performance ------------------------------ #
        loss_train_fname = "{}/{}_loss_train.csv".format(res_dir, opt.dataset_name)
        with open(loss_train_fname, 'w') as train_loss_csvfile:
            for item in loss_history:
                train_loss_csvfile.write("%s\n" % item)
        pred_train_fname = "{}/{}_pred_train.csv".format(res_dir, opt.dataset_name)
        with open(pred_train_fname, 'w') as pred_train_csvfile:
            for item in pred_history:
                pred_train_csvfile.write("%s\n" % item)
        loss_val_fname = "{}/{}_loss_val.csv".format(res_dir, opt.dataset_name)
        with open(loss_val_fname, 'w') as train_val_csvfile:
            for item in loss_val_history:
                train_val_csvfile.write("%s\n" % item)
        pred_val_fname = "{}/{}_pred_val.csv".format(res_dir, opt.dataset_name)
        with open(pred_val_fname, 'w') as pred_val_csvfile:
            for item in pred_val_history:
                pred_val_csvfile.write("%s\n" % item)
    else:
        from sklearn.metrics import confusion_matrix, classification_report
        import pandas as pd

        weight_file = '%s/%s_%s_model.pth' % (res_dir, opt.dataset_name, model_type)
        if os.path.exists(weight_file):
            if opt.is_parall:
                model.load_state_dict(torch.load(weight_file))
            else:
                checkpoint = torch.load(weight_file)
                model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()})

            model.eval()

            # 随机选择一些样本
            # indices = np.random.choice(len(val_loader), size=16, replace=False)
            # images = [val_loader[i][0] for i in indices]
            # labels = [val_loader[i][1] for i in indices]
            y_preds = []
            y_trues = []

            for iter, (x_, y_) in enumerate(val_loader):
                if is_coding_test:
                    if iter > 500:
                        break

                """
                if iter == val_loader.dataset.__len__() // opt.batch_size:
                    break
                """
                if cuda:
                    x_, y_ = x_.cuda(), y_.cuda()

                with torch.no_grad():
                    y_pred = model(x_)
                    y_pred = torch.argmax(y_pred, dim=1)
                if iter == 0:
                    y_preds = y_pred
                    y_trues = y_
                else:
                    y_preds = torch.cat((y_preds, y_pred), dim=0)
                    y_trues = torch.cat((y_trues, y_), dim=0)

                print(
                    "[Batch %d/%d]" % (iter, len(val_loader))
                )

            label_name = ['010_olivinite', '021_pyroxene_hornblendite', '022_gabbro', '023_diabase', '024_anorthosite',
                          '031_diorite', '041_syenite', '044_monzonite', '061_Syenogranite', '062_monzonitic_granite',
                          '063_granodiorite', '064_alkali_feldspar_granite', '065_tonalite']
            # 计算每个类别的样本数量
            label_counts = torch.bincount(y_trues)
            if len(label_counts) == opt.n_classes:
                #输出报告
                report = classification_report(y_trues.cpu(), y_preds.cpu(), output_dict=True)
                report_file = "%s/classification_report.csv" % (res_dir)
                df = pd.DataFrame(report).transpose()
                df.to_csv(report_file, index=True)

                # 计算混淆矩阵
                cm = confusion_matrix(y_trues.cpu().numpy(), y_preds.cpu().numpy())
                draw_confusion_matrix(cm, opt.n_classes, res_dir)
            else:
                print("label_counts < opt.n_classes")


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def draw_confusion_matrix(cm, label_class_num, res_dir):
    normalize = True
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # 绘制混淆矩阵
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.yticks(range(opt.n_classes))
    plt.xticks(range(opt.n_classes))
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

    plt.savefig('%s/%s_%s_confusion_matrix.png' % (res_dir, opt.dataset_name, opt.model_type), dpi=300)
    plt.savefig('%s/%s_%s_confusion_matrix.svg' % (res_dir, opt.dataset_name, opt.model_type), dpi=300)
    # plt.show()


if __name__ == '__main__':
    main()
