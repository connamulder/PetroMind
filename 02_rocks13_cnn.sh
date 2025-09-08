#!/bin/bash
#### rocks13 train and val

# step 1.1 VGG16 training from scratch
# python ./02_cnn_train.py --do_train --n_epochs 200 --batch_size 128 --n_stop 10 --model_type VGG16 --img_size 224 --is_parall
# python ./02_cnn_train.py --n_epochs 200 --batch_size 128 --n_stop 10 --model_type VGG16 --img_size 224 --is_parall
# step 1.2 VGG16 pre-trained training
# python ./02_cnn_train.py --do_train --n_epochs 200 --batch_size 128 --n_stop 10 --model_type VGG16 --img_size 224 --is_pretrained --is_parall
# python ./02_cnn_train.py --n_epochs 200 --batch_size 128 --n_stop 10 --model_type VGG16 --img_size 224 --is_pretrained --is_parall
# step 1.3 VGG16 pre-trained training with imbalance_loader
#python ./02_cnn_train.py --do_train --n_epochs 200 --batch_size 128 --n_stop 10 --model_type VGG16 --img_size 224 --is_pretrained --is_parall --imbalance_loader
#python ./02_cnn_train.py --n_epochs 200 --batch_size 128 --n_stop 10 --model_type VGG16 --img_size 224 --is_pretrained --is_parall --imbalance_loader

# step 2.1 Inception-v3 training from scratch
#python ./02_cnn_train.py --do_train --n_epochs 200 --batch_size 128 --n_stop 10 --model_type INCEPTIONV3 --img_size 299 --is_parall
#python ./02_cnn_train.py --n_epochs 200 --batch_size 128 --n_stop 10 --model_type INCEPTIONV3 --img_size 299 --is_parall
# step 2.2 Inception-v3 pre-trained training
#python ./02_cnn_train.py --do_train --n_epochs 200 --batch_size 128 --n_stop 10 --model_type INCEPTIONV3 --img_size 299 --is_pretrained --is_parall
#python ./02_cnn_train.py --n_epochs 200 --batch_size 128 --n_stop 10 --model_type INCEPTIONV3 --img_size 299 --is_pretrained --is_parall

# step 3.1 ResNet50 training from scratch
#python ./02_cnn_train.py --do_train --n_epochs 200 --batch_size 128 --n_stop 10 --model_type resnet50 --img_size 224 --is_parall
#python ./02_cnn_train.py --n_epochs 200 --batch_size 128 --n_stop 10 --model_type resnet50 --img_size 224 --is_parall
# step 3.2 ResNet50 pre-trained training
#python ./02_cnn_train.py --do_train --n_epochs 200 --batch_size 128 --n_stop 10 --model_type resnet50 --img_size 224 --is_pretrained --is_parall
#python ./02_cnn_train.py --n_epochs 200 --batch_size 128 --n_stop 10 --model_type resnet50 --img_size 224 --is_pretrained --is_parall
# step 3.3 ResNet50 pre-trained training with imbalance_loader
#python ./02_cnn_train.py --do_train --n_epochs 200 --batch_size 128 --n_stop 10 --model_type resnet50 --img_size 224 --is_pretrained --is_parall --imbalance_loader
#python ./02_cnn_train.py --n_epochs 200 --batch_size 128 --n_stop 10 --model_type resnet50 --img_size 224 --is_pretrained --is_parall --imbalance_loader
# step 3.4 ResNet50 training from scratch with imbalance_loader
#python ./02_cnn_train.py --do_train --n_epochs 200 --batch_size 128 --n_stop 10 --model_type resnet50 --img_size 224 --is_parall --imbalance_loader
#python ./02_cnn_train.py --n_epochs 200 --batch_size 128 --n_stop 10 --model_type resnet50 --img_size 224 --is_parall --imbalance_loader

# step 4.1 DesNet training from scratch
#python ./02_cnn_train.py --do_train --n_epochs 200 --batch_size 128 --n_stop 10 --model_type desnet --img_size 224 --is_parall
#python ./02_cnn_train.py --n_epochs 200 --batch_size 128 --n_stop 10 --model_type desnet --img_size 224 --is_parall
# step 4.2 DesNet pre-trained training
#python ./02_cnn_train.py --do_train --n_epochs 200 --batch_size 128 --n_stop 10 --model_type desnet --img_size 224 --is_pretrained --is_parall
#python ./02_cnn_train.py --n_epochs 200 --batch_size 128 --n_stop 10 --model_type desnet --img_size 224 --is_pretrained --is_parall
# step 4.3 DesNet pre-trained training with imbalance_loader
#python ./02_cnn_train.py --do_train --n_epochs 200 --batch_size 128 --n_stop 10 --model_type desnet --img_size 224 --is_pretrained --is_parall --imbalance_loader
#python ./02_cnn_train.py --n_epochs 200 --batch_size 128 --n_stop 10 --model_type desnet --img_size 224 --is_pretrained --is_parall --imbalance_loader
# step 4.4 DesNet training from scratch with imbalance_loader
#python ./02_cnn_train.py --do_train --n_epochs 200 --batch_size 128 --n_stop 10 --model_type desnet --img_size 224 --is_parall --imbalance_loader
#python ./02_cnn_train.py --n_epochs 200 --batch_size 128 --n_stop 10 --model_type desnet --img_size 224 --is_parall --imbalance_loader

# step 5.1 EfficientNet-B7 training from scratch
python ./02_cnn_train.py --do_train --n_epochs 200 --batch_size 128 --n_stop 10 --model_type effinet --img_size 224 --is_parall
python ./02_cnn_train.py --n_epochs 200 --batch_size 128 --n_stop 10 --model_type effinet --img_size 224 --is_parall
# step 5.2 EfficientNet-B7 pre-trained training
python ./02_cnn_train.py --do_train --n_epochs 200 --batch_size 128 --n_stop 10 --model_type effinet --img_size 224 --is_pretrained --is_parall
python ./02_cnn_train.py --n_epochs 200 --batch_size 128 --n_stop 10 --model_type effinet --img_size 224 --is_pretrained --is_parall

# step 6.1 ViT pre-trained training
#python ./02_cnn_train.py --do_train --n_epochs 200 --batch_size 128 --n_stop 10 --model_type ViT --img_size 224 --is_pretrained --is_parall
#python ./02_cnn_train.py --n_epochs 200 --batch_size 128 --n_stop 10 --model_type ViT --img_size 224 --is_pretrained --is_parall
# step 6.2 pre-trained training with imbalance_loader
#python ./02_cnn_train.py --do_train --n_epochs 200 --batch_size 128 --n_stop 10 --model_type ViT --img_size 224 --is_pretrained --is_parall --imbalance_loader
#python ./02_cnn_train.py --n_epochs 200 --batch_size 128 --n_stop 10 --model_type ViT --img_size 224 --is_pretrained --is_parall --imbalance_loader

python ./smsDemo.py --message "EfficientNet-B7训练结束！"