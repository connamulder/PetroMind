# PetroMind: A multimodal petrographic model for rock image classification and lithological description generation

<div align="center">
<strong>Author: Zhongliang Chen, Chaojie Zheng, Mingming Zhang, Zhaoqi Hu, Jianchao Duan</strong>
  
<strong>Geological Survey of Anhui Province (Anhui Institute of Geological Sciences)</strong>
<strong>School of Resources and Environment Engineering, Hefei University of Technology</strong>
</div>

This is the official repository for paper **"PetroMind: A multimodal petrographic model for rock image classification and lithological description generation"**. [[paper](https://)]

## Please share a <font color='orange'>STAR ⭐</font> if this project does help

## Preparation
Create a virtual environment and install the required libraries.
```shell
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

## Running the code

### 1. PetroMind Instruction Fine-tuning
Execute the PetroMind instruction fine-tuning script.
```shell
01_train_two_stage.sh
```

### 2. CNNs Training on the Rocks-13
Execute the CNNs training script for the Rocks-13 dataset.
```shell
02_rocks13_cnn.sh
```


```bash
@article{chen2025petromind,
  title={PetroMind: A multimodal petrographic model for rock image classification and lithological description generation},
  author={Chen Zhongliang,Zheng Chaojie, Hu Zhaoqi, Duan Jianchao},
  journal={},
  year={},
  publisher={},
  doi={}
}
```

## 🙏 Acknowledgement
Our code is based on [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL). We sincerely appreciate their contributions and authors for releasing source codes. We thank Professor Su Guifen from the Cores and Samples Center of Natural Resources, China Geological Survey for permission for photographing in the rock specimen room.

## 🤖 Contributing
Feel free to contact c_mulder@163.com if you have any question.
