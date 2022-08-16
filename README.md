

# 基于Paddle复现

## 1. 论文简介

论文名称：[Polarized Self-Attention: Towards High-quality Pixel-wise Regression](https://arxiv.org/pdf/2107.00782.pdf)



作者基于OCRNet_msacle-HRNetV2_W48设计了一种新的通道注意力和空间注意力模块，并应用于HRNet的Basic模块上，使得结果在Cityscapes验证集上达到了SOTA精度。

## 2.复现精度

注意：本文复现环境是在baiduaistudio上的notebook环境，所以有些配置参数也是基于notebook环境的。 如果想完全跑通该repo在其他环境下也可自行更改一些路径配置，比较简单此处不在啰嗦。

在Cityscapes的测试集的测试效果如下表,达到验收指标，miou=87.15 满足精度要求 86.8



精度和loss可以在train.log中看到训练的详细过程

## 3.环境依赖

通过以下命令安装所需要的环境

~~~shell
pip install -r requirements.txt
~~~

## 4.数据集介绍

Cityscapes数据集，下载地址为[Cityscapes Dataset – Semantic Understanding of Urban Street Scenes (cityscapes-dataset.com)](https://www.cityscapes-dataset.com/)

数据集格式  

```css
.
└── cityscapes
    ├── leftImg8bit
    │   └── train
	│		└── acchen
	│	└── val
	│		└── acchen
    └── gtFine
        ├── train
        ├── val
```

## 5. 快速开始

### 克隆本项目

~~~shell
git clone https://github.com/marshall-dteach/PSA.git
~~~
### 训练模型

  ~~~shell
python train.py \
--config configs/psanet/psa_hrnetv2_psa_cityscapes_1024x2048_150k.yml \
--do_eval

python -m paddle.distributed.launch train.py \
		--config configs/psanet/psa_hrnetv2_psa_cityscapes_1024x2048_150k.yml \
  ~~~
  ### 验证模型

  ~~~shell
python val.py \
        --config configs/psanet/psa_hrnetv2_psa_cityscapes_1024x2048_150k.yml \
        --model_path ../model.pdparams \
  ~~~
### 模型导出

~~~shell
python3 export.py \
--config ./test_tipc/configs/psanet/psa_hrnetv2_psa_cityscapes_1024x2048_150k.yml \
--model_path=./test_tipc/output/psaet/norm_gpus_0_autocast_null/best_model.pdparams\ ---save_dir=./test_tipc/output/PSANet/norm_gpus_0_autocast_null\
~~~



### TIPC

~~~shell
bash test_tipc/prepare.sh ./test_tipc/configs/psanet/train_infer_python.txt 'lite_train_lite_infer'
bash test_tipc/test_train_inference_python.sh ./test_tipc/configs/psanet/train_infer_python.txt 'lite_train_lite_infer'
~~~

### 动态推理图

比较模型预测与ground truth  
![1](https://user-images.githubusercontent.com/63546191/169755335-068bbf51-25c2-4bc3-a589-adcc5c2261eb.png)  
![2](https://user-images.githubusercontent.com/63546191/169755356-e49bd5d2-b293-467f-8822-c40e959536e7.png)  
![3](https://user-images.githubusercontent.com/63546191/169755371-fe093a13-7115-4b86-9faf-1104c8c4c8b0.png)  
![4](https://user-images.githubusercontent.com/63546191/169755407-3fb01395-ec1d-4398-bfc8-20d42ce3950b.png)  
![5](https://user-images.githubusercontent.com/63546191/169755436-936867a7-d53f-4588-9b48-72fff455dc70.png)  
![6](https://user-images.githubusercontent.com/63546191/169755571-93992eb7-2a6e-4e3f-aa5f-10105d45f505.png)  

### 6. 代码详细结构说明

~~~shell
PaddleSeg
├── configs         # My model configuration stays here.  
├── test_tipc       # test_tipc stays here.
├── deploy          # deploy related doc and script.
├── paddlelseg  
│   ├── core        # the core training, val and test file.
│   ├── datasets    # Data stays here.
│   ├── models  
│   ├── transforms  # the online data transforms
│   └── utils       # all kinds of utility files
├── export.py
├── tools           # Data preprocess including fetch data, process it and split into training and validation set
├── train.py
├── val.py
|—— predict.py
~~~

## 7.模型信息

|   信息   |        描述         |
| :------: | :-----------------: |
| 模型名称 |  OCRNet-HRNet+psa   |
| 框架版本 | PaddlePaddle==2.2.2 |

## 8. 说明

感谢百度提供的算力，以及举办的本场比赛，让我增强对paddle的熟练度，加深对模型的理解！
