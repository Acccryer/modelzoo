# NestFuse

---

### The re-implementation of IEEE Transactions 2020 NestFuse paper idea

![](figure/framework.png)

![](figure/train.png)

This code is based on [H. Li, X. -J. Wu and T. Durrani, "NestFuse: An Infrared and Visible Image Fusion Architecture Based on Nest Connection and Spatial/Channel Attention Models," in IEEE Transactions on Instrumentation and Measurement, vol. 69, no. 12, pp. 9645-9656, Dec. 2020](https://ieeexplore.ieee.org/document/9127964)

---

## Description 描述

- **基础框架：** AutoEncoder
- **任务场景：** 用于红外可见光图像融合，Infrared Visible Fusion (IVF)。
- **项目描述：** Nestfuse 的 PyTorch 实现。fusion strategy 基于注意力机制。
- **论文地址：**
  - [arXiv](https://arxiv.org/abs/2007.00328)
  - [IEEEXplore](https://ieeexplore.ieee.org/document/9127964)
- **参考项目：**
  - [imagefusion-nestfuse](https://github.com/hli1221/imagefusion-nestfuse) 官方代码。
---

## Idea 想法

[MS-COCO 2014](http://images.cocodataset.org/zips/train2014.zip)(T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollar, and C. L. Zitnick. Microsoft coco: Common objects in context. In ECCV, 2014. 3-5.) is utilized to train our auto-encoder network.

In our fusion strategy, we focus on two types of features: spatial attention model and channel attention model. The extracted multi-scale deep features are processed in two phases.

---

## Structure 文件结构

```shell
├─ data_test             # 用于测试的不同图片
│  ├─Road          	  	# Gray  可见光+红外
│  └─Tno           		# Gray  可见光+红外
│ 
├─ data_result    # run_infer.py 的运行结果。使用训练好的权重对fusion_test_data内图像融合结果 
│ 
├─ models       
│  ├─ fusion_strategy            # 融合策略              
│  └─ NestFuse                   # 网络模型
│ 
├─ runs              # run_train.py 的运行结果
│  └─ train_01-18_17-36
│     ├─ checkpoints # 模型权重
│     └─ logs        # 用于存储训练过程中产生的Tensorboard文件
|
├─ utils      	                # 调用的功能函数
│  ├─ util_dataset.py            # 构建数据集
│  ├─ util_device.py        	# 运行设备 
│  ├─ util_fusion.py             # 模型推理
│  ├─ util_loss.py            	# 结构误差损失函数
│  ├─ util_train.py            	# 训练用相关函数
│  └─ utils.py                   # 其他功能函数
│ 
├─ configs.py 	    # 模型训练超参数
│ 
├─ run_infer.py   # 该文件使用训练好的权重将data_test内的测试图像进行融合
│ 
└─ run_train.py      # 该文件用于训练模型

```

## Usage 使用说明

### 准备
#### 环境安装
提供了Dockerfile，下面使用Dockerfile安装
1. 构建Dockerfile，
```
docker build -t densefuse:latest ${your_path}/DenseFuse/
```
2. 运行并进入容器
```
docker run -it --name densefuse --net=host -v /mnt/:/mnt -v /mnt_qne00/:/mnt_qne00 --privileged --shm-size=300g densefuse:latest
```
#### 数据集
本次适配使用COCO2017数据集，下载地址：https://cocodataset.org/#home
或者在共享目录/mnt_qne00/dataset/coco/train2017/

### Trainng

#### 从零开始训练

* 参数说明：参考README.md, 需要求改参数可以在test.sh中修改
                                                                          
* 设置完成参数后，执行**bash run_scripts/test.sh**即可开始训练：

| 参数名              | 说明                                                                              |
|------------------|---------------------------------------------------------------------------------|
| image_path       | 用于训练的数据集的路径                                                                     |
| gray             | 为`True`时会进入灰度图训练模式，生成的权重用于对单通道灰度图的融合; 为`False`时会进入彩色RGB图训练模式，生成的权重用于对三通道彩色图的融合; |
| train_num        | `MSCOCO/train2017`数据集包含**118,287**张图像，设置该参数来确定用于训练的图像的数量                        |
| resume_path      | 默认为None，设置为已经训练好的**权重文件路径**时可对该权重进行继续训练，注意选择的权重要与**gray**参数相匹配                  |
| device           | 模型训练设备 cpu or gpu                                                               |
| batch_size       | 批量大小                                                                            |
| num_workers      | 加载数据集时使用的CPU工作进程数量，为0表示仅使用主进程，（在Win10下建议设为0，否则可能报错。Win11下可以根据你的CPU线程数量进行设置来加速数据集加载） |
| learning_rate    | 训练初始学习率                                                                            |
| num_epochs       | 训练轮数                                                                               |

```python
    # 数据集相关参数
    parser.add_argument('--image_path', default=r'E:/project/Image_Fusion/DATA/COCO/train2017', type=str, help='数据集路径')
    parser.add_argument('--gray', default=True, type=bool, help='是否使用灰度模式')
    parser.add_argument('--train_num', default=10000, type=int, help='用于训练的图像数量')
    # 训练相关参数
    parser.add_argument('--resume_path', default=None, type=str, help='导入已训练好的模型路径')
    parser.add_argument('--device', type=str, default=device_on(), help='训练设备')
    parser.add_argument('--batch_size', type=int, default=4, help='input batch size, default=4')
    parser.add_argument('--num_workers', type=int, default=0, help='载入数据集所调用的cpu线程数')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs to train for, default=10')
    parser.add_argument('--lr', type=float, default=1e-4, help='select the learning rate, default=1e-2')
    # 打印输出
    parser.add_argument('--output', action='store_true', default=True, help="shows output")
```

* 你可以在运行窗口看到类似的如下信息：

```
D:\Python\Miniconda\envs\fusion_gpu\python.exe E:\Git_Project\Image-Fusion\NestFuse_2020\run_train.py 
==================模型超参数==================
----------数据集相关参数----------
image_path: ../dataset/COCO_train2014
gray_images: True
train_num: 80000
----------训练相关参数----------
device: cuda
batch_size: 16
num_epochs: 4
num_workers: 0
learning rate : 0.0001
resume_path: None
==================模型超参数==================
Loaded 80000 images
训练数据载入完成...
设备就绪...
Tensorboard 构建完成，进入路径：./runs\train_01-18_17-36\logs_Gray_epoch=4
然后使用该指令查看训练过程：tensorboard --logdir=./
测试数据载入完成...
initialize network with normal type
网络模型及优化器构建完成...
Epoch [1/4]: 100%|██████████| 5000/5000 [32:34<00:00,  2.56it/s, pixel_loss=0.1000, ssim_loss=0.1832, lr=0.000100]
Epoch [2/4]: 100%|██████████| 5000/5000 [27:50<00:00,  2.99it/s, pixel_loss=0.1027, ssim_loss=0.1788, lr=0.000090]
Epoch [3/4]: 100%|██████████| 5000/5000 [28:46<00:00,  2.90it/s, pixel_loss=0.0002, ssim_loss=0.0000, lr=0.000081]
Epoch [4/4]: 100%|██████████| 5000/5000 [27:49<00:00,  2.99it/s, pixel_loss=0.0001, ssim_loss=0.0000, lr=0.000073]
Finished Training
训练耗时：7024.94秒
Best loss: 0.002408
```

* Tensorboard查看训练细节：
  * **logs**文件夹下保存Tensorboard文件
  * 进入对于文件夹后使用该指令查看训练过程：`tensorboard --logdir=./`
  * 在浏览器打开生成的链接即可查看训练细节







