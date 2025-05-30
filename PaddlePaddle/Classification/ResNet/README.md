
# ResNet50



## 1. 模型概述
ResNet是一种深度卷积神经网络模型，采用了残差网络（ResNet）的结构，通过引入残差块（Residual Block）以解决深度神经网络训练中的梯度消失和表示瓶颈问题。ResNet模型在各种计算机视觉任务中表现优异，如图像分类、目标检测和语义分割等。由于其良好的性能和广泛的应用，ResNet已成为深度学习和计算机视觉领域的重要基础模型之一。
当前支持的ResNet模型包括：ResNet50、ResNet18 和ResNet101。


## 2. 快速开始
使用本模型执行训练的主要流程如下：
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 构建Docker环境：介绍如何使用Dockerfile创建模型训练时所需的Docker环境。
4. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

请参考[基础环境安装](../../../doc/Environment.md)章节，完成训练前的基础环境检查和安装。

### 2.2 准备数据集
#### 2.2.1 获取数据集

ResNet50运行在ImageNet 1k上，这是一个来自ILSVRC挑战赛的广受欢迎的图像分类数据集。您可以点击[此链接](https://image-net.org/download-images)从公开网站中下载数据集。

#### 2.2.2 解压数据集

- 解压训练数据集
    ```
    mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
    tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
    find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
    cd ..
    ```
- 解压测试数据集
    ```
    mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
    ```

解压后的数据集目录结构如下所示：
```
    └── ImageNet2012
        ├──train
        │   ├──类别1
        │   │   ├──图片1
        │   │   ├──图片2
        │   │   └── ...
        │   ├──类别2
        │   │   ├──图片1
        │   │   ├──图片2
        │   │   └── ...
        │   └── ...
        ├──val
        │    ├──图片1
        │    ├──图片2
        │    ├──图片3
        │    ├──图片4
        │    └── ...
        ├──train_list.txt
        └──val_list.txt
   ```
train_list.txt、val_list.txt 的格式如下：

```
# 每一行采用"空格"分隔图像路径与标注

# 下面是 train_list.txt 中的格式样例
train/n01440764/n01440764_10026.JPEG 0
...

# 下面是 val_list.txt 中的格式样例
val/ILSVRC2012_val_00000001.JPEG 65
...
```
具体说明可参考[PaddleClas数据集说明](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/training/single_label_classification/dataset.md#1-%E6%95%B0%E6%8D%AE%E9%9B%86%E6%A0%BC%E5%BC%8F%E8%AF%B4%E6%98%8E)

### 2.3 构建Docker环境

使用Dockerfile，创建运行模型训练所需的Docker环境。
1. 执行以下命令，进入Dockerfile所在目录。
    ```
    cd <modelzoo-dir>/PaddlePaddle/Classification/ResNet
    ```
    其中： `modelzoo-dir`是ModelZoo仓库的主目录。

2. 执行以下命令，构建名为`paddle_r50`的镜像。
    ```
    DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0 docker build . -t paddle_r50
    ```
3. 执行以下命令，启动容器。
    ```
    docker run  -itd --name r50_pd -v <dataset_path>:/imagenet  --net=host --ipc=host --device /dev/tcaicard0 --device /dev/tcaicard1 --device /dev/tcaicard2 --device /dev/tcaicard3 --shm-size=128g paddle_r50 /bin/bash
    ```

    其中：`-v`参数用于将主机上的目录或文件挂载到容器内部，对于模型训练，您需要将主机上的数据集目录挂载到docker中的`/imagenet`目录。更多容器配置参数说明参考[文档](../../../doc/Docker.md)。

4. 执行以下命令，进入容器。
    ```
    docker exec -it r50_pd /bin/bash
    ```

5. 执行以下命令，启动虚拟环境。
    ```
    conda activate paddle_env
    ```


### 2.4 启动训练
1. 在Docker环境中，进入训练脚本所在目录。
    ```
    cd /workspace/Classification/ResNet/run_scripts
    ```

2. 运行训练。该模型支持单机单卡、单机四卡、单机八卡、两机八卡.

    -  单机单卡
    ```
    python run_resnet.py --model_name resnet50 --nproc_per_node 4 --bs 64 --lr 0.256 --device sdaa  --epoch 50 --dataset_path /imagenet  --grad_scale True --autocast True
    ```

    - 单机四卡
    ```
    python run_resnet.py --model_name resnet50  --nproc_per_node 16 --bs 64 --lr 1.024 --device sdaa  --epoch 50 --dataset_path /imagenet  --grad_scale True --autocast True

    ```

    - 单机八卡
    ```
    python run_resnet.py --model_name resnet50  --nproc_per_node 32 --bs 64 --lr 2.048 --device sdaa  --epoch 90 --dataset_path /imagenet  --grad_scale True --autocast True

    ```

    - 两机八卡
    ```
    # node0
    python run_resnet.py --model_name resnet50 --master_addr ip_address1,ip_address2 --master_port 29500 --nproc_per_node 16 --bs 64 --lr 2.048 --device sdaa  --epoch 50 --dataset_path /imagenet  --grad_scale True --autocast True --nnode 2

    # node1
    python run_resnet.py --model_name resnet50  --master_addr ip_address1,ip_address2 --master_port 29500 --nproc_per_node 16 --bs 64 --lr 2.048 --device sdaa  --epoch 50 --dataset_path /imagenet  --grad_scale True --autocast True --nnode 2

    #请将上述命令中的ip_address1, ip_address2 修改为所使用节点的ip地址，其中ip_address1为主节点
    ```

    更多训练参数参考[README](run_scripts/README.md)

### 2.5 训练结果

|加速卡数量  | 模型 |  混合精度 |Batch size|Shape| AccTop1|
|:-:|:-:|:-:|:-:|:-:|:-:|
|1| ResNet18 |是|256|224*224|  70.36% |
|1| ResNet50 |是|256|224*224|  76.09% |
|1| ResNet101 |是|256|224*224| 78.14% |

