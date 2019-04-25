
## NAS

[2018,DSO-NAS] You Only Search Once: Single Shot Neural Architecture Search via Direct Sparse Optimization.[PDF](https://arxiv.org/pdf/1811.01567.pdf)

Auto is the new black — Google AutoML, Microsoft Automated ML, AutoKeras and auto-sklearn

https://medium.com/@santiagof/auto-is-the-new-black-google-automl-microsoft-automated-ml-autokeras-and-auto-sklearn-80d1d3c3005c

## classifier

- NAS一般是依据人类设计的CNN构造cell,堆叠cell单元。Facebook Ross Girshick，Kaiming He等设计一个基于图论的网络生成器生成随机网络。
实验效果在RandWire-WS数据集，RandWire-WS相比MobileNet v2，Amoeba-C没有太大提升，在COCO目标检测数据集相比ResNeXt-50和ResNeXt-101，
在FLOPs计算量相同情况下，最高有1.7%的提升。为保证公平，论文的随机网络生成器只迭代250 epoch，如果迭代更高的epoch是不是可以生成准确率更高
计算更快的网络模型？
  - 缺点：每个随机生成的网络都需要在完整数据集训练，论文没有说明使用GPU数量和天数

[Exploring Randomly Wired Neural Networks for Image Recognition](https://arxiv.org/pdf/1904.01569.pdf)

## Detection

- 中科院自动化所和旷视联合提出，Object Detection with FPN on COCO优于ResNet101,但是FLOPs比ResNet50低。基于ShuffleNetV2的架构也有较好的表现。

DetNAS: Neural Architecture Search on Object Detection [PDF](https://arxiv.org/pdf/1903.10979v1.pdf)

- Google基于AutoML提出Detection模型，基于RetinaNet网络，解决FPN多尺度金字塔问题。通过Neural Architecture Search搜索各种类型的
top-down,bottom-up特征层的连接方式（还是连连看），取得state-of-art的mAP同时降低推断时间。

NAS-FPN: Learning Scalable Feature Pyramid Architecture for Object Detection.[pdf](https://arxiv.org/pdf/1904.07392.pdf)

## Recognition

- 华为等提出的人脸识别模型，在MS-Celeb-1M和LFW数据集state-of-art。主流人脸识别模型集中于度量学习（Metric Learning）和分类损失函数函数改进（Cross-Entropy Loss，Angular-Softmax Loss，
Additive Margin Softmax LossArcFace Loss等）。论文基于强化学习的NAS设计，network size 和latency作为reward(论文的实验没有对比测试latency或者模型尺寸)，仅说明最小网络参数NASC 16M。
这是NAS在人脸识别的首测尝试，分类，检测，识别都有涉及，图像分割应该不远。

[Neural Architecture Search for Deep Face Recognition](https://arxiv.org/pdf/1904.09523.pdf)

## Semantic  Segmentation

- 驭势科技，新加坡国立大学等联合提出轻量型分类和语义分割网络，成为"东风"网络，主要解决Speed/Accuracy trade-off问题，性能接近于deeplab v2但是速度>50 fps。
论文的backbone基于典型residual block，但是提出Acc(x)和Lat(x)用于评价准确率和推断时间，节省神经网络的随机搜索。另外没有选用RNN/LSTM,而是Random select an architecture。
论文提出的算法，训练200个模型架构时已经去除438个模型架构，每个模型训练5-7个小时(8卡机)。训练200个架构月400GPU/days。
模型在嵌入式硬件TX2和1080Ti均有评测，准确率和性能均有明显优势。去年有项目曾使用ENet/ICNet，速率尚可准确率不如人意，也许"东风"系列有明显改善。

[Partial Order Pruning: for Best Speed/Accuracy Trade-off in Neural Architecture Search](https://arxiv.org/pdf/1903.03777.pdf)

- 李飞飞团队作品。

[Auto-DeepLab:Hierarchical Neural Architecture Search for Semantic Image Segmentation](https://arxiv.org/pdf/1901.02985.pdf)

# Pruning

- 清华大学和旷视科技提出，基于MobileNet V1/V2 网络的自动化通道剪枝，相比AMC和NetAdapt有提升

MetaPruning: Meta Learning for Automatic Neural Network Channel Pruning [PDF](https://arxiv.org/pdf/1903.10258.pdf)


- 伊利诺伊大学厄巴纳-香槟分校提出的以及channel select算法，论文对mobilenetv1/2 MNasNet 性能提高，推断延迟降低。

Network Slimming by Slimmable Networks:Towards One-Shot Architecture Search for Channel Numbers. [PDF](https://arxiv.org/pdf/1903.11728.pdf)

## ReID

-澳大利亚欧缇莫的大学

Auto-ReID: Searching for a Part-aware ConvNet for Person Re-Identification [PDF](https://arxiv.org/pdf/1903.09776.pdf)


# Super-Resolution 

- 小米AI团队团队提出的超分辨率模型。

[Fast, Accurate and Lightweight Super-Resolution with Neural Architecture Search](https://arxiv.org/pdf/1901.07261.pdf)

# Architecture

- facebook开源框架，基于MCTS和DNN,解决分类，目标检测，风格迁移，图像描述4个任务。

[AlphaX: eXploring Neural Architectures with Deep Neural Networks and Monte Carlo Tree Search](https://arxiv.org/pdf/1903.11059.pdf)

## Benchmark on ImageNet

| Architecture       | Top-1 (%) | Top-5 (%) | Params (M) | +x (M) | GPU | Days |
| ------------------ | --------- | --------- | ---------- | ------ | -   | -    |
| [Inception-v1](https://arxiv.org/pdf/1409.4842.pdf)       | 30.2      | 10.1      | 6.6        | 1448   | -   | -    |
| [MobileNet-v1](https://arxiv.org/abs/1704.04861)       | 29.4      | 10.5      | 4.2        | 569    | -   | -    |
| [ShuffleNet](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/0642.pdf)         | 26.3      | -         | ~5         | 524    | -   | -    |
| [MobileNetV2](https://arxiv.org/pdf/1801.04381.pdf)     |28.0          |  -       | 3.4M  | 300M  | - | - |
| MobileNetV2-1.4 |25.3          |  -       |6.9M   | 585M  | - | - |
| [NASNet-A]((http://openaccess.thecvf.com/content_cvpr_2018/papers/Zoph_Learning_Transferable_Architectures_CVPR_2018_paper.pdf))           | 26.0      | 8.4       | 5.3        | 564    | 450 | 3-4  |
| NASNet-B           | 27.2      | 8.7       | 5.3        | 488    | 450 | 3-4  |
| NASNet-C           | 27.5      | 9.0       | 4.9        | 558    | 450 | 3-4  |
| [AmobebaNet-A](https://arxiv.org/pdf/1802.01548.pdf)       | 25.5      | 8.0       | 5.1        | 555    | 450 |  7   |
| AmobebaNet-B       | 26.0      | 8.5       | 5.3        | 555    | 450 |  7   |
| AmobebaNet-C       | 24.3      | 7.6       | 6.4        | 555    | 450 |  7   |
| [Progressive NAS](http://openaccess.thecvf.com/content_ECCV_2018/papers/Chenxi_Liu_Progressive_Neural_Architecture_ECCV_2018_paper.pdf)    | 25.8      | 8.1       | 5.1        | 588    | 100 | 1.5  |
| [DARTS-V2](https://arxiv.org/abs/1806.09055)           | 26.9      | 9.0       | 4.9        | 595    |  1  |  1   |
| [GDAS](http://xuanyidong.com/bibtex/Four-Hours-CVPR19.txt) | 26.0      | 8.5       | 5.3        | 581    |  1  |  0.21   |
| [RandWire-WS](https://arxiv.org/pdf/1904.01569.pdf)        | 25.3±0.25 | 7.8       | 5.6±1      |583±6.2 |  -  |   -     |
