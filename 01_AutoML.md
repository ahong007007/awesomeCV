
# NAS

[2018,DSO-NAS] You Only Search Once: Single Shot Neural Architecture Search via Direct Sparse Optimization.[PDF](https://arxiv.org/pdf/1811.01567.pdf)

Auto is the new black — Google AutoML, Microsoft Automated ML, AutoKeras and auto-sklearn

https://medium.com/@santiagof/auto-is-the-new-black-google-automl-microsoft-automated-ml-autokeras-and-auto-sklearn-80d1d3c3005c

## [2019年3月]

1、澳大利亚欧缇莫的大学

Auto-ReID: Searching for a Part-aware ConvNet for Person Re-Identification [PDF](https://arxiv.org/pdf/1903.09776.pdf)

2、清华大学和旷视科技提出，基于MobileNet V1/V2 网络的自动化通道剪枝，相比AMC和NetAdapt有提升

MetaPruning: Meta Learning for Automatic Neural Network Channel Pruning [PDF](https://arxiv.org/pdf/1903.10258.pdf)

3、中科院自动化所和旷视联合提出，Object Detection with FPN on COCO优于ResNet101,但是FLOPs比ResNet50低。基于ShuffleNetV2的架构也有较好的表现。

DetNAS: Neural Architecture Search on Object Detection [PDF](https://arxiv.org/pdf/1903.10979v1.pdf)

3、facebook开源框架，基于MCTS和DNN,解决分类，目标检测，风格迁移，图像描述4个任务。

AlphaX: eXploring Neural Architectures with Deep Neural Networks and Monte Carlo Tree Search [PDF](https://arxiv.org/pdf/1903.11059.pdf)

4、伊利诺伊大学厄巴纳-香槟分校提出的以及channel select算法，论文对mobilenetv1/2 MNasNet 性能提高，推断延迟降低。

Network Slimming by Slimmable Networks:Towards One-Shot Architecture Search for Channel Numbers. [PDF](https://arxiv.org/pdf/1903.11728.pdf)

## classifier

NAS一般是依据人类设计的CNN构造cell,堆叠cell单元。Facebook Ross Girshick，Kaiming He等设计一个基于图论的网络生成器生成随机网络。
实验效果在RandWire-WS数据集，RandWire-WS相比MobileNet v2，Amoeba-C没有太大提升，在COCO目标检测数据集相比ResNeXt-50和ResNeXt-101，
在FLOPs计算量相同情况下，最高有1.7%的提升。

Exploring Randomly Wired Neural Networks for Image Recognition.[pdf](https://arxiv.org/pdf/1904.01569.pdf)
5、






