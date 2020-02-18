# Optimization

---

## Convolution Operator

- Why GEMM is at the heart of deep learning
  - [Why GEMM is at the heart of deep learning](https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/)

- 卷积算法优化，GEMM(GEneral Matrix to Matrix Multiplication)->GEMM−based Convolution。从C代码看，有两层循环，相对于GEMM为何反而能更高效？另外论文
提出算法uIndirectGEMM在pytorch(NCHW layout),Transposed Convolution operator,depthwise convolutions之外才更高效，很少有论文这么直接。

  - [The Indirect Convolution Algorithm](https://arxiv.org/pdf/1907.02129.pdf)

---

## Training

- google提出一个理论Internal Covariate Shift(ICS)：在深度学习训练过程中，由于网络参数变化引起内部节点数据分布发送变化。ICS导致训练时网络学习速度降低，梯度易饱和，减缓网络收敛速度。
  - 论文提出的Batch Normalization(BN)包括4个简单公式：求均值（#TODO 什么样的维度求均值？）、求方差、归一正则化、线性变化(缩放和平移)。
  - BN优势：神经网络每层分布相对稳定，加速模型学习速率。减少对网络参数的敏感度，简化调参，网络学习更加稳定；具有一定正则化效果。
  - 适用场景：每个 mini-batch 比较大，数据分布比较接近。在进行训练之前，要做好充分的 shuffle，否则效果差很多。
  - 在ICS基础上，BatchNorm 延伸各种变体：LayerNorm/WeightNorm/CosineNorm。
  - 推断阶段，输入是单个图像，BN是怎么使用的？利用训练时计算的α和β，直接对输入数据进行平移和缩放。
  - [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf)

- CVPR2019论文，布兰迪斯大学和微软联合提出。论文的motation来自于模型的剪枝（pruned）是因为卷积核之间正交性低，而具有跳跃连接的
ResNet/DenseNet等在一定程度改善。论文认为卷积核的冗余是由于训练策略引起。先训练整个网络，根据metric drop掉p%的filter，再训练剩余
的网络，之后增加drop的filter（初始化方式：现有filters正交，迭代这个过程N次。复现论文需要4个额外参数： full network and the sub-network iterations,
，滤波器drop的百分比，drop/relearn交替次数N,以及滤波器评价metric。
缺点或不足：
1、论文提出的4个超参数，metric给出计算公式，其他三个没有给出选择的依据，实验也不充分说明各个变量的变化趋势。
2、论文在ResNet-101训练的Object Detection模型，从41.7mAP提高到44.5mAP，这比CVPR2019所有的目标检测模型涨点都要高，
可惜论文描述不详细，是不是可以再写一个ICCV2019的论文？
  - RePr: Improved Training of Convolutional Filters.[pdf](https://arxiv.org/pdf/1811.07275.pdf)

- 商汤提出Switchable Whitening，相比Batch Normalization (BN) , Instance Normalization，Layer Normalization (LN)，
论文在classification (CIFAR-10/100, ImageNet), semantic segmentation (ADE20K, Cityscapes), domain adaptation
(GTA5, Cityscapes), and image style transfer (COCO)均有良好表现。
  - [Switchable Whitening for Deep Representation Learning](https://arxiv.org/pdf/1904.09739.pdf)
