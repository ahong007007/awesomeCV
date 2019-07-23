
#
- Why GEMM is at the heart of deep learning
  - [Why GEMM is at the heart of deep learning](https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/)

- 卷积算法优化，GEMM(GEneral Matrix to Matrix Multiplication)->GEMM−based Convolution。从C代码看，有两层循环，相对于GEMM为何反而能更高效？另外论文
提出算法uIndirectGEMM在pytorch(NCHW layout),Transposed Convolution operator,depthwise convolutions之外才更高效，很少有论文这么直接。

  - [The Indirect Convolution Algorithm](https://arxiv.org/pdf/1907.02129.pdf)

# Optimization




# Training

1、CVPR2019论文，布兰迪斯大学和微软联合提出。论文的motation来自于模型的剪枝（pruned）是因为卷积核之间正交性低，而具有跳跃连接的
ResNet/DenseNet等在一定程度改善。论文认为卷积核的冗余是由于训练策略引起。先训练整个网络，根据metric drop掉p%的filter，再训练剩余
的网络，之后增加drop的filter（初始化方式：现有filters正交，迭代这个过程N次。复现论文需要4个额外参数： full network and the sub-network iterations,
，滤波器drop的百分比，drop/relearn交替次数N,以及滤波器评价metric。
缺点或不足：
1、论文提出的4个超参数，metric给出计算公式，其他三个没有给出选择的依据，实验也不充分说明各个变量的变化趋势。
2、论文在ResNet-101训练的Object Detection模型，从41.7mAP提高到44.5mAP，这比CVPR2019所有的目标检测模型涨点都要高，
可惜论文描述不详细，是不是可以再写一个ICCV2019的论文？

RePr: Improved Training of Convolutional Filters.[pdf](https://arxiv.org/pdf/1811.07275.pdf)

# Optimization

1、商汤提出Switchable Whitening，相比Batch Normalization (BN) , Instance Normalization，Layer Normalization (LN)，
论文在classification (CIFAR-10/100, ImageNet), semantic segmentation (ADE20K, Cityscapes), domain adaptation
(GTA5, Cityscapes), and image style transfer (COCO)均有良好表现。

[Switchable Whitening for Deep Representation Learning](https://arxiv.org/pdf/1904.09739.pdf)