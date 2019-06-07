
# Backbone

- Res2Net,南开大学提出。计算机视觉的主题是提取更好的特征表示，多尺度特征提取是图像分类，识别，检测，分割的重要手段，
FPN/ResNet/ResNeXt/DLA/DenseNet等模型都在构造各种提高性能的连接，赏心悦目的美学结构，终极目标应该是何凯明等人提出的
网络随机生成器。Res2Net的基本结构很容易理解，基本单元拆分Res2Net为分组卷积和SENet，显著降低计算量同时提高准确率。论文
在分类，检测，语义分割，实体分割，显著性分割等领域均做了充分的实验，比如Res2Net-50相比ResNet-50，在ImageNet数据集
top-1分类误差降低0.93%，而FLOPs降低69%。期待源码以及更多领域提高性能和实时性。

  - [Res2Net: A New Multi-scale Backbone Architecture](https://arxiv.org/pdf/1904.01169.pdf)


- 威斯康星大学麦迪逊分校和亚马逊等联合提出移动端分类神经网络架构：ANTNets(Attention NesTed Network),相比MobileNetv2，ImageNet
Top1 提升 0.8%，速度提升20%（157ms iphone 5s).论文主要是设计神经网络，架构基于Block堆叠，每个Block包括1x1，3x3 dwise，Channel attention，Group-wise，
Inverted Residual Block。Channel attention与SENet不同，论文提出的Channel attention是自适应学习，从输入端到输出端Reduction Ratio (r)逐渐增加。
分类网络有两个设计方向：何凯明等提出的RandWire-WS和各个conv组件排列组合，怎么看都像是升级版的连连看。

  缺点：depth multiplier (a = 1.4)时与MobileNetv2性能接近，差别不明显。

  - [ANTNets: Mobile Convolutional Neural Networks for Resource Efficient Image Classification](https://arxiv.org/pdf/1904.03775.pdf)

- CVPR2019论文，印度坎普尔提出一种改进的卷积方式HetConv(Heterogeneous Kernel-Based Convolution)。相比标准卷积，inception mobilenet等提出
depthwise conv、pointwise conv、groupwise conv减少模型计算量，轮提出的异形卷积HetConv可以看做分组卷积的一种变体，只不过卷积核是有3x3,1x1组成，
延伸可以使用5x5,7x7，这样并排使用且分组卷积的方式。优点：论文详细对比了DP,DW,GW等卷积方式的计算量。缺点：VGG-16模型以及CIFAR-10数据集实验无说服力，
仅在ResNet-50和ImageNet实验说明减少30% FLOPs，缺少在检测，分割等领域的实验对比。

  - [HetConv: Heterogeneous Kernel-Based Convolutions for Deep CNNs](https://arxiv.org/pdf/1903.04120.pdf)

- google提出的backbone模型，主要是对现有模型基础上depth/width/resolution尺度变换，即三个维度系数的设计（手工调参？）。这些维度都是>1的乘法，为什么准确率提高的同时，计算量还降低？模型没有结合SENet等常见trick，貌似用使用更多的5x5卷积。

  - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/pdf/1905.11946.pdf)

## others
- Facebook作品，论文没有提出任何模型，或针对特定任务改进。论文任务公开数据集COCO/ImageNet/OpenIamge等对地理位置或收入水平低区域存在偏差，相应的图像数据分布较少。话说假如训练的模型可以识别全球目标的，在欠发达区域使用率也较少吧。

  - [Does Object Recognition Work for Everyone?](https://arxiv.org/pdf/1906.02659.pdf)

## 待记录

[An Empirical Study of Spatial Attention Mechanisms in Deep Networks](https://arxiv.org/pdf/1904.05873.pdf)



https://paperswithcode.com/area/computer-vision
