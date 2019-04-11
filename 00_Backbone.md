# Backbone

1、Res2Net,南开大学提出。计算机视觉的主题是提取更好的特征表示，多尺度特征提取是图像分类，识别，检测，分割的重要手段，
FPN/ResNet/ResNeXt/DLA/DenseNet等模型都在构造各种提高性能的连接，赏心悦目的美学结构，终极目标应该是何凯明等人提出的
网络随机生成器。Res2Net的基本结构很容易理解，基本单元拆分Res2Net为分组卷积和SENet，显著降低计算量同时提高准确率。论文
在分类，检测，语义分割，实体分割，显著性分割等领域均做了充分的实验，比如Res2Net-50相比ResNet-50，在ImageNet数据集
top-1分类误差降低0.93%，而FLOPs降低69%。期待源码以及更多领域提高性能和实时性。

Res2Net: A New Multi-scale Backbone Architecture.[pdf](https://arxiv.org/pdf/1904.01169.pdf)


2、威斯康星大学麦迪逊分校和亚马逊等联合提出移动端分类神经网络架构：ANTNets(Attention NesTed Network),相比MobileNetv2，ImageNet
Top1 提升 0.8%，速度提升20%（157ms iphone 5s).论文主要是设计神经网络，架构基于Block堆叠，每个Block包括1x1，3x3 dwise，Channel attention，Group-wise，
Inverted Residual Block。Channel attention与SENet不同，论文提出的Channel attention是自适应学习，从输入端到输出端Reduction Ratio (r)逐渐增加。
分类网络有两个设计方向：何凯明等提出的RandWire-WS和各个conv组件排列组合，怎么看都像是升级版的连连看。

缺点：depth multiplier (a = 1.4)时与MobileNetv2性能接近，差别不明显。

ANTNets: Mobile Convolutional Neural Networks for Resource Efficient Image Classification.[pdf](https://arxiv.org/pdf/1904.03775.pdf)

# 待记录

