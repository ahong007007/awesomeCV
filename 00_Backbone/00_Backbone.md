# Backbone

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

---

## Framework

-PyTorch官方文档框架介绍。

  -[PyTorch: An Imperative Style, High-Performance Deep Learning Library](https://arxiv.org/pdf/1912.01703v1.pdf)

---

## survey

- 关于CNN的一篇综述。包括CNN各个组件，演化历史，内容不够深入也不够全面，但是可以作为复习的框架。

  -[A Survey of the Recent Architectures of Deep Convolutional Neural Networks](https://arxiv.org/pdf/1901.06032.pdf)

- <https://paperswithcode.com/task/image-classification>

---

## backbone

- Res2Net,南开大学提出。计算机视觉的主题是提取更好的特征表示，多尺度特征提取是图像分类，识别，检测，分割的重要手段，
FPN/ResNet/ResNeXt/DLA/DenseNet等模型都在构造各种提高性能的连接，赏心悦目的美学结构，终极目标应该是何凯明等人提出的
网络随机生成器。Res2Net的基本结构很容易理解，基本单元拆分Res2Net为分组卷积和SENet，显著降低计算量同时提高准确率。论文
在分类，检测，语义分割，实体分割，显著性分割等领域均做了充分的实验，比如Res2Net-50相比ResNet-50，在ImageNet数据集
top-1分类误差降低0.93%，而FLOPs降低69%。期待源码以及更多领域提高性能和实时性。

  - [Res2Net: A New Multi-scale Backbone Architecture](https://arxiv.org/pdf/1904.01169.pdf) :star::star::star::star::star:

- CVPR2019论文，印度坎普尔提出一种改进的卷积方式HetConv(Heterogeneous Kernel-Based Convolution)。相比标准卷积，inception mobilenet等提出
depthwise conv、pointwise conv、groupwise conv减少模型计算量，轮提出的异形卷积HetConv可以看做分组卷积的一种变体，只不过卷积核是有3x3,1x1组成，
延伸可以使用5x5,7x7，这样并排使用且分组卷积的方式。优点：论文详细对比了DP,DW,GW等卷积方式的计算量。缺点：VGG-16模型以及CIFAR-10数据集实验无说服力，
仅在ResNet-50和ImageNet实验说明减少30% FLOPs，缺少在检测，分割等领域的实验对比。

  - [HetConv: Heterogeneous Kernel-Based Convolutions for Deep CNNs](https://arxiv.org/pdf/1903.04120.pdf)

- google提出的backbone模型，主要是对现有模型基础上depth/width/resolution尺度变换，即三个维度系数的设计（手工调参？）。这些维度都是>1的乘法，为什么准确率提高的同时，计算量还降低？模型没有结合SENet等常见trick，貌似用使用更多的5x5卷积。

  - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/pdf/1905.11946.pdf)

- FaceBook作品，在ImageNet Top 1 86.4%。

  - [2019][Fixing the train-test resolution discrepancy](https://arxiv.org/pdf/1906.06423.pdf)

- Google大脑团队提出，backbone基于EfficientNet，self-training framework训练CNN，主要训练过程包括1.在ImageNet真值集训练EfficientNet,此为teacher model。 2.基于训练的EfficientNet对没有标签的ImageNet图像(300M图片量),生成伪标签。
3.在ImageNet和伪标签训练集，训练student model，训练完成后的student model变成teacher model。重复迭代以上过程，不断生成新的student model。
  - teacher model训练时没有数据增广策略，而student model时数据采用dropout,Randaugment,stochastic depth策略。
  - 每次生成的student model参数量大于teacher model。
  - 训练时有个trick,data balancing。某类中图像较少，复制这类图像。某类中图像较多，置信度较高。
  - 站在巨人的肩上很重要，EfficientNet、Randaugment都是Google自家产品，TPU自助不限量，大佬太忙，有个idea就可以让小弟做，作出成果的比例不是一般的高。
  - [Self-training with Noisy Student improves ImageNet classification](https://arxiv.org/pdf/1911.04252.pdf)

- Facebook何凯明提出。论文提出在不仅在ImagetNet，其它PASCAL VOC, COCO检测/分割7个数据集，MoCo的方法unsupervised性能要强于supervised，那么MoCo+EfficientNet之类的backbone，是否可以强者更强，摘取更高准确率？
  - [Momentum Contrast for Unsupervised Visual Representation Learning](https://128.84.21.199/pdf/1911.05722.pdf)

---

## Tiny Backbone

- google经典作品MobileNet.主要Depthwise Separable Convolution替代普通卷积，特征分辨率缩放因子。
  - Depthwise Separable Convolution=Depthwise conv+pointwise conv：具体计算过程示意图可参看material目录。
  - MobileNet提出有影响力的缩放因子：通道缩放因子α和分辨率特征因子β，但是超参数是固定的，损失特征表达能力。EfficientNet延续缩放特征的设计思路，但是可学习的。
  - MobileNet V1没有shortcut连接，是直筒型结构。
  - 下采样时strided-conv替代pooling：小模型不容易过拟合，易出现欠拟合。加入pooling层容易丢失有用信息，增加模型欠拟合可能性。
  - 看论文和看leetcode一样，看懂不一定会，细节多扣为什么这样，而不是那样。作者认为的理所当然，自己不能当成必然。
  - [2018][MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf)

- google在移动端设计的backbone又一个经典作品。论文主要有两点改进：Inverted residuals和Linear bottlenecks。
  - Inverted residuals:ResNet是1*1 0.25倍压缩→标准卷积提特征→1*1 4倍扩张，而Mobile v2采用1*1 6倍扩展→ Depthwise卷积→1*1 6倍压缩。
  如果在residuals模块采用压缩方式，DW提取特征过少。
  - Linear bottlenecks:指在bottlenecks取消Relu6。因为Relu6对于负的输入，输出为0。residuals模块在压缩特征，经过Relu6后会损失更多特征表达能力。
在Xception论文中，已经证明Depthwise-conv后不接relu会更好效果。
  - [2018][MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/pdf/1801.04381.pdf)


- 威斯康星大学麦迪逊分校和亚马逊等联合提出移动端分类神经网络架构：ANTNets(Attention NesTed Network),相比MobileNetv2，ImageNet
Top1 提升 0.8%，速度提升20%（157ms iphone 5s).论文主要是设计神经网络，架构基于Block堆叠，每个Block包括1x1，3x3 dwise，Channel attention，Group-wise，
Inverted Residual Block。Channel attention与SENet不同，论文提出的Channel attention是自适应学习，从输入端到输出端Reduction Ratio (r)逐渐增加。
分类网络有两个设计方向：何凯明等提出的RandWire-WS和各个conv组件排列组合，怎么看都像是升级版的连连看。
  - 缺点：depth multiplier (a = 1.4)时与MobileNet v2性能接近，差别不明显。
  - [ANTNets: Mobile Convolutional Neural Networks for Resource Efficient Image Classification](https://arxiv.org/pdf/1904.03775.pdf)

---  

## others

- Facebook作品，论文没有提出任何模型，或针对特定任务改进。论文任务公开数据集COCO/ImageNet/OpenIamge等对地理位置或收入水平低区域存在偏差，相应的图像数据分布较少。话说假如训练的模型可以识别全球目标的，在欠发达区域使用率也较少吧。
  - [Does Object Recognition Work for Everyone?](https://arxiv.org/pdf/1906.02659.pdf)

---

## Attention

[An Empirical Study of Spatial Attention Mechanisms in Deep Networks](https://arxiv.org/pdf/1904.05873.pdf)

---

## Knowledge

- 感受野计算：
  - 初始feature map 感受野为1。
  - 每经一个conv k*k 卷积，感受野r=r+(k-1)。
  - maxpool2x2 或者stride 2下采样，感受野r = rx2。
  - (conv k*k +maxpool2x2)或(conv k*k,s=2) r=rx2 +k-1。
  - 1*1不敢变感受野，FC和GAP感受野是输入图像。
  - 多分支感受野是最大分支之路。shotcut不改变感受野。
  - ReLU/BN/droupout元素不改变感受野。
  - CNN的感受野通常大于输入分辨率。
  - 深度CNN为保持分辨率每个conv都要加padding，所以等效到输入图像的padding非常大。
- [A guide to convolution arithmetic for deep learning](https://arxiv.org/pdf/1603.07285.pdf)

- [A guide to receptive field arithmetic for Convolutional Neural Networks](https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807)