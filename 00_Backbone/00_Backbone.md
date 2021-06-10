# Backbone

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

---

## Table of Contents

- [Framework](#Framework)
- [survey](#survey)
- [backbone](#backbone)
- [tiny_backbone](#tiny_backbone)
- [Attention](#Attention)
- [Knowledge](#Knowledge)
- [multilabel-classification](#multilabel-classification)
- [others](#others)

---

## Framework

- PyTorch官方文档框架介绍。

  - [PyTorch: An Imperative Style, High-Performance Deep Learning Library](https://arxiv.org/pdf/1912.01703v1.pdf)
- 阿里巴巴退出移动端加速推断框架。
  - [MNN: A Universal and Efficient Inference Engine](https://arxiv.org/pdf/2002.12418.pdf)
  - <https://github.com/alibaba/MNN>

## survey

- 关于CNN的一篇综述。包括CNN各个组件，演化历史，内容不够深入也不够全面，但是可以作为复习的框架。

  -[A Survey of the Recent Architectures of Deep Convolutional Neural Networks](https://arxiv.org/pdf/1901.06032.pdf)

- <https://paperswithcode.com/task/image-classification>
- <https://www.reddit.com/r/computervision/>
- <https://www.reddit.com/r/MachineLearning/>

---

## backbone

- 亚马逊团队，在ResNet50改进。
  - [Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/pdf/1812.01187.pdf)

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
  - backbone的设计转换为grid search问题。
  - 为什么三个维度同时搜索？论文对比发现单个维度对模型的性能影响有限。是深度卷积神经网络的深度、宽度、分辨率，分别设到什么程度，可以帮助网络更好地拟合非线性特性，提取图像语义特征并用合理的计算资源完成模式识别工作。
  - 参数量并不等于实际计算量，resnet150参数量是efficientnet-b211倍，但是速度是efficientnet-b2的87%。
  - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/pdf/1905.11946.pdf)

- FaceBook作品，提出FixResNeXt，ImageNet Top 1 86.4%。当训练与测试时使用的图像分辨率差异较大时，分类器模型会出现性能差异，一般解决方法是数据增强，包括图像的裁剪，水平翻转和色彩抖动。
论文中很多噱头，只是表明图像不同分辨率之间的差异。
  - 论文训练集中使用分辨率低的图像训练，分辨率高的图像做测试。
  - fine-tuning。在训练模型后，通过图像的scale调节，fine-tuning模型。
  - 论文验证scale的策略不仅仅在ImageNet有效，在其他任务迁移后也可提升性能。
  - [2019][Fixing the train-test resolution discrepancy](https://arxiv.org/pdf/1906.06423.pdf)

- FaceBook最新提出在FixRes基础上延伸的FixEfficientNet，top-1：88.5%和top-5：98.7%的准确率，state-of-art水平。
  - FixRes解决的是训练，测试两个阶段图像预处理方面的不同导致的性能差异，FixEfficientNet论文只是一个技术报告，貌似创新点在于 fine-tuning阶段的label smoothing。
  - FixRes时论文写了很多水公式，实际技巧几行代码就能数清楚。而FixEfficientNet SOTA水平，按照国内的揍性，怎么也得水出一片顶会论文吧。
  - [2020][Fixing the train-test resolution discrepancy: FixEfficientNet](https://arxiv.org/pdf/2003.08237v1.pdf)

- Google大脑团队提出NoisyStudent，backbone基于EfficientNet，self-training framework训练CNN，主要训练过程包括1.在ImageNet真值集训练EfficientNet,此为teacher model。 2.基于训练的EfficientNet对没有标签的ImageNet图像(300M图片量),生成伪标签。
3.在ImageNet和伪标签训练集，训练student model，训练完成后的student model变成teacher model。重复迭代以上过程，不断生成新的student model。
  - teacher model训练时没有数据增广策略，而student model时数据采用dropout,Randaugment,stochastic depth策略。
  - 每次生成的student model参数量大于teacher model。
  - 训练时有个trick,data balancing。某类中图像较少，复制这类图像。某类中图像较多，置信度较高。
  - 站在巨人的肩上很重要，EfficientNet、Randaugment都是Google自家产品，TPU自助不限量，大佬太忙，有个idea就可以让小弟做，作出成果的比例不是一般的高。
  - [Self-training with Noisy Student improves ImageNet classification](https://arxiv.org/pdf/1911.04252.pdf)

- Facebook何凯明提出。论文提出在不仅在ImagetNet，其它PASCAL VOC, COCO检测/分割7个数据集，MoCo的方法unsupervised性能要强于supervised，那么MoCo+EfficientNet之类的backbone，是否可以强者更强，摘取更高准确率？
  - [Momentum Contrast for Unsupervised Visual Representation Learning](https://128.84.21.199/pdf/1911.05722.pdf)

- CSPNet：可以增强CNN学习能力的新型Backbone.
  - [CSPNet: A New Backbone that can Enhance Learning Capability of CNN](https://arxiv.org/pdf/1911.11929.pdf)
  - <https://github.com/WongKinYiu/CrossStagePartialNetworks>

- Facebook何凯明团队提出，手动设计网络和NAS结合。
  - [Designing Network Design Spaces](https://arxiv.org/pdf/2003.13678.pdf)

- 亚马逊李沐等提出ResNeSt超越ResNet的各种变体：SENet、ResNeXt、EfficientNet，实现在分类，检测，实例分割等各类视觉任务的提升.
  - backbone是计算机视觉的基础，目标检测，分割和各种姿态分析都是基于backbone开发。NAS可以显著提高backbone，但是浪费较多CPU/GPU资源，论文提出一种基于ResNet手工设计的神经网络架构,增加channel之间的特征表达，可用于CV的各个领域。
  - ResNeSt更像是SK-Net，分组卷积，attention机制的排列组合。
  - 性能显著提升，参数量却没有明显增加? 并行的cardinal group没有增加显存和运算时间？
  - 模型的训练有很多策略，Large Mini-batch Distributed Training，Label Smoothing，Mixup、Large Crop Size、Regularization等。没有什么算法是一招制敌。
  - [ResNeSt: Split-Attention Networks](https://hangzhang.org/files/resnest.pdf)

- CMU设计的知识蒸馏提高ResNet50的分类准确率。
  - 论文使用Teacher Ensembles的方式，不需要标注真值，label由teacher mean 输出(ImageNet由人工标注，每个图不是只有一个目标，不能表示复杂的图像信息).
  - KL散度度量teacher 和student概率分布差异，KL(p||q)简化为cross-entropy loss。
  - 知识蒸馏的天花板是teacher模型的精度，论文用Ensembles方式，
  - [MEAL V2: Boosting Vanilla ResNet-50 to 80%+ Top-1 Accuracy on ImageNet without Tricks](https://arxiv.org/pdf/2009.08453.pdf)

- ICLR2021盲审论文，LambdaResNets 在实现 SOTA ImageNet 准确性的同时，运行速度是 EfficientNets 的4.5 倍左右。
  - [LambdaNetworks: Modeling long-range Interactions without Attention](https://openreview.net/pdf?id=xTJEN-ggl1b)

- 谷歌在imageNet刷榜作品。轮提出的方法是改进self trainging.
  - confirmation bias:不管是teacher还是student，都会有错误偏差，论文的模型如何解决？
  - 资源：a cluster of 2,048 TPUv3 cores  +不开源的JFT data。JFT数据集远高于ImageNet，胜之不武。
  - imagenet榜单前15名都是Google家的，强者恒强，垄断了ImageNet. 
  - [Meta Pseudo Labels](https://arxiv.org/pdf/2003.10580v4.pdf)
  
- 谷歌团体再次刷榜之作,证明训练策略和扩展策略，比网络结构更重要.
  - 1.如果可能过拟合，则缩放模型深度；否则，缩放模型宽度；
  - 2.增加图像分辨率,但是速度比以往的论文中推荐的速度更慢。  
  - [Revisiting ResNets: Improved Training and Scaling Strategies](https://arxiv.org/pdf/2103.07579.pdf)

- Deepmind提出的一种替代Batch normal方法。
  - 轮声称BN计算量大，但是实验效果无论是速度，还是参数，看上去没有显著提升？
    
  - [High-Performance Large-Scale Image Recognition Without Normalization](https://arxiv.org/pdf/2102.06171v1.pdf)

---

## tiny_backbone

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

## Attention

- [An Empirical Study of Spatial Attention Mechanisms in Deep Networks](https://arxiv.org/pdf/1904.05873.pdf)

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

## multilabel-classification

TODO 多标签分类问题

- 介绍文本，图像、视频音频等应用中，如何实现Multi-label的基础问题。
  - [Mining Multi-label Data](http://lkm.fri.uni-lj.si/xaigor/slo/pedagosko/dr-ui/tsoumakas09-dmkdh.pdf)

- 用机器学习的方式，解决多标签分类的问题。
  - [A review on multilabel algorithm](https://www.researchgate.net/publication/263813673_A_Review_On_Multi-Label_Learning_Algorithms)

- Multilabel问题的实际应用。
  - [A Tutorial on Multilabel Learning](https://dl.acm.org/doi/pdf/10.1145/2716262)

- AI Lab开源的ML-Images。
  -[Tencent ML-Images: A Large-Scale Multi-Label Image Database for Visual Representation Learning](https://arxiv.org/pdf/1901.01703.pdf)

---  

## others

- Facebook作品，论文没有提出任何模型，或针对特定任务改进。论文任务公开数据集COCO/ImageNet/OpenIamge等对地理位置或收入水平低区域存在偏差，相应的图像数据分布较少。话说假如训练的模型可以识别全球目标的，在欠发达区域使用率也较少吧。
  - [Does Object Recognition Work for Everyone?](https://arxiv.org/pdf/1906.02659.pdf)
