# Detection

## Table of Contents

- [survey](#survey)
- [Facial_Detector](#Facial_Detector)
- [Object_Detection](#Object_Detection)
- [Transformer](#Transformer)
- [Tiny](#Tiny)
- [Imbalance](#Imbalance)
- [loss](#loss)
- [one_stage](one_stage)

---

## survey

- survey/overview/review

- state of art
  - <https://paperswithcode.com/sota/object-detection-on-coco>

- 香港中文大学，商汤等联合提出的MMDetection，包括检测模型，实体分割等state-of-art模型框架源码，属业界良心。

  - [2019.06][MMDetection: Open MMLab Detection Toolbox and Benchmark](https://arxiv.org/pdf/1906.07155.pdf)

- 西安电子科技大学关于目标检测的论文综述。
  - [2019.07][A Survey of Deep Learning-based Object Detection](https://arxiv.org/pdf/1907.09408.pdf)

- 新加坡管理大学论文综述。
  - [2019.08][Recent Advances in Deep Learning for Object Detection](https://arxiv.org/pdf/1908.03673.pdf)
  
- 中东科技大学(Middle East Technical University)一篇关于目标检测领域imbalance problem的综述。imbalance problem包括Class imbalance，
   Scale imbalance，Spatial imbalance， objective imbalance。论文对各个方面进行归纳，提出问题和分析解决方案。
  **话说imbalance中，头部问题是？**  
  - [2019.09][Imbalance Problems in Object Detection: A Review](https://arxiv.org/pdf/1909.00169.pdf)
  - [github](https://github.com/kemaloksuz/ObjectDetectionImbalance)

- 西弗吉尼亚大学提出一种评估目标检测理论准确率上限：91.6% on VOC (test2007), 78.2% on COCO (val2017), and 58.9% on OpenImages V4 (validation)。
  - [2019][Empirical Upper-bound in Object Detection and More](https://arxiv.org/pdf/1911.12451.pdf)

- 目标检测发展20年历程。
  - [Object Detection in 20 Years: A Survey](https://arxiv.org/pdf/1905.05055.pdf)

- TIDE,分析object detection和instance segmentation错误分布的工具。
  - 分析多个数据集，Pascal, COCO, Cityscapes, and LVIS等，错误原因包括Classification Error，Localization Error等6中类型，主要测试算法短板。
  - [TIDE: A General Toolbox for Identifying Object Detection Errors](https://arxiv.org/pdf/2008.08115.pdf)

---

## dataset

- OpenImages V5 dataset
  - 1.74M images, 14.6M bounding boxes, and 500 categories consisting of five different levels.

- COCO

- Object365

---

## Facial_Detector

- 天津大学、武汉大学、腾讯AI实验室提出的人脸检测模型，主要针对移动端设计（backbone MobileNet v2)在高通845上达到140fps的实时性。论文主要提出一个解决类别不均衡问题（侧脸、正脸、抬头、低头、表情、遮挡等各种类型）：增加困难类别和样本的损失函数权重。

  -缺点：但是MobileNet v2的检测框架应该没有这么快，并且论文的预测特征点和3D旋转角时，使用全连接网络，计算量大，应该更耗时才对。论文只给出特征点预测，但是一张图片N个人，如何区分特征点属于何人？论文没有告知如何计算检测框。
  - 论文应该很多细节没有讲述清晰，应该是idea分拆，写成连续剧的节奏。

  - [2019.02][PFLD:A Practical Facial Landmark Detector](https://arxiv.org/pdf/1902.10859.pdf)

- 伦敦帝国学院,InsightFace联合提出，单阶段的人脸检测模型，在WIDER FACE hard test set超过两阶段人脸检测模型。
  - 论文提出多目标任务学习方法，同时预测 face score, face box, five facial landmarks, and 3D position and correspondence。(face detection and alignment)
  - Dense Regression Branch:就是将2D的人脸映射到3D模型上，再将3D模型解码为2D图片，然后计算经过编解码的图片和原始图片的差别。中间用到了图卷积。
  - 训练阶段，用OHEM来平衡positive 和negative的anchors。
  - FPN之间添加Context Module，增强图像的感受野，用deformable convolution network (DCN) 代替3x3卷积，提高非刚性目标的上下文建模能力。
  - [RetinaFace: Single-stage Dense Face Localisation in the Wild](https://arxiv.org/pdf/1905.00641.pdf)

---

## Object_Detection

- 目标检测领域的奠基之作，王少青，何凯明，Ross Girshick以及孙剑等强强联手。
论文是RCNN系列的延续，主要包括三个部分：backbone ,Region Proposal Network以及head(包括RoI pooling)。主流两阶段框架也是在此基础上修正。
  - backbone:从最初的VGG，ResNet，以及Attention机制等，分类的backbone可直接使用替代。backbone是整个网络的共享的卷积层。
  - Region Proposal Network(RPN)：候选区域生成网络，主要包括一层3*3卷积，两个1*1卷积分支：分类（前景和背景）和坐标修正，对应损失函数分布是Softmax Loss和Smooth L1 Loss。RPN在共享卷积图滑动(40*60),生成9种anchor(三种比例1:1,1:2,2:1三种面积128×128，256×256，512×512，面积应该是原图尺寸，因为特征图只有40*60)。
  256-d表示特征channel数。#TODO感受野是多少？
  - ROIPooling : 每个ROI选取对应的特征，并归一下特征尺寸6*6。#TODO为甚是6*6？
  - ROIPooling:有两次取整操作，Region proposal的xywh取整，对xywh整数区域评价分成K*K单元，对每一单元边界取整。
  - head:包括NMS和损失函数SoftmaxLoss、SmoothL1Loss。

  - [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/pdf/1506.01497.pdf) :star::star::star::star::star:

- ICCV2017论文，微软亚洲研究院代季峰等提出DCN(Deformable Convolutional Network),将固定位置的卷积改造为适应物体形变的可变形卷积。
提出两个模块：deformable convolution 和deformable RoI pooling。所谓的deformable，是在原deformable convolution基础上增加可学习的offset(增加感受野范围)。同理RoI pooling计算上增加偏差实现deformable RoI pooling。论文提出的DCN俨然已经是目标检测领域刷分必备插件。

  - 1.针对任意形变的目标，offset都一样？
  - 2.x(p0+pn+Δpn)由于存在小数，通过双线性插值实现。G(q; p)·x(q)和双线性插值什么关系？
  - [2017.03][Deformable Convolutional Networks](https://arxiv.org/pdf/1703.06211.pdf)
  
- Deformable ConvNet V2（DCNv2),DCNv1升级版。论文任务DCNv1虽然可解决变形目标的检测，但是受irrelevant image content影响。论文提出三个改进措施：1、增加更多的Deformable Convolution，Conv3-Conv5的3x3 convolution都换成了Deformable Conv。2、让Deformable Conv不仅能学习offset，还能学习每个采样点的权重（modulation）3、模拟R-CNN的feature（knowledge distillation）。
  - Feature Mimicking如何实现？
  -[2018.12][Deformable ConvNets v2: More Deformable, Better Results](https://arxiv.org/pdf/1811.11168.pdf)
  
- CVPR2018论文，加利福尼亚大学圣迭戈分校提出。轮针对两点进行改进：样本减少易过拟合（模型增加，样本减少不过拟合？），使用不同IoU阈导致mismatch。
出发点是在目标检测分阶段提高IoU的阈值。
  - Faster R-CNN完成了对目标候选框的两次预测:RPN分类和回归、ROIpooling之后检测和回归。论文延续这一思想，设计三个阶段RCNN的检测和回归，并且逐阶段
  提升IoU阈值训练检测器。（论文证明三个阶段性能最好）。更新什么样的技术手段，可以再次提升多阶段的检测和回归？#TODO 可以有这样的尝试。
  - 检测输出：多个header输出的均值作为这个proposal最终的分数
  - 疑问：1.Cascade R-CNN是R-CNN的多层级联，损失函数也是级联？
  - 2.iterative bounding box和Cascade R-CNN框架相同，只是损失函数不同？
  - [2017.12][Cascade R-CNN: Delving into High Quality Object Detection](https://arxiv.org/pdf/1712.00726.pdf):star::star::star::star::star:
  
- Cascade R-CNN更新续篇。从网络结构看是增加Instance segmentation分支，检测(ResNeXt-101)45.8AP->(ResNeXt-152)50.2AP,38.6->42.3AP.
  - [2019.06][Cascade R-CNN: High Quality Object Detection and Instance Segmentation](https://arxiv.org/pdf/1906.09756.pdf)

- 国防科技大学和旷视科技联合提出，典型的RPN+FPN架构，backbone基于SNet,增加Context Enhancement
Module(FPN多尺度分辨率特征融合)和spatial attention module（RPN->1x1卷积实现空间注意力模型），
实验结果相对于MobileNetV2-SSDLite速度和精度均有提高。

  - [2019.03][ThunderNet: Towards Real-time Generic Object Detection](https://arxiv.org/pdf/1903.11752.pdf)

- CVPR2019论文、商汤，浙江大学等联合提出的Libra R-CNN。motivation来自于作者认为的三个不平衡：数据不平衡，特征不平衡，
损失函数不平衡。数据不平衡采用：N总样本根据IoU分成K个子样本,增加困难样本的采样概率。特征不平衡采用：ResNet Identity 和
non-local模块修正语义特征。损失函数不平衡：论文设计Balanced L1 Loss（**待验证和理解**）。
  论文提出的三个不平衡，可以认为是3个trick，可以集成到其他模型，改进检测的精度。
  - [Libra R-CNN: Towards Balanced Learning for Object Detection](https://arxiv.org/pdf/1904.02701.pdf)

- 中国科学院大学等提出Trident Networks，既模型backbone包含的tricks:Multi-branch Block,Weight sharing among branches,Scale-aware Training Scheme(不同尺度目标位于不同分支)，模型最终集万千tricks于一身，基于ResNet-101-Deformable，在COCO test-dev set取得state-of-art，48.4 mAP。

  - [2019][Scale-Aware Trident Networks for Object Detection](https://arxiv.org/pdf/1901.01892v1.pdf)
  
- 微软等提出GCNet(global context network),基于Non-Local Network（NLNet）和Squeeze-Excitation Network (SENet)集成版。论文验证对于Non-Local Network，不同的query points可得到一致的attention maps（为什么Non-Local是不一致的attention map?），因此可以简化NL模块为Context Modeling，结合Transform既为Global context (GC) block。个人认为Global context (GC) block是空间注意力机制的延伸。论文模型在集成X101+DCN++Cascade+GC r4，在COCO test-dev set取得state-of-art，48.4 mAP，再次证明集成的伟大。

  - [GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond](https://arxiv.org/pdf/1904.11492v1.pdf)[2019.04]

- 北京大学等提出的一种改善型backbone，类似于HRNet和Cascade R-CNN（Cascade R-CNN是级联detector,而本文CBNet是级联backbone）。论文最强指标Cascade Mask R-CNN +Triple-ResNeXt152在COCO数据集实现53.3AP，性能上是数据榜首。  
  -[2019.09][CBNet: A Novel Composite Backbone Network Architecture for Object Detection](https://arxiv.org/pdf/1909.03625.pdf) :star::star::star::star:

- google在EfficientNet基础上开发的EfficientDet。
  - backbone:基于EfficientNet.
  - BiFPN: 两个特征金字塔叠加，同时融合了任意两层的feature，有尺寸不一致，会upsample/pooling保证相同的分辨力.
  - efficientnet也好，efficientdet也好，都是参数驱动型的网络，参数是贯穿每一层的。
  - [EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/pdf/1911.09070.pdf)
  - <https://zhuanlan.zhihu.com/p/129016081>
  - <https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch>

- 商汤香港中文大学联合实验室提出，帮助港中文商汤联合实验室取得OpenImage Object Detection Challenge 2019 冠军。
  - 提出了基于任务间空间自适应解耦（task-aware spatial disentanglement，TSD）的检测算法，对于分类任务和回归任务，分别让其学习各自适应的proposal和特征提取器。
  - soft-NMS algorithm 更新到adj-NMS：NMS阈值0.5，先过滤一部分检测框，然后在更新NMS阈值权重。
  - 引入progressive constraint（PC）损失，来帮助检测器性能大幅度超越传统的检测器头部。
  - trick包括deformable convnet，multi-scale testing，averaging the parameters of epoch。
  - [CVPR2020][1st Place Solutions for OpenImage2019 - Object Detection and Instance Segmentation](https://arxiv.org/pdf/2003.07557.pdf)

- Google出品，基于Google自家作品Data Augmentation，pre-training,Self-training(Noisy Student training),排列组合，宣布输出SpineNet。
  - [Rethinking Pre-training and Self-training](https://arxiv.org/pdf/2006.06882v1.pdf)

- 霍普金斯大学&谷歌提出，backbone使用递归调用金字塔，以及可切换的空洞卷积（SAC，Switchable Atrous Convolution），实现在检测，实例分割，全景分割的STOA。
  - DetectoRS = Detector + RFP + SAC = Detector + Recursive Feature Pyramid + Switchable Atrous Convolution
  - Recursive Feature Pyramid=top-down+bottom-up的双路特征金字塔。
  - 论文提出可切换的空洞卷积，是不是也可以做可切换的DCN？
  - [DetectoRS: Detecting Objects with Recursive Feature Pyramid and Switchable Atrous Convolution](https://arxiv.org/pdf/2006.02334)

- Google Brain团队Quoc V. Le等大佬出品，55.1 COCO test-dev,目标检测领域的STOA，论文主要是对FPN的改进。
  - EfficientNet比较好理解，用NAS搜索出backbone的最佳resolution/depth/width。
  - 基于EfficientNet基础上，对FPN改进成为BiFPN，Cross-Scale Connections和Weighted Feature Fusion。
  - Cross-Scale Connections：吸收FPN,PANet，NAS-FPN的经验，做三点改进：移除只有一个输入的节点，添加同一level的原始特征融合，重复三次top-down &bottom-up，加强更高特征的融合。
  - Weighted Feature Fusion:一般特征融合方式是直接相加或拼接。论文提出给各个feature增加权重系数，改善特征的重要性。
  - 论文的模型到处是各种因子，128个TPU调参，炼丹味道太浓。不过开源的模型可以直接拿来主义。
  - [ECCV2020][EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/pdf/1911.09070.pdf)

---

## Transformer

- Facebook出品，舍弃了目标检测领域的two-stage one_stage方法，直接用NLP领域的Transformer替代anchor方法，预测目标。
  - 论文没有使用trick，对比经典的Faster-RCNN算法，没有anchor也不用nms。
  - 论文还指出进一步研究的方向：Transformer的特点对大目标检测很好但是对小目标检测不好；由于N的限制，每次只能检测100个；GFLOPs持平，但是速度降低一倍。
  - 开山之作，会带一波检测领域的论文。
  - [End-to-End Object Detection with Transformers](https://arxiv.org/pdf/2005.12872.pdf)

---

## Tiny

- 自滑铁卢大学提出YOLO Nano 的网络，他们通过人与机器协同设计模型架构大大提升了性能。YOLO Nano 大小只有 4.0MB 左右，比 Tiny YOLOv2 和 Tiny YOLOv3 分别小了 15.1 倍和 8.3 倍，性能却有较为显著的提升。

  -[2019.10][YOLO Nano: a Highly Compact You Only Look Once Convolutional Neural Network for Object Detection](https://arxiv.org/pdf/1910.01271.pdf)

- 腾讯，香港科技大学联合提出，One-stage目标检测框架RefineDetLite，解决在CPU场景下实时性问题。backbone基于Res2Net，anchor refinement module (ARM)提出粗略anchor，object detection module (ODM) 融合
特征金字塔特征修订anchor，生成最终的分类信息和检测框。论文的训练策略包括IOU-guided loss对负样本抑制, classes-aware weighting 解决样本类别不平衡和balanced multitask training多任务训练，最终实现
RefineDetLite++在MSCOCO数据集29.6AP&131ms。
  - 论文提出的模型，虽然一直倡导非GPU而是CPU，实际在Intel i7-6700@3.40GHz测试，貌似和mobile实际运行有差距。 
  - 论文提出的RefineDetLite，包含coarse loss module和refined loss module，仅仅是没有ROI pooling，个人感觉和two-stage的目标检测框架没太多区别。
  - [2019][RefineDetLite: A Lightweight One-stage Object Detection Framework for CPU-only Devices](https://arxiv.org/pdf/1911.08855.pdf) 

---

## Imbalance

- 中国科学技术大学提出，在Focal Loss基础上解决目标检测前景和背景不平衡问题。论文主要提出三个观点：decoupling objectness from classification, biased initialization, threshold movement。
不过从数据对比，性能提升不明显。
  - [2019][Revisiting Foreground-Background Imbalance in Object Detectors](https://arxiv.org/pdf/1909.04868.pdf)

---

## loss

- [Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression](https://arxiv.org/pdf/1902.09630.pdf)[2019.02]

- AAAI2020论文，天津大学，中国人民公安大学等提出。IoU预测目标框是目标检测领域的重要组成。常规IoU loss计算方法,仅用于候选框和gt有交叠区域，无交叠时没有梯度生成。GIoU loss惩罚项，但是gt和候选框存在包关系时直接退化为IoU。论文在两个loss基础上引入惩罚项，提出Distance-IoU(DIoU) Loss和Complete IoU(CIoU) Loss用于目标检测BBox进一步回归，收敛速度快，准确率高，容易集成于NMS。
  - DIoU Loss可以直接最小化两个目标框的距离，在包含框在水平或垂直方向时，回归loss下降更快。
  - Complete IoU(CIoU),包含交叠像素面积,中心点距离，长宽比，可以更好描述检测框的位置关系。
  - NMS阶段使用DIoU作为评价尺度，同时考虑IoU和中心点距离，进一步提高目标检测性能。
  - 论文提出的trick，在PASAC2007和COCO2017均明显涨点。在state-of-art模型比如Cascade R-CNN等，是否有进一步良好表现？
  - [2019][Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression](https://arxiv.org/pdf/1911.08287.pdf) :star::star::star::star::star:

---

## one_stage

- 商汤和香港中文大学联合提出，ICLR2019论文，实在没看懂啥意思。

  - [Feature Intertwiner for Object Detection](https://arxiv.org/pdf/1903.11851.pdf)[2019.03]
  
- 阿德莱德大学沈春华项目组提出的目标检测方向新论文FCOS,去除传统目标检测的FPN操作,添加Center-ness分支，直接anchor free预测
(l; t; r; b) 四维向量（节省anchor相关的超参数设定以及计算），依赖于NMS，直接生成目标检测框。论文提出的FCOS希望可以应用于
后续的语义分割，姿态估计等领域。论文的性能在one-stage领域state-of-art，节省大量FPN的计算，但是没有任何关于速度的指标，比较遗憾。

  - [FCOS: Fully Convolutional One-Stage Object Detection](https://arxiv.org/pdf/1904.01355.pdf)[2019.04]

- 清华大学提出HAR-Net，在single-stage 目标检测框架FPN-Retina-Net基础上改进，混合实现attention mechanism，包括spatial attention, channel attention和aligned attention。spatial attention通过堆叠空洞卷积实现增加感受视野，channel attention通过squeeze-excitation (SE) block实现，aligned attention通过deformable convolution实现。论文思路清晰，实现单阶段目标检测模型性能的state-of-art(45.8% mAP COCO)，并超越two stage Cascade RCNN。论文没有实时性的数据。

  - [HAR-Net: Joint Learning of Hybrid Attention for Single-stage Object Detection](https://arxiv.org/pdf/1904.11141.pdf)[2019.04]

- 中国科学院大学,牛津大学和华为联合提出one-stage模型，MS-COCO dataset数据集测试，达到47mAP（逼近two-stage PANet准确率）,超越所有one-stage模型。论文模型backbone
基于Hourglass，提出Center pooling，Corner pooling，Cascade corner pooling（论文的triplet）确定目标的边界。
NVIDIA Tesla P100 GPU运行，CenterNet511-104 340ms/image，比CornerNet511-104 300ms略慢。

  - [CenterNet: Object Detection with Keypoint Triplets](https://arxiv.org/pdf/1904.08189.pdf)[2019.04]

- 普林斯顿大学提出CornerNet改进版的目标检测模型，CornerNet-Saccade比CornerNet提速6倍且性能提升1% AP,CornerNet-Squeeze比YOLOv3更快也更准确。
论文首先把图片Downsizing，经过hourglass网络与Attention maps生成候选区域并裁剪，对每个裁剪区域再经过hourglass生成目标检测框。最后对所有检测框合并。
论文使用缩放代替CNN的降采样，较少运算量，但是缩小图像的分辨率对小目标的检测准确率应该有影响吧。个人感觉比CenterNet实用性更好：要么选择准确率高的two-stage,
要么选择速度和准确率平衡的轻量化网络。感叹目标检测领域的飞速发展。CV行业在分类和检测日新月异。

  - [2019.04][CornerNet-Lite: Efficient Keypoint Based Object Detection](https://arxiv.org/pdf/1904.08900.pdf)
  - [github](https://github.com/princeton-vl/CornerNet-Lite)
  
- 中科院自动化所模式识别实验室提出目标检测领域anchor based and anchor-free本质不同：训练模型时正负样本采样策略，相同正负样本情况下可以达到同样的性能；根据检测目标的静态特性，提出ATSS自适应训练样本选择策略(adaptive training sample selection)。
在ASTT改进下， anchor based and anchor-free性能均有提升，且达到state-of-art 50.7%AP。
  - 目标检测一般划分为anchorbased and anchor-free。anchorbased可划分为one stage 和two stage；anchor-free可细分为keypoint-based和center-based。
  - ATSS过程很容易理解，替代直接使用IOU作为候选框的阈值，首先选择ground-truth中心最近的k个候选框，计算均值和方差作为IOU阈值，去除中心点在检测目标之外的候选框。整个过程中只有一个超参数k，且对模型影响小。
  - 论文没有多网络模型任何改进，仅仅是训练时样本的改进，大道至简，返璞归真。
  - [2019][Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection](https://arxiv.org/pdf/1912.02424v1.pdf) :star::star::star::star::star:
  - <https://github.com/sfzhang15/ATSS>

- 西安交通大学，商汤科技能联合提出，实现anchor free的目标检测和实例分割，在cornerNet/CenterNet基础上改进.backbone基于Hourglass Network,两个分支：预测检测框的左上角，右下角角点的同时，预测centripetal Shift；另外一个分支实现Instance Mask.
  - 目标重叠度较高的情况下，centripetal Shift可以实现左上角和右下角的配对？
  - COCO test-dev 检测48AP ,实体分割40.2AP。
  - #TODO 物体的角点没有任何特征，为什么不直接预测目标？
  - [2020][CentripetalNet: Pursuing High-quality Keypoint Pairs for Object Detection](https://arxiv.org/pdf/2003.09119.pdf)
  - <https://github.com/KiveeDong/CentripetalNet>

- 马里兰大学帕克分校提出的快速目标检测方法，backbone基础上包括Corner Attentive Module，Center Attentive Module，Attention Transitive Module，
在三个attention模块基础上Aggregation Attentive Module生成最终的检测框。
  - [SaccadeNet: A Fast and Accurate Object Detector](https://arxiv.org/pdf/2003.12125.pdf)

- Alexey Bochkovskiy等作品，计算机视觉领域最好的论文之一。不仅仅是一篇论文，更是一篇综述，把CNN的各种trick总结一遍，并且给出设计原则和依据。实验数据翔实，有理有据。
  - 集成学习，WRC, CSP,CmBN, SAT, Mish activation, Mosaic data augmentation, DropBlock regularization, and CIoU loss，各种trick集成一身，设计臃肿，但是速度飞快，性能高超。
  - 按照作者的思路，也可以在two-stage阶段写一篇佳作。
  - [YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/pdf/2004.10934.pdf)

- Baidu推出的PP-YOLO，集成万千trick一身，速度和准确率都超过EfficientDet and YOLOv4.
  - PP-YOLO的速度快，不见得是网络架构做的好，百度在paddlepaddle做了很多改进。比如百度在ResNet50基础上的开发ResNet50-vd，在底层实现的加速优化。
  - [PP-YOLO: An Effective and Efficient Implementation of Object Detector](https://arxiv.org/pdf/2007.12099v3.pdf)

---

## NMS_Series

2017----Soft-NMS----Improving Object Detection With One Line of Code

2018----Softer-NMS- Rethinking Bounding Box Regression for Accurate Object Detection

2018----IoUNet----Acquisition of Localization Confidence for Accurate Object Detection

2018----CVPR----Improving Object Localization with Fitness NMS and Bounded IoU Loss

2018----NIPS----Sequential Context Encoding for Duplicate Removal

M2Det: A Single-Shot Object Detector based on Multi-Level Feature Pyramid Network [PDF](https://arxiv.org/pdf/1811.04533.pdf) [Github](https://github.com/qijiezhao/M2Det)

Grid R-CNN [PDF](https://arxiv.org/pdf/1811.12030.pdf)

---

## framwork

- detectron2继承Detectron和maskrcnn-benchmark，提供丰富的目标检测模型。
  - [detectron2](https://github.com/facebookresearch/detectron2)
[simpledet](https://github.com/tusimple/simpledet)

mmdetection
maskrcnn-benchmark

---

## tricks

1.backbone与特征提取

目标检测的backbone一般基于ImageNet预训练的图像分类网络。图像分类问题只关注分类和感受视野，不用关注物体定位，
但是目标检测领域同时很关注空间信息。如果下采样过多，会导致最后的feature map很小，小目标很容易漏掉。很多基础架构网络，
比如ResNet、Xception、DenseNet、FPN、DetNet、R-CNN，PANet、等神经网络提取图像的上下文信息，不断在特征提取方向优化。

2.基于Anchor生成的算法

比如Sliding window、Region Proposal Network (RPN) 、CornerNet、meta-anchor等。

3.IoU计算

UnitBox，IoU-Net，GIoU

4.损失函数

包括L1和L2，Focal loss，增加困难类别和样本的损失函数权重

5.优化NMS

包括Soft-NMS,Softer-NMS,以及Relation Netwrok，ConvNMS，NMS Network，Yes-Net等。

6.通用tricks

类别不平衡（增加样本）,数据增强

## 待记录

Adaptive NMS: Refining Pedestrian Detection in a Crowd.[pdf](https://arxiv.org/pdf/1904.03629.pdf)

CVPR2019
[All You Need is a Few Shifts: Designing Efficient Convolutional Neural Networks for Image Classification](https://arxiv.org/pdf/1903.05285.pdf)