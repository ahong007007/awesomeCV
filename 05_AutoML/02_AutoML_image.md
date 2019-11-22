
# Detection

- 中科院自动化所和旷视联合提出，Object Detection with FPN on COCO优于ResNet101,但是FLOPs比ResNet50低。基于ShuffleNetV2的架构也有较好的表现。

  - [DetNAS: Neural Architecture Search on Object Detection](https://arxiv.org/pdf/1903.10979v1.pdf)[2019.03]

- Google基于AutoML提出Detection模型，基于RetinaNet网络，解决FPN多尺度金字塔问题。通过Neural Architecture Search搜索各种类型的
top-down,bottom-up特征层的连接方式（还是连连看），取得state-of-art的mAP同时降低推断时间。100 TPU的模型也不是可轻易实现。

  - [NAS-FPN: Learning Scalable Feature Pyramid Architecture for Object Detection](https://arxiv.org/pdf/1904.07392.pdf)[2019.04]
  
 - 西工大提出，基于one-stage object detector (FCOS)的基础，基本运算包括可分离卷积，空洞卷积，可变形卷积，搜索空间包括FPN，Prediction Head和Head Weight Sharing。
 论文第一句话说的好：The success of deep neural networks relies on significant architecture engineering，现在Deep learning就是在各种架构上作文章。
 论文的trick使用比NAS-FPN多，但是性能仅仅相比one-stage模型提升1%，相比two-stage还是有较大差距，说明搜索架构的backbone还很重要，关键有一个较优的先验知识。
 论文虽然在搜索时间上有优势（强化学习30GPU days可完成？存疑），性能上却没有优势。
 
  - [NAS-FCOS: Fast Neural Architecture Search for Object Detection](https://arxiv.org/pdf/1906.04423.pdf)[2019.06]

 - 华为诺亚方舟和中山大学联合提出Auto-FPN,主要针对目标检测的两个更新：Auto-fusion和Auto-head。Auto-fusion针对FPN的特征融合改进，既任意N层level feature特征融合,主要通过空洞卷积+rate,skip connection,depthwise-separable conv以及上采样/下样实现特征分辨率对齐和融合。Auto-head采用split-transform-merge策略，search space是input nodes和intermediate nodes。
  - Auto-FPN可以和Google的NAS-FPN对比阅读。NAS-FPN追求的是高准确率，在COCO-dev达到48.3mAP,Auto-FPN强调的是节省参数，相比SSD-ResNet101(31.2mAP)，Params节省12%。
  - 论文引用了FPN-NAS，但是没有做任何同一纬度的性能数据对比。
  - 论文实验数据集包括Pascal VOC, COCO, BDD, VisualGenome and ADE demonstrate，COCO具有说服力实验数据较少。
 
  - [2019][ICCV][Auto-FPN: Automatic Network Architecture Adaptation for Object Detection Beyond Classification]

- Google大脑出品,EfficientDet有两部分组成：backbone基于EfficientNet，BiFPN作为feature network，和class/box net layers共享参数，并且在不同分辨率特征重复多次。
论文没有说EfficientDet的TPU和训练参数，搜索空间，但是同志们，那是EfficientNet和BiFPN血的付出。
  - [EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/pdf/1911.09070.pdf)
---

# Recognition

- 华为等提出的人脸识别模型，在MS-Celeb-1M和LFW数据集state-of-art。主流人脸识别模型集中于度量学习（Metric Learning）和分类损失函数函数改进（Cross-Entropy Loss，Angular-Softmax Loss，
Additive Margin Softmax LossArcFace Loss等）。论文基于强化学习的NAS设计，network size 和latency作为reward(论文的实验没有对比测试latency或者模型尺寸)，仅说明最小网络参数NASC 16M。
这是NAS在人脸识别的首测尝试，分类，检测，识别都有涉及，图像分割应该不远。

  - [Neural Architecture Search for Deep Face Recognition](https://arxiv.org/pdf/1904.09523.pdf)[2019.04]

## Semantic  Segmentation

- CVPR2019论文，驭势科技，新加坡国立大学等联合提出轻量型分类和语义分割网络，成为"东风"网络，主要解决Speed/Accuracy trade-off问题，性能接近于deeplab v2但是速度>50 fps。
论文的backbone基于典型residual block，但是提出Acc(x)和Lat(x)用于评价准确率和推断时间，节省神经网络的随机搜索。另外没有选用RNN/LSTM,而是Random select an architecture。
论文提出的算法，训练200个模型架构时已经去除438个模型架构，每个模型训练5-7个小时(8卡机)。训练200个架构月400GPU/days。
模型在嵌入式硬件TX2和1080Ti均有评测，准确率和性能均有明显优势。去年有项目曾使用ENet/ICNet，速率尚可准确率不如人意，也许"东风"系列有明显改善。

[Partial Order Pruning: for Best Speed/Accuracy Trade-off in Neural Architecture Search](https://arxiv.org/pdf/1903.03777.pdf)[2019.03]

- 李飞飞团队作品。基于DeepLab系列框架搜索。

  - [Auto-DeepLab:Hierarchical Neural Architecture Search for Semantic Image Segmentation](https://arxiv.org/pdf/1901.02985.pdf)[2019.01]


## Graph CNN

- 2019ICLR论文，Uber等联合提出，基于Graph CNN实现的NAS,性能虽然没有太惊艳，但是基于Graph CNN应该有更广阔用处。

  - [Graph HyperNetworks for Neural Architecture Search](https://arxiv.org/pdf/1810.05749.pdf)[2018.10]


## Architecture

- facebook开源框架，基于MCTS和DNN,解决分类，目标检测，风格迁移，图像描述4个任务。

  - [AlphaX: eXploring Neural Architectures with Deep Neural Networks and Monte Carlo Tree Search](https://arxiv.org/pdf/1903.11059.pdf)[2019.03]

---
# Data Augmentation
- https://paperswithcode.com/task/data-augmentation

- 韩国kakaobrain作品。搜索空间包括autocontrast,cutout，把数据集分成K-fold，每个fold使用超参数（p是否使用增强的概率,λ数据增强的程度）并行训练，K-fold排序top-N策略组合。实验部分ResNet-200在Imagenet性能优于谷歌Augmentation,但是数据数据没有谷歌丰富，在目标检测数据集也有良好表现。

  - [Fast AutoAugment](https://arxiv.org/pdf/1905.00397.pdf)
  
- Goole大脑Zoph带队又一CVPR2019论文。论文主要针对图像分分类的数据增强操作，采用16种图像预处理方法：ShearX/Y,TranslateX/Y, Rotate, AutoContrast, Invert, Equalize, Solarize, Posterize, 
Contrast, Color, Brightness, Sharpness,Cutout, Sample Pairing，结合各种预处理的幅度和概率，生成2.9×10^32搜索空间。结合Google自家的RL NAS方式和不差钱的GPU群，硬生生的基于ResNet/AmoebaNet backbone
在ImageNet再攀高峰。训练细节已不在重要，也很难有GPU群复现，Google脑洞大开不怕浪费的做法，持续引领NAS领域。

  - [2019][CVPR][AutoAugment:Learning Augmentation Strategies from Data](https://zpascal.net/cvpr2019/Cubuk_AutoAugment_Learning_Augmentation_Strategies_From_Data_CVPR_2019_paper.pdf)
  - https://github.com/tensorflow/models/tree/master/research/autoaugment 
- Google大脑出品,Zoph带队,和CVPR2019一篇文章AutoAugment相同的idea，不过从ImageNet扩展到COCO，从全图的预处理方法扩展到检测框内图像的局部处理。论文提出的数据增强方式是训练过程常用的技巧：Color operations（Equalize, Contrast, Brightness），Geometric operations（e.g., Rotate,ShearX, TranslationY）
Bounding box operations（BBox Only Equalize,BBox Only Rotate, BBox Only FlipLR），硬生地设计(22×6×6)^2×5 ≈ 9.6×10^28的搜索空间(当然可以再增加)，延续NAS的设计思路（强化学习+RNN），
让神经网络选择数据增强的方式和过程。
    1、图像增强的方式没有什么亮点，但是9.6×10^28的搜索空间，想想都头大。
    2、不仅仅目标检测，其他分类，分割等计算机视觉任务都可以通过NAS-Data Augmentation训练模型？
    3、The RNN controller is trained over 20K augmentation policies. The search employed 400 TPU’s over 48 hours,土豪就是这么任性。
    4、Google最近很多论文都是基于NAS实现，NAS-FPN -> MobileNet v3-> EfficientNet -> NAS Data Augmentation，在EfficientNet时Google的调参就是满满的异类(initial learning rate 0.256 that decays by 0.97 every 2.4 epochs).
    Google不如一鼓作气让NAS给模型调参，真正实现AutoML,也能解放调参侠的工作量。
  - [Learning Data Augmentation Strategies for Object Detection](https://arxiv.org/pdf/1906.11172.pdf)[2019.06]
  - https://github.com/tensorflow/tpu/tree/master/models/official/detection

- Google大脑出品,依然Zoph带队，招数相同(从AutoAugment 16种缩减到14种)，但是不是让CNN学习怎么数据增广，而是随机选择数据增广的方式(纳尼，数据增广策略不都是随机的嘛)，python代码只有4行。相比Baseline有提高可以理解，
但是相比基于深度学习训练策略的AutoAugment，Fast AutoAugment，Population Based Augmentation还要好？(数据集为CIFAR-10,SVHN,ImageNet),只是在COCO数据集略输一筹。Google已经在NAS+Augment写了三篇论文，下一步该怎么玩？写一个Auto AutoAugment,说数据增强实验效果比之前随机数据增强方式更好。
 
  - [RandAugment: Practical data augmentation with no separate search](https://arxiv.org/pdf/1909.13719.pdf)
  - https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet

# Super-Resolution 

- 小米AI团队团队提出的超分辨率模型。

  - [Fast, Accurate and Lightweight Super-Resolution with Neural Architecture Search](https://arxiv.org/pdf/1901.07261.pdf)[2019.01]


  - [Architecture Search for Image Inpainting]
  
  