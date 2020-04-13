# visual-tracking

---

## Table of Contents

- [survey](#survey)
- [Framework](#Framework)
- [tracking](#tracking)
- [Dataset](#Dataset)

## survey

- 意大利萨勒诺大学等对MOT综述

  - [Deep Learning in Video Multi-Object Tracking: A Survey](https://arxiv.org/pdf/1907.12740.pdf)[2019.07]

---

## Framework
  
  <https://github.com/STVIR/pysot>

- FairMOT
  - [A Simple Baseline for Multi-Object Tracking](https://arxiv.org/pdf/2004.01888.pdf)
  -<https://github.com/ifzhang/FairMOT>

---

## tracking

- SiamFC，ECCV2016论文，牛津大学Luca Bertinetto等提出，深度学习方法在目标跟踪领域的破冰之作。使用函数f(z,x)来比较模板图像z域候选图像x的相似度，
相似度越高，则得分越高。首个基于深度特征却又能保持实时性的跟踪方案，跟踪速度在GPU上达到了86fps（帧每秒），而且其性能超过了绝大多数实时跟踪器。
  -缺点：a.在真实世界存在多个目标干扰，遮挡，以及移位等因素，feature map会有多个相应，主要通过高斯窗滤波干扰目标。
  -b.训练出的网络主要关注外观特征而无视语义信息，容易造成背景干扰。
  - [Fully-Convolutional Siamese Networks for Object Tracking](https://arxiv.org/pdf/1606.09549.pdf)

- SiamRPN CVPR2018论文，商汤，北航和清华共同提出，实时性到160fps(backbone采用AlexNet)。论文提出模型包括两个子网络：Siamese subnetwork （特征提取）
和region proposal subnetwork（包括分类和检测分支）。换个角度，跟踪当做的单样本检测任务，就是把第一帧的BBox视为检测的样例，在其余帧里面检测与它相似的目标。

  - [High Performance Visual Tracking with Siamese Region Proposal Network](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_High_Performance_Visual_CVPR_2018_paper.pdf)

- CVPR2019论文，中科院自动化所（王强）和牛津大学提出SiamMask,目标跟踪和视频分割结合的多任务学习网络。VOT2015以后数据集难度不断增加，
目标预测从轴对齐矩形框到旋转框，其实是mask的一种近视。直接生成mask，可以获取更高精确度的旋转矩形框，这是论文的初衷。

  - 缺点：a.SiamMask的mask预测分支采用SharpMask语义分割模型，精度可使用替代模型提高。
  - b.目前tracking没有专门处理消失问题（object traker如果从当前画面离开或完全遮挡），特别的，siammask挺容易受到具有语义的distractor影响。
  - [Fast Online Object Tracking and Segmentation: A Unifying Approach](https://arxiv.org/pdf/1812.05050.pdf)

- CVPR2019论文，中国科学院大学和微软联合提出，主要解决Siamese网络架构一般使用较浅的网络架构（比如alexnet）。分析影响神经网络的三个因素
：padding,receptive field size和stride，并提出针对性改善的Cropping-Inside Residual (CIR)单元。模型在OTB和VOT等数据集取得
state-of-art，达到论文提出的改变神经网络deeper和wider的目标。

  - [Deeper and Wider Siamese Networks for Real-Time Visual Tracking](https://arxiv.org/pdf/1901.01660.pdf)

- CVPR2019论文，中科院自动化所和商汤联合提出，同样解决Siamese网络架构一般使用较浅的网络架构（比如alexnet）。
性能：论文提出的SiamRPN++在VOT2018取得性能最优的同时，速度在NVIDIA Titan Xp GPU 35fps,MobileNetv2保持性能的同时可运行70fps.
改进措施：
  - a.论文同样发现padding对特征提取有损伤,降低padding和stride的影响。（stride降低同时使用空洞卷积提高感受野）。
  - b.模型基于ResNet架构，并提出层级级联SiamRPN block用于协方差计算，多层次特征图预测目标的相似性。
  - c.提出Depthwise Cross Correlation (DW-XCorr)，大幅度降低参数和稳定模型训练。
  - 孪生网络和目标跟踪进入深度学习的深度网络时代。
  - [SiamRPN++: Evolution of Siamese Visual Tracking with Very Deep Networks](https://arxiv.org/pdf/1901.01660.pdf)

- 商汤，北航等联合提出，多目标跟踪（MOT）框架，可以学会充分利用长期和短期线索来处理MOT场景中的复杂情况。针对短期匹配，使用Siamese-RPN，
长期匹配和矫正，使用ReID。视频检测，分割，跟踪等均可以使用这种机制。

  - [Multi-Object Tracking with Multiple Cues and Switcher-Aware Classification](https://arxiv.org/pdf/1901.06129.pdf)

- 哈尔滨工业大学和华为联合提出STAIN(Siamese Attentional Keypoint Network)。一般视频追踪基于discriminative correlation filters和Siamese network， 
而论文提出在三个方面改进：backbone network, attentional mechanism 和detection component。backbone network基于hourglass network设计，cross-attentional
改进空间和时序注意力机制，检测模型基于华为提出的corner point和centroid point。
  - [Siamese Attentional Keypoint Network for High Performance Visual Tracking](https://arxiv.org/pdf/1904.10128.pdf)
  
- CVPR2019论文。
  - [SPM-Tracker: Series-Parallel Matching for Real-Time Visual Object Tracking](https://arxiv.org/pdf/1904.04452.pdf)[2019.04]  

- 北京理工大学，阿联酋阿布扎比联合提出，基于Teacher-Student模式提高tracking运行速率。
  - 疑问：Teacher-Student是一种训练方式还是一种模型设计方式？
  - [Teacher-Students Knowledge Distillation for Siamese Trackers](https://arxiv.org/pdf/1907.10586.pdf)

- 旷视，浙大联合提出，对SiamFC框架的改进。
  -[2019][SiamFC++: Towards Robust and Accurate Visual Tracking with Target Estimation Guidelines](https://arxiv.org/pdf/1911.06188.pdf)

- 德国亚琛工业大学，英国牛津大学大学联合提出，基于re-detection思路解决Tracking+video segmentation问题。
近两年在Siamese方向很多论文，re-detection作为传统方法思路不算剑走偏锋，但是论文横跨Tracking+video segmentation领域，同时在10个有影响力的数据集
做实验，佩服这份工作细致和耐心，在某些领域可能2-3个数据集就能发一篇水文。
  - 论文backbone基于Faster R-CNN，aligning proposals取代cross-correlation，Re-Detection Head包括from First Frame和from Previous Frame两部分。Tracklet Dynamic
Programming Algorithm相当于对之前帧特征的融合。Object Segmentation分支采用Box2Seg预测当前帧的分割。
  - Faster R-CNN在COCO数据集仅支持80分类，而论文Siam R-CNN声称支持任意目标跟踪。proposals从1000增加到10000可以增加召回，但是网络性能很慢(1fps)，
  采用利用previous-frame re-detections(up to 100)可以实现95.5%召回。previous-frame在VOS领域也是常用操作之一。
  -[2019][Siam R-CNN: Visual Tracking by Re-Detection](https://arxiv.org/pdf/1911.12836.pdf)

- 中科院提出的SiamMan，backbone基于Siam网络架构，多任务学习包括三个分支：分类，回归和定位。个人认为由于引入空洞卷积，多尺度特征，多尺度Attention等trick，特征表达能力强。
  -引入mask分支，或者previous-frame,准确率是不是更好。
  -网络需要足够的特征表达能力，实时性可能欠佳。workstation（Intel i7-7800X）, 8G memory, 2*RTX2080 GPUs 实现45fps。
  -[2019][SiamMan: Siamese Motion-aware Network for Visual Tracking](https://arxiv.org/pdf/1912.05515.pdf)

- [SiamRPN++: Evolution of Siamese Visual Tracking with Very Deep Networks](http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_SiamRPN_Evolution_of_Siamese_Visual_Tracking_With_Very_Deep_Networks_CVPR_2019_paper.pdf)

- SiamDW
  - [Deeper and Wider Siamese Networks for Real-Time Visual Tracking](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_Deeper_and_Wider_Siamese_Networks_for_Real-Time_Visual_Tracking_CVPR_2019_paper.pdf)

- [SiamFC++: Towards Robust and Accurate Visual Tracking with Target Estimation Guidelines](https://arxiv.org/pdf/1911.06188.pdf)

- 中科院自动化所等提出将检测和跟踪统一到一个框架。
  - 检测和跟踪区别：检测属于class-specific，和类别相关，类别内无区别。跟踪和类别无关，但是和同一个目标相关。检测不需要模板，而跟踪需要模板。
  - 论文提出target-guidance模块，引导检测器定位跟踪目标。
  - 提出anchored更新策略，避免模型的过拟合。
  - [Bridging the Gap Between Detection and Tracking: A Unified Approach](http://openaccess.thecvf.com/content_ICCV_2019/papers/Huang_Bridging_the_Gap_Between_Detection_and_Tracking_A_Unified_Approach_ICCV_2019_paper.pdf)

- [GradNet: Gradient-Guided Network for Visual Object Tracking]

- 中科院，微软提出，将tracking 和Instance Detection统一框架。
  - [Tracking by Instance Detection: A Meta-Learning Approach](https://arxiv.org/pdf/2004.00830.pdf)

- 计算机视觉已经不满足于图像的分类/检测/分割和视频理解的多任务学习，开始迈向点云。图像+视频+点云,4维度世界（三维度空间和时间维度），连接真实世界。

  - [Tracking Objects as Points](https://arxiv.org/pdf/2004.01177.pdf)
  - <https://github.com/xingyizhou/CenterTrack>

---

## Dataset

- CVPR2019 Tracking and Detection Challenge

  - [CVPR19 Tracking and Detection Challenge:How crowded can it get?](https://arxiv.org/pdf/1906.04567.pdf)
  
## 待更新

Martin大神新作，需要仔细研读

[Learning Discriminative Model Prediction for Tracking](https://arxiv.org/pdf/1904.07220v1.pdf)

4、Siamese Cascaded Region Proposal Networks for Real-Time Visual Tracking(CRPN,目标跟踪）
作者：Heng Fan, Haibin Ling
论文链接：https://arxiv.org/pdf/1812.06148.pdf
5、LaSOT: A High-quality Benchmark for Large-scale Single Object Tracking(目标跟踪）
作者：Heng Fan, Liting Lin, Fan Yang, Peng Chu, Ge Deng, Sijia Yu, Hexin Bai, Yong Xu, Chunyuan Liao, Haibin Ling
论文链接：https://arxiv.org/pdf/1809.07845.pdf
project链接：https://cis.temple.edu/lasot/
6、Leveraging Shape Completion for 3D Siamese Tracking
作者：Silvio Giancola, Jesus Zarzar, Bernard Ghanem
论文链接：https://arxiv.org/abs/1903.01784
7、Cross-Classification Clustering: An Efficient Multi-Object Tracking Technique for 3-D Instance Segmentation in Connectomics（多目标跟踪)
作者：Yaron Meirovitch, Lu Mi, Hayk Saribekyan, Alexander Matveev, David Rolnick, Casimir Wierzynski, Nir Shavit
论文链接：https://arxiv.org/abs/1812.01157
8、Multiview 2D/3D Rigid Registration via a Point-Of-Interest Network for Tracking and Triangulation (POINT^2)
作者：Haofu Liao, Wei-An Lin, Jiarui Zhang, Jingdan Zhang, Jiebo Luo, S. Kevin Zhou
论文链接：https://arxiv.org/abs/1903.03896

Graph Convolutional Tracking

<http://nlpr-web.ia.ac.cn/mmc/homepage/jygao/gct_cvpr2019.html#>

Learning Discriminative Model Prediction for Tracking.[pdf](https://128.84.21.199/pdf/1904.07220.pdf)

## trade off

illumination, deformation,occlusion and motion,speed