# video understanding

---

## Table of Contents

- [survey](#survey)
- [video-recognition](#video-recognition)
- [Moving-Objects](#Moving-Objects)
- [ReID](#ReID)
- [tricks](#tricks)
- [dataset](#dataset)

---

## survey

 survey/overview/Review

- [Deep Learning for Videos: A 2018 Guide to Action Recognition](http://blog.qure.ai/notes/deep-learning-for-videos-action-recognition-review)

- 天津大学关于无人机在计算机视觉领域的检测和跟踪进行全面的综述，包括当前面临的挑战，提出未来的发展和改进方向，提供一个大规模的无人机捕获数据集visDrone：
包含image object detection, video object detection, single object tracking, multi-object tracking四大类别数据集。
  - [Vision Meets Drones: Past, Present and Future](https://arxiv.org/pdf/2001.06303.pdf)

- 麻省理工学院(MIT)综述：包含问题定义，数据集，数据预处理，模型，评价尺度等。
  - [Video Action Understanding: A Tutorial](https://arxiv.org/pdf/2010.06647.pdf)
  
- <https://paperswithcode.com/task/action-classification>

- <https://github.com/open-mmlab/mmaction2>


---

## Framework

- <https://github.com/open-mmlab/mmaction>

- SenseTime X-Lab关于视频理解框架，包括SlowFast，R(2+1)D，R3D，TSN、TIN、TSM等。
  - <https://github.com/Sense-X/X-Temporal>

- Multi-Moments in Time Challenge 2019
  - <http://moments.csail.mit.edu/challenge_iccv_2019.html>

---

## video-Object-Detection

- 谷歌提出了一种目标检测的新方法Context R-CNN，简单地说，就是利用摄像头长时间的拍摄内容，推理出模糊画面里的目标。
  - 借助于视频关键帧(关键帧如何确定？),长时间的上下文信息(short attention,long attention，月度级计算量是不是很大？),检测目标。
  - 视频监控，野外无人值守，可以实现长时间的视频综合分析。如何借助多个camera，长时间定位那？
  - [Context R-CNN: Long Term Temporal Context for Per-Camera Object Detection](https://arxiv.org/pdf/1912.03538v3.pdf)

## video-recognition

- ICCV2019论文,MIT提出。传统2D卷积和3D卷积难以兼顾空间和时序信息，论文提出TSM模块偏移channel维度的特征，实现相邻帧之间的信息交换。
  - TSM的思想类似于二维卷积中的TCN.
  - 2D 空洞卷积是否也可引入？
  - [TSM: Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/pdf/1811.08383.pdf)
  - <https://github.com/mit-han-lab/temporal-shift-module>

- Fackbook出品。论文引入生物学中灵长类视网膜细胞启发，在视网膜节细胞中，80%是P-cell, 20%是M-cell，其中M-cell，
接受高帧率信息，负责响应运动变化，对空间和颜色信息不敏感。P-cell处理低帧率信息，负责精细的空间和颜色信息。对应论文两个分支：Slow pathway和
Fast pathway，分别处理低帧率图像空间语义信息和高帧率运动信息Slow pathway channels是Fast pathway 1/8,但是显著提高整个模型的准确率，Kinetics达到了79%的精度。
  - Slow pathway处理空间语义信息，Fast pathway捕获动作语义信息。
  - Slow path以较低的采样率来处理输入视频（2D卷积+3D卷积），提取随时间变化较慢的外观特征，为了提取鲁邦的外观特征，卷积核的空间通道数较大；
  - Fast path以较高的采样率来处理输入视频（3D卷积），提取随时间变化较快的运动特征，为了降低该通道的复杂度，卷积核的空间通道数较小；
  - Lateral connections: (fast->slow)两个path的特征进行融合，进行行为识别。
  - 每层的输出，Slow为{T,S^2,C}，而Fast为{αT,S^2,βC},Time-strided convolution将两者尺寸匹配。  
    Fast pathway ：higher temporal resolution and lower channel capacity。
  - Slow pathway是Fast pathway 计算量20%，但是两个通道不是孤立的，各个特征分辨率均有特征融合。
  - 模型训练时使用128个GPU，视频理解领域还需要更简洁的特征表达能力。
  - slowFast类比经典的特征金字塔，是不是说可以有不同的分支旁路？多个level的类似FPN特征进行融合？  
  - [2019][ICCV][SlowFast Networks for Video Recognition](https://arxiv.org/pdf/1812.03982v3.pdf)

- Facebook出品，基于自家Fast-slow更新。
  - 基本思路在google EfficientNet延伸，在3D卷积中对各个系数进行调整：持续视觉，帧率，图像特征分辨率，宽度和深度。搜索空间要比EfficientNet更复杂。
  - 设计 stepwise network expansion approach，每个step中，对各个维度单独扩张分别训练一个model，选择扩张效果最好的维度。大大减小搜索优化的复杂度。
  - 参考坐标下降法,每次对单个维度进行expand。  
  - 使用了channel-wise separable convolution，model非常小，block的width非常小.  
  - [X3D: Expanding Architectures for Efficient Video Recognition](https://arxiv.org/pdf/2004.04730.pdf)

- AAAI2020论文，清华+商汤+港中文联合实现，基于TMS,作者思考如何将时间信息嵌入到空间信息中，使得可以一次性联合学习两种信息。
  - [Temporal Interlacing Network](https://arxiv.org/pdf/2001.06499.pdf)

- Google提出适配移动端的视频处理方法。
  - 1.neural architecture search设计视频处理的backbone
  - 2.Stream Buffer计算，解耦存储空间，适用于处理任意长度的训练和推断视频序列。
  - 3.ensembling technique改善准确率同时提高效率。
  - [MoViNets: Mobile Video Networks for Efficient Video Recognition](https://arxiv.org/pdf/2103.11511v2.pdf)

- 字节跳动提出的一个视频特征提取方法，借鉴2D卷积的SEnet，提出Spatio-Temporal Excitation(时空特征)，Channel Excitation,Motion Excitation
  分别从三个角度融合时序特征和空间特征。
  - 如何实现网络结构：residual block的非skip分支中添加
  - 效果：something-v2效果一般，egoGesture和jester较为明显，Kinetics没有测试。而且只是比较了计算量，没有比较inference time。
  - [ACTION-Net: Multipath Excitation for Action Recognition](https://arxiv.org/pdf/2103.07372.pdf)
  - <https://arxiv.org/pdf/2103.07372.pdf>
  - <https://github.com/V-Sense/ACTION-Net>

## video-segment

- 微软提出的视频分割方法。
  - [A Transductive Approach for Video Object Segmentation](https://arxiv.org/pdf/2004.07193.pdf)
  - <https://github.com/microsoft/transductive-vos.pytorch>
  
- google提出的视频分割方法，主要借助于Teacher-student迭代生成伪标签，用于模型训练。图片的训练方法延伸到视频分割中，是否也可以作为行为识别，姿态估计等过程？
  - 数据集准备Labeled data和Unlabeled data。
  - 1.Labeled data训练Teacher network。
  - 2.Teacher network在未标注图像生成pseudo-labels。
  - 3.Student network在pseudo-labels数据集训练。
  - 4.Student network在Labeled data数据集fine-tune。
  - 5.把Student network当做Teacher network，重复步骤2的过程，直到指定的迭代次数。  
  - [2021][Naive-Student: Leveraging Semi-Supervised Learning in Video Sequences for Urban Scene Segmentation](https://arxiv.org/pdf/2005.10266v4.pdf)

## Moving-Objects

- 运动相机检测运动目标，很有挑战性高。
  - [2019][Moving Objects Detection with a Moving Camera: A Comprehensive Review]

## ReID

- 北京理工大学等对ReID的综述文章。
  - [2019][Deep Learning for Person Re-identification:A Survey and Outlook](https://arxiv.org/pdf/2001.04193.pdf)

- 阿联酋IIAI研究院提出ReID模型。图像匹配和人脸识别，一般基于representation learning，泛化能力较弱。论文提出local matching， adaptive convolution kernels去和
匹配图像卷积（检索的feature map patch，和gallery feature map匹配）。另外提出一种假设，在一台摄像机附近的人仍然可能在另外一台摄像机附近（这种假设对一篇特征匹配应该用处不大）。

  - [2019][Interpretable and Generalizable Deep Image Matching with Adaptive Convolutions](https://arxiv.org/pdf/1904.10424.pdf)

- 京东提出的FastReID框架，在Market1501&DukeMTMC&MSMT17数据集SOTA.
  -[FastReID: A Pytorch Toolbox for Real-world Person Re-identification](https://arxiv.org/pdf/2006.02631.pdf)

## Visual-Dialog

- [History for Visual Dialog: Do we really need it?](https://arxiv.org/pdf/2005.07493.pdf)

## Multimodal

- Facebook提出的视频与音频多模态视频识别模型，Audiovisual SlowFast
  - 延伸： Appearance, Motion, Audio, Text, Radar, Lidar, Touch,各种信息源可能对Multimodal学习都有效。
  - [Audiovisual SlowFast Networks for Video Recognition]<https://arxiv.org/pdf/2001.08740.pdf>
  
---

## training

- 视频处理不仅运行速度慢(模型参数量大),训练速度也慢。Ross Girshick，何凯明等提出使用Multigrid Method的方法提高模型的训练速度。
  - [A Multigrid Method for Efficiently Training Video Models](https://arxiv.org/pdf/1912.00998v2.pdf)

## tricks

1.3D卷积，各种C3D, I3D,R(2+1)D,P3D,R(2+1)D
2.光流，从FlowNet到FlowNet2.0，flow of flow,
3.LSTM convLSTM
4.Graph CNN

Spatial temporal graph convolutional networks for skeletonbased action recognition AAAI2018.

Videos as Space-Time Region Graphs ECCV2018.

## dataset
