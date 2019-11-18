# survey/overview/review
---
# Classify

---
# video Segmentation

BubbleNets: Learning to Select the Guidance Frame in Video Object Segmentation by Deep Sorting Frames


- DAVIS2019 challenge，提出无监督multi-object track,数据集为DAVIS 2017基础上的修正。测试指标仍然没有time。

  - [The 2019 DAVIS Challenge on VOS:Unsupervised Multi-Object Segmentation](https://arxiv.org/pdf/1905.00737.pdf)

---
# Action Recognition
数据集：kinetics,HMDB
- CVPR2019论文，中科院自动化研究所模式识别实验室和中科大提出的Skeleton-based行为识别(Action Recognition)算法，基于
注意力机制增强的图卷积AGC-LSTM网络，高效提取和判别空间特征和时序特征，NTU RGB+D dataset 和 
Northwestern-UCLA dataset数据集state-of-art水平。

  缺点：需要Skeleton提取或gt，本身人体关键点提取就是一个课题。

  - [An Attention Enhanced Graph Convolutional LSTM Network for Skeleton-Based Action Recognition](https://arxiv.org/pdf/1902.09130.pdf)


- CVPR2019论文，海康威视提出,主要解决视频中时间和空间特征提取和融合问题，主要方法是分解3D卷积为3个正交的2D卷积，既C3D（3x3x3）分解为3个正交的2D卷积（1x3x3,3x1x3,3x3x1）
论文称为CoST模块（Collaborative SpatioTemporal），C3D计算量为3k^2，而CoST计算量是3k^2-3k+1，k=3时节省30%计算量。论文在Moments in Time Challenge 2018赢得第一名。

  - [Collaborative Spatiotemporal Feature Learning for Video Action Recognition](https://arxiv.org/pdf/1903.01197.pdf)

- CVPR2019论文，印第安那大学提出。受optial flow启发，为降低运算量，在低分辨率特征图计算可微分卷积"flow"表示层,进一步迭代实现flow of flow的运动信息表示方法。
论文同时准确率和实时性做对比试验（给出运算速率对比的都是好文章），865ms/帧，准确率和实时性取得平衡（I3D Two-Stream 9354ms）。虽然论文的某些理论是实验出来的
（After Block 3计算flow性能最好，以及Flow-Conv-Flow要优于Flow-Conv-Flow-Conv-Flow，并且没有给出合理解释），但是论文的实验过程值得学习。

  - [Representation Flow for Action Recognition](https://arxiv.org/pdf/1810.01455.pdf)

- CVPR2019论文。AAAI2018.
  - [Peeking into the Future: Predicting Future Person Activities and Locations in Videos]

- ICIP2019论文，奥卢大学和西安大学提出。NAS解决行为识别问题。搜索空间是视频的3D卷积，搜索算法是差分，评价数据集UCF101。感觉论文写的较为简单。DARTS简化NAS搜索空间，相信更多的视频理解论文会涌现。

  - [Video Action Recognition Via Neural Architecture Searching](https://arxiv.org/pdf/1907.04632.pdf)[2019.07]

- 
  - [ICCV][2019][EvaNet: The first evolved video architectures]()

- Google团队提出AssembleNet，主要解决视频行为识别问题，框架也可推广到其他视频分析领域。视频分析一般框架为3D CNN和Two stream。论文采用类似Two stream策略，自动化搜索时序特征和空间特征的multi-stream连接方式和神经网络架构。
randomly connecting思想来自于RandWire network，网络架构可看做有向无环图，实现任意stream之间的连接(high-level connectivity)，训练方法采用进化学习。
实验结果在Moments in Time和Charades数据集，均优于I3D，slowFast等视频分析框架。
 - 论文没有说明使用TPU数量和训练时间，依Google的个性，应该是大力出奇迹。
 - 强化学习/DARTS/one shot等在图像领域的成功case，可以在视频领域来一波。
 - [2019][AssembleNet: Searching for Multi-Stream Neural Connectivity in Video Architectures](https://arxiv.org/pdf/1905.13209.pdf)

- Google提出的Tiny Video Networks(TVN)，基于AutoMl进化学习算法，搜索空间包括时间和空间特征分辨率，卷积/池化，Non-local layer, context gating,Squeeze-excitation layer等，搜索空间包含10^13运算单元。约束目标包括运行时间和网络参数。
论文实验的baseline包括(2+1)D ResNets，I3D,S3D等，在Moments-in-time，HMDB,MLB-YouTube，Charades4个数据集验证，将GPU秒级的处理时间降低到10ms，CPU也只有37ms。
论文将TVN根据准确率和计算时间划分为TVN-1~TVN-4,同时基于scale策略(类似EfficientNet对各个参数的scale)，在准确率和实时性取得平衡。
 -  Google在AutoML不仅仅是半壁江山的地位了，在显卡集群和先发优势面前，很多小团队难以望其项背。
 - [Tiny Video Networks: The fastest video understanding networks](https://arxiv.org/pdf/1910.06961.pdf)
---

## ReID

-澳大利亚欧缇莫的大学

  - [Auto-ReID: Searching for a Part-aware ConvNet for Person Re-Identification](https://arxiv.org/pdf/1903.09776.pdf)[2019.03]



