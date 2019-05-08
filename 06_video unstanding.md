# video understanding/video analysis

## dataset
- 美图联合清华大学开源的视频分类数据集，规模较大且类别丰富。COIN 数据集采用分层结构，
即第一层是领域（Domain）、第二层是任务（Task）、第三层是步骤（Step），其中包含与
日常生活相关的 11827 个视频，涉及交通工具、电器维修、和家具装修等 12 个领域的 180
 个任务，共 778 个步骤。


  - [COIN: A Large-scale Dataset for Comprehensive Instructional Video Analysis](https://arxiv.org/pdf/1903.02874.pdf)



# video Segmentation

BubbleNets: Learning to Select the Guidance Frame in Video Object Segmentation by Deep Sorting Frames


- DAVIS2019 challenge，提出无监督multi-object track,数据集为DAVIS 2017基础上的修正。测试指标仍然没有time。

  - [The 2019 DAVIS Challenge on VOS:Unsupervised Multi-Object Segmentation](https://arxiv.org/pdf/1905.00737.pdf)


## Action Recognition
数据集：kinetics,HMDB
- CVPR2019论文，中科院自动化研究所模式识别实验室和中科大提出的Skeleton-based行为识别(Action Recognition)算法，基于
注意力机制增强的图卷积AGC-LSTM网络，高效提取和判别空间特征和时序特征，NTU RGB+D dataset 和 
Northwestern-UCLA dataset数据集state-of-art水平。

  缺点：需要Skeleton提取或gt，本身人体关键点提取就是一个课题。

  - [An Attention Enhanced Graph Convolutional LSTM Network for Skeleton-Based Action Recognition](https://arxiv.org/pdf/1902.09130.pdf)


- CVPR2019论文，海康威视提出,主要解决视频中时间和空间特征提取和融合问题，主要方法是分解3D卷积为3个正交的2D卷积，既C3D（3x3x3）分解为3个正交的2D卷积（1x3x3,3x1x3,3x3x1）
论文称为CoST模块（Collaborative SpatioTemporal），C3D计算量为3k^2，而CoST计算量是3k^2-3k+1，k=3时节省30%计算量。论文在Moments in Time Challenge 2018赢得第一名。

  - [Collaborative Spatiotemporal Feature Learning for Video Action Recognition](https://arxiv.org/pdf/1903.01197.pdf)

3、CVPR2019论文，印第安那大学提出。受optial flow启发，为降低运算量，在低分辨率特征图计算可微分卷积"flow"表示层,进一步迭代实现flow of flow的运动信息表示方法。
论文同时准确率和实时性做对比试验（给出运算速率对比的都是好文章），865ms/帧，准确率和实时性取得平衡（I3D Two-Stream 9354ms）。虽然论文的某些理论是实验出来的
（After Block 3计算flow性能最好，以及Flow-Conv-Flow要优于Flow-Conv-Flow-Conv-Flow，并且没有给出合理解释），但是论文的实验过程值得学习。

[Representation Flow for Action Recognition](https://arxiv.org/pdf/1810.01455.pdf)


4、CVPR2019论文。AAAI2018.
Peeking into the Future: Predicting Future Person Activities and Locations in Videos


# ReID

1、阿联酋IIAI研究院提出ReID模型。图像匹配和人脸识别，一般基于representation learning，泛化能力较弱。论文提出local matching， adaptive convolution kernels去和
匹配图像卷积（检索的feature map patch，和gallery feature map匹配）。另外提出一种假设，在一台摄像机附近的人仍然可能在另外一台摄像机附近（这种假设对一篇特征匹配应该用处不大）。

[Interpretable and Generalizable Deep Image Matching with Adaptive Convolutions](https://arxiv.org/pdf/1904.10424.pdf)

# tricks

1.3D卷积，各种C3D, I3D,R(2+1)D,P3D,R(2+1)D
2.光流，从FlowNet到FlowNet2.0，flow of flow,
3.LSTM convLSTM
4.Graph CNN



Spatial temporal graph convolutional networks for skeletonbased action recognition AAAI2018.

Videos as Space-Time Region Graphs ECCV2018.


