# video understanding

## 行为识别

1、CVPR2019论文，中科院自动化研究所模式识别实验室和中科大提出的Skeleton-based行为识别(Action Recognition)算法，基于
注意力机制增强的图卷积AGC-LSTM网络，高效提取和判别空间特征和时序特征，NTU RGB+D dataset 和 
Northwestern-UCLA dataset数据集state-of-art水平。

缺点：需要Skeleton提取或gt，本身人体关键点提取就是一个课题。

An Attention Enhanced Graph Convolutional LSTM Network for Skeleton-Based Action Recognition.[pdf](https://arxiv.org/pdf/1902.09130.pdf)


2、CVPR2019论文，海康威视提出,主要解决视频中时间和空间特征提取和融合问题，主要方法是分解3D卷积为3个正交的2D卷积，既C3D（3x3x3）分解为3个正交的2D卷积（1x3x3,3x1x3,3x3x1）
论文称为CoST模块（Collaborative SpatioTemporal），C3D计算量为3k^2，而CoST计算量是3k^2-3k+1，k=3时节省30%计算量。论文在Moments in Time Challenge 2018赢得第一名。

Collaborative Spatiotemporal Feature Learning for Video Action Recognition.[pdf](https://arxiv.org/pdf/1903.01197.pdf)