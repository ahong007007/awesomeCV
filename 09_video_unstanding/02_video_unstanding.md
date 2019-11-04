# video understanding/video analysis
# survey
 - 澳大利亚麦觉理大学提出，行为识别论文综述,包括Vison Based和Sensor Based两个方向。
  - [Different Approaches for Human Activity Recognition– A Survey](https://arxiv.org/pdf/1906.05074.pdf)
## dataset
- 美图联合清华大学开源的视频分类数据集，规模较大且类别丰富。COIN 数据集采用分层结构，
即第一层是领域（Domain）、第二层是任务（Task）、第三层是步骤（Step），其中包含与
日常生活相关的 11827 个视频，涉及交通工具、电器维修、和家具装修等 12 个领域的 180
 个任务，共 778 个步骤。


  - [COIN: A Large-scale Dataset for Comprehensive Instructional Video Analysis](https://arxiv.org/pdf/1903.02874.pdf)

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


