
#video recognition

 
# ReID

- 北京理工大学等对ReID的综述文章。 
  - [2019][Deep Learning for Person Re-identification:A Survey and Outlook](https://arxiv.org/pdf/2001.04193.pdf)

- 阿联酋IIAI研究院提出ReID模型。图像匹配和人脸识别，一般基于representation learning，泛化能力较弱。论文提出local matching， adaptive convolution kernels去和
匹配图像卷积（检索的feature map patch，和gallery feature map匹配）。另外提出一种假设，在一台摄像机附近的人仍然可能在另外一台摄像机附近（这种假设对一篇特征匹配应该用处不大）。

  - [2019][Interpretable and Generalizable Deep Image Matching with Adaptive Convolutions](https://arxiv.org/pdf/1904.10424.pdf)

# tricks

1.3D卷积，各种C3D, I3D,R(2+1)D,P3D,R(2+1)D
2.光流，从FlowNet到FlowNet2.0，flow of flow,
3.LSTM convLSTM
4.Graph CNN


Spatial temporal graph convolutional networks for skeletonbased action recognition AAAI2018.

Videos as Space-Time Region Graphs ECCV2018.


