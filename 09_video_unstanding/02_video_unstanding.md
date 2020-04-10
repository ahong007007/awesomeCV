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

- 天津大学关于无人机在计算机视觉领域的检测和跟踪进行全面的综述，包括当前面临的挑战，提出未来的发展和改进方向，提供一个大规模的无人机捕获数据集visDrone：
包含image object detection, video object detection, single object tracking, multi-object tracking四大类别数据集。
  - [Vision Meets Drones: Past, Present and Future](https://arxiv.org/pdf/2001.06303.pdf)

---

## video-recognition

- Fackbook出品。论文引入生物学中灵长类视网膜细胞启发，在视网膜节细胞中，80%是P-cell, 20%是M-cell，其中M-cell，
接受高帧率信息，负责响应运动变化，对空间和颜色信息不敏感。P-cell处理低帧率信息，负责精细的空间和颜色信息。对应论文两个分支：Slow pathway和
Fast pathway，分别处理低帧率图像空间语义信息和高帧率运动信息Slow pathway channels是Fast pathway 1/8,但是显著提高整个模型的准确率，Kinetics达到了79%的精度。
  - Slow pathway是Fast pathway 计算量20%，但是两个通道不是孤立的，各个特征分辨率均有特征融合。
  - 模型训练时使用128个GPU，视频理解领域还需要更简洁的特征表达能力。
  - [2019][ICCV][SlowFast Networks for Video Recognition](https://arxiv.org/pdf/1812.03982v3.pdf)

- Facebook出品，基于自家Fast-slow更新。基本思路在google EfficientNet延伸，在3D卷积中对各个系数进行调整：持续视觉，帧率，图像特征分辨率，宽度和深度。搜索空间要比EfficientNet更复杂。
  - [X3D: Expanding Architectures for Efficient Video Recognition](https://arxiv.org/pdf/2004.04730.pdf)

## Moving-Objects

- 运动相机检测运动目标，很有挑战性高。
  - [2019][Moving Objects Detection with a Moving Camera: A Comprehensive Review]

## ReID

- 北京理工大学等对ReID的综述文章。
  - [2019][Deep Learning for Person Re-identification:A Survey and Outlook](https://arxiv.org/pdf/2001.04193.pdf)

- 阿联酋IIAI研究院提出ReID模型。图像匹配和人脸识别，一般基于representation learning，泛化能力较弱。论文提出local matching， adaptive convolution kernels去和
匹配图像卷积（检索的feature map patch，和gallery feature map匹配）。另外提出一种假设，在一台摄像机附近的人仍然可能在另外一台摄像机附近（这种假设对一篇特征匹配应该用处不大）。

  - [2019][Interpretable and Generalizable Deep Image Matching with Adaptive Convolutions](https://arxiv.org/pdf/1904.10424.pdf)

## tricks

1.3D卷积，各种C3D, I3D,R(2+1)D,P3D,R(2+1)D
2.光流，从FlowNet到FlowNet2.0，flow of flow,
3.LSTM convLSTM
4.Graph CNN

Spatial temporal graph convolutional networks for skeletonbased action recognition AAAI2018.

Videos as Space-Time Region Graphs ECCV2018.

## dataset
