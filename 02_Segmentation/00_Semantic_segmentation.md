# Object Segmentation

## Table of Contents

- [survey](#survey)
- [Semantic_Segmentation](#Semantic_Segmentation)

## survey

survey/overview/review

- 图像分割综述
  - [2020][Image Segmentation Using Deep Learning:A Survey](https://arxiv.org/pdf/2001.05566.pdf)

- [DAVIS 2018](https://davischallenge.org/challenge2018/publications.html/ "DAVIS2018")

- [The 2018 DAVIS Challenge on Video Object Segmentation](https://arxiv.org/pdf/1803.00557.pdf)

- [PReMVOS: Proposal-generation, Refinement and Merging for Video Object Segmentation](https://arxiv.org/pdf/1807.09190.pdf)

- [Image Segmentation: Tips and Tricks from 39 Kaggle Competitions](https://neptune.ai/blog/image-segmentation-tips-and-tricks-from-kaggle-competitions?utm_source=reddit&utm_medium=post&utm_campaign=blog-image-segmentation-tips-and-tricks-from-kaggle-competitions)

- 图像分割loss综述，南京大学Jun Ma总结,研究loss不可多得的材料。
  - 扩展：分类，检测，分割，视频理解，3D点云都有各个方向的loss设计，综述一下，应该会有惊喜。
  - [Segmentation Loss Odyssey](https://arxiv.org/pdf/2005.13449v1.pdf)
  - <https://github.com/JunMa11/SegLoss>

- loss算法综述。
  - [A survey of loss functions for semantic segmentation](https://arxiv.org/pdf/2006.14822.pdf)
  - <https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions>

---

## Semantic_Segmentation

- 旷视提出的实时语义分割模型DFANet。旷视在移动终端接连发力，不仅仅是CV三大顶会，在手机的各种内置算法也是相当强悍。
一般语义分割模型只是二层级联（UNet变体）,论文在降低backbone分辨率基础上做三层级联，高低特征分辨率各种拼接，
fc attention的增加，充分实现不同分辨率下特征图的融合。实验效果相比ICNet以及ENet明显提升。

  - [DFANet: Deep Feature Aggregation for Real-Time Semantic Segmentation](https://share.weiyun.com/5NgHbWH)