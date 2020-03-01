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

---

## Semantic_Segmentation

- 旷视提出的实时语义分割模型DFANet。旷视在移动终端接连发力，不仅仅是CV三大顶会，在手机的各种内置算法也是相当强悍。
一般语义分割模型只是二层级联（UNet变体）,论文在降低backbone分辨率基础上做三层级联，高低特征分辨率各种拼接，
fc attention的增加，充分实现不同分辨率下特征图的融合。实验效果相比ICNet以及ENet明显提升。

  - [DFANet: Deep Feature Aggregation for Real-Time Semantic Segmentation](https://share.weiyun.com/5NgHbWH)