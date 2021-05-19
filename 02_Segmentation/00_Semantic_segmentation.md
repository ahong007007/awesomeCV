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

- DeepLab系列中的ASPP和DenseNet中的密集连接相结合，构成了DenseASPP。
  - feature和计算量会不会显著提升？类似于NAS-FPN稠密连接的架构思路。
  - [CVPR2018][DenseASPP for Semantic Segmentation in Street Scenes](https://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_DenseASPP_for_Semantic_CVPR_2018_paper.pdf)
  
- 旷视提出的实时语义分割模型DFANet。旷视在移动终端接连发力，不仅仅是CV三大顶会，在手机的各种内置算法也是相当强悍。
一般语义分割模型只是二层级联（UNet变体）,论文在降低backbone分辨率基础上做三层级联，高低特征分辨率各种拼接，
fc attention的增加，充分实现不同分辨率下特征图的融合。实验效果相比ICNet以及ENet明显提升。

  - [DFANet: Deep Feature Aggregation for Real-Time Semantic Segmentation](https://share.weiyun.com/5NgHbWH)
  
- 论文提出一个问题class confusion：较大的目标在较小尺度分割效果好，小目标在大尺度上分割效果好，而传统的multi-scale推断方法采用averaging or max pooling比较简单粗暴，没有考虑权重信息。
  论文提出multi-scale的attention机制，对图像分层处理：
  - 1.训练时用2尺度，推断时用3-4个尺度。2个尺度的图像特征金字塔，学习的attention参数直接在推断时使用，减少训练时对显卡的依赖。
  - 2.推断时用了2x,1x,0.5x甚至0.25x,attention参数直接复制，问题是多尺度的语义分割，特别是2x的图像，推理时间是不是显著增加？
  - [Hierarchical Multi-Scale Attention for Semantic Segmentation](https://arxiv.org/pdf/2005.10821v1.pdf)
  - <https://github.com/NVIDIA/semantic-segmentation>