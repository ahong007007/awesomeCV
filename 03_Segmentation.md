
# Video Object Segmentation

## [DAVIS 2018](https://davischallenge.org/challenge2018/publications.html/ "DAVIS2018")

  - [The 2018 DAVIS Challenge on Video Object Segmentation](https://arxiv.org/pdf/1803.00557.pdf)

  - [PReMVOS: Proposal-generation, Refinement and Merging for Video Object Segmentation](https://arxiv.org/pdf/1807.09190.pdf)



# Semantic Segmentation
- 旷视提出的实时语义分割模型DFANet。旷视在移动终端接连发力，不仅仅是CV三大顶会，在手机的各种内置算法也是相当强悍。
一般语义分割模型只是二层级联（UNet变体）,论文在降低backbone分辨率基础上做三层级联，高低特征分辨率各种拼接，
fc attention的增加，充分实现不同分辨率下特征图的融合。实验效果相比ICNet以及ENet明显提升。

  - [DFANet: Deep Feature Aggregation for Real-Time Semantic Segmentation](https://share.weiyun.com/5NgHbWH)


# Instance Segmentation

- Ross Girshick，何凯明等人提出TensorMask，解决密集滑动窗口的目标实体分割。从论文的图2的效果看，TensorMask
可能不如Mask R-CNN，也许作者挑选图错误或图的说明错误。

  - [TensorMask: A Foundation for Dense Object Segmentation](https://arxiv.org/pdf/1903.12174.pdf)

- CVPR2019论文，华中科技大学和地平线联合提出。Motivation来自于Mask RCNN的有classification和classification score，但是Mask
没有score，导致的mask quality不匹配（引入Mask IoU计算，避免detection IoU相同而mask无法优化的问题）。论文在Mask RCNN基础上增加
MaskIoU Head，代码也是facebook 开源框架maskrcnn_benchmark基础上直接修订，简单有效。论文的Ablation study实验证明：a.MaskIoU的框架最有效方式，
b.target category训练方式。个人感觉MaskIoU和score不是线性相关，应该还有很多坑可以填。

  - [Mask Scoring R-CNN](https://arxiv.org/pdf/1903.00241.pdf)


