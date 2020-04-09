# Graph CNN

---

## Table of Contents

- [survey](#survey)
- [awesome](#awesome)
- [classifier](#classifier)
- [ReID](#ReID)
- [tracking](#tracking)
- [Multi-Label](#Multi-Label)

---

## survey

- [Deep Learning on Graphs: A Survey](https://arxiv.org/pdf/1812.04202.pdf)

- 清华大学孙茂松教授组发表综述论文，全面阐述 GNN 及其方法和应用，并提出一个能表征各种不同 GNN 模型中传播步骤的统一表示。
  - [2019][Graph Neural Networks:A Review of Methods and Applications](https://arxiv.org/pdf/1812.08434.pdf)

- GNN在计算机视觉中的应用包括场景图生成、点云分类和动作识别。
  - [2019][A Comprehensive Survey on Graph Neural Networks](https://arxiv.org/pdf/1901.00596v1.pdf)

- ICLR2019论文。理论证明GCN的强大。
  - [2019][How powerful are graph neural networks](https://arxiv.org/pdf/1810.00826.pdf)
  
- GCN kernel介绍
  - [2019][Graph Kernels: A Survey](https://arxiv.org/pdf/1904.12218.pdf)


---

## awesome

- [naganandy/graph-based-deep-learning-literature](https://github.com/naganandy/graph-based-deep-learning-literature)

- 图卷积网络评测集。
  - [Open Graph Benchmark](https://ogb.stanford.edu/)

---

## classification

1、CVPR 2019论文，中山大学和加利福尼亚大学洛杉矶分校联合提出，主要提出Graph CNN代替CNN实现分类和分割等计算机视觉任务。论文用数学定义
Graph CNN是MLP，CNN，non-local network更抽象定义，在ImageNet-1k Classification， COCO Object Detection and Segmentation以及
CUHK03 Person Re-identification均有不俗战绩。论文的代码已经开源，但是在github只有已经训练模型的demo，没有更深一步的模型架构以及如何训练模型。

Adaptively Connected Neural Networks.[pdf](https://arxiv.org/pdf/1904.03579.pdf)

---

## ReID

Learning Context Graph for Person Search.[pdf](https://arxiv.org/pdf/1904.01830.pdf)

---

## tracking

Graph Convolutional Tracking

http://nlpr-web.ia.ac.cn/mmc/homepage/jygao/gct_cvpr2019.html#

---

## Multi-Label

1、CVPR2019论文，旷视和南京大学联合提出，主要解决多标签图像识别问题。由于图像的目标之间存在依赖关系，一般使用提取region（类似Faster RCNN的RPN）和LSTM
建立多标签之间上下文关系。论文Image representation learning由ResNet提取图像特征，GCN学习inter-dependent分类器，Correlation Matrix
生成多标签预测。另外论文模型延伸可用于图像检索。论文模型已开源，可以研究GCN的使用。

缺点：论文的相关系数矩阵A严重依赖于数据的分布，既GCN依赖于节点之间的关系，其实帽子在自然场景不一定和人搭配，也许戴在狗身上，个人认为这种多标签学习不一定适用自然场景。

Multi-Label Image Recognition with Graph Convolutional Networks.[(pdf)](https://arxiv.org/pdf/1904.03582.pdf)

## Annotation

1.多伦多大学与英伟达联合提出Curve-GCN，基于图卷积模型的一种高效交互式图像标注方法（需要人工使用多边形或矩形框框选模目标框，自动完成图像的像素分割）。
回忆一下faster RCNN,这不就是手工实现RPN的角色，再让Curve-GCN像素标注。如果让CNN实现目标检测，让Curve-GCN标注，这就是妥妥的实体分割模型。
另外训练Curve-GCN需要的标注数据集，也得人工完成图像的原始标注吧。先有鸡还是先有蛋。

Fast Interactive Object Annotation with Curve-GCN.[pdf](https://arxiv.org/pdf/1903.06874.pdf)

---

## Human Pose Regression

1、CVPR2019论文，罗格斯大学和宾汉姆顿大学联合提出，基于Graph CNN的3D姿态回归模型。使用2D Pose Estimation Network提取RGB Image keypoint，多层Feature和2D location
拼接，作为Semantic Graph Convolutional Network的输入，输出为3D Pose。论文提出的Semantic Graph Convolutions，融合CNN增加Graph CNN的感受野，NonLocal
模块保持Graph CNN的全局信息。

[semantic Graph Convolutional Networks for 3D Human Pose Regression](https://arxiv.org/pdf/1904.03345.pdf)

---

## Video Classification

- [I Know the Relationships: Zero-Shot Action Recognition via Two-Stream Graph Convolutional Networks and Knowledge Graphs]()

---

## point

- 阿卜杜拉国王科技大学(KAUST)提出DeepGCN系列，ICCV2019 oral论文，贯通GCN和CNN，并且实现对点云分割处理。CNN一般用于处理图像，视频音频，文本，而GCN处理社交网络，城市网络，点云等。GCN受限于梯度弥散问题，层数比较少（3-6 lays）
.论文延续CNN策略，基于DenseNet,ResNet等(skip connections)，让多个GCN拼接，同时使用空洞卷积等，增加感受视野。论文在3D数据集(S3DIS)效果,点云的语义分割好于pointNet++等模型，在生物数据集PPI也做相关验证.
  -[2019][ICCV][Can GCNs Go as Deep as CNNs?](https://arxiv.org/pdf/1904.03751.pdf)
  -[2019][DeepGCNs: Making GCNs Go as Deep as CNNs](https://arxiv.org/pdf/1910.06849.pdf)

---

## 待记录

Deep Learning on Graphs For Computer Vision — CNN, RNN, and GNN

<https://medium.com/@utorontomist/deep-learning-on-graphs-for-computer-vision-cnn-rnn-and-gnn-c114d6004678>
