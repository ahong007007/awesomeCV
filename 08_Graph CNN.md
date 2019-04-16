

# overview

Deep Learning on Graphs: A Survey.[pdf](https://arxiv.org/pdf/1812.04202.pdf)

Graph Neural Networks:A Review of Methods and Applications.[pdf](https://arxiv.org/pdf/1812.08434.pdf)
# Video Classification
I Know the Relationships: Zero-Shot Action Recognition via Two-Stream Graph Convolutional Networks and Knowledge Graphs
.[pdf]()

# Classification

1、CVPR 2019论文，中山大学和加利福尼亚大学洛杉矶分校联合提出，主要提出Graph CNN代替CNN实现分类和分割等计算机视觉任务。论文用数学定义
Graph CNN是MLP，CNN，non-local network更抽象定义，在ImageNet-1k Classification， COCO Object Detection and Segmentation以及
CUHK03 Person Re-identification均有不俗战绩。论文的代码已经开源，但是在github只有已经训练模型的demo，没有更深一步的模型架构以及如何训练模型。

Adaptively Connected Neural Networks.[pdf](https://arxiv.org/pdf/1904.03579.pdf)

# ReID

Learning Context Graph for Person Search.[pdf](https://arxiv.org/pdf/1904.01830.pdf)
# tracking

Graph Convolutional Tracking

http://nlpr-web.ia.ac.cn/mmc/homepage/jygao/gct_cvpr2019.html#

# Multi-Label

1、CVPR2019论文，旷视和南京大学联合提出，主要解决多标签图像识别问题。由于图像的目标之间存在依赖关系，一般使用提取region（类似Faster RCNN的RPN）和LSTM
建立多标签之间上下文关系。论文Image representation learning由ResNet提取图像特征，GCN学习inter-dependent分类器，Correlation Matrix
生成多标签预测。另外论文模型延伸可用于图像检索。论文模型已开源，可以研究GCN的使用。

缺点：GCN依赖于节点之间的关系，其实帽子在自然场景不一定和人搭配，也许戴在狗身上，个人认为这种多标签学习不一定适用自然场景。

Multi-Label Image Recognition with Graph Convolutional Networks.[(pdf)](https://arxiv.org/pdf/1904.03582.pdf)

# 待记录

Can GCNs Go as Deep as CNNs?.[pdf](https://arxiv.org/pdf/1904.03751.pdf)

Deep Learning on Graphs For Computer Vision — CNN, RNN, and GNN

https://medium.com/@utorontomist/deep-learning-on-graphs-for-computer-vision-cnn-rnn-and-gnn-c114d6004678




