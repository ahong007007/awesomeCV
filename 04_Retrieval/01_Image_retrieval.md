# Image retrieval

- 图像检索过程：1、对于输入图像I，计算图像的特征f(I)。2、计算待查询图像q的特征f(q)。3、计算f(I)和f(q)欧式距离d，d正比于图像相似度。
  - VLAD(Vector of Locally Aggregated Descriptors):将若干局部描述子构建一个向量(先提取N个局部特征xi，将这N个特征与K个聚类中心求残差)，用向量表示图像的全局描述子。

-  VLAD 和 BoW、Fisher Vector 等都是图像检索领域的经典方法。NetVLAD是基于CNN实现VLAD图像检索方法，类似有 NetRVLAD、NetFV、NetDBoW等传统图像检索和CNN结合的
方法。VLAD过程中求聚类中心和特征向量较为容易，因为符合函数a<sub>k</sub>不可导，也无法方向传播。把a<sub>k</sub>当做残差的加权，转换为分类问题，可以通过softmax求解。
  - 疑问：intra-normalization将每一个D维的特征分别作归一化，L2 normalization又实现什么功能？
  - [NetVLAD: CNN architecture for weakly supervised place recognition](https://arxiv.org/pdf/1511.07247.pdf)
  
  - [ACTNET: end-to-end learning of feature activations and multi-stream aggregation for effective instance image retrieval]

