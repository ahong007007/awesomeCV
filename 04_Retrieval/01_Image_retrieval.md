# Image retrieval

- 图像检索过程：1、对于输入图像I，计算图像的特征f(I)。2、计算待查询图像q的特征f(q)。3、计算f(I)和f(q)欧式距离d，d正比于图像相似度。
  - VLAD(Vector of Locally Aggregated Descriptors):将若干局部描述子构建一个向量(先提取N个局部特征xi，将这N个特征与K个聚类中心求残差)，用向量表示图像的全局描述子。

- VLAD 和 BoW、Fisher Vector 等都是图像检索领域的经典方法。NetVLAD是基于CNN实现VLAD图像检索方法，类似有 NetRVLAD、NetFV、NetDBoW等传统图像检索和CNN结合的
方法。VLAD过程中求聚类中心和特征向量较为容易，因为符号函数a<sub>k</sub>不可导，也无法方向传播。把a<sub>k</sub>当做残差的加权，转换为分类问题，可以通过softmax求解。
  - 疑问：intra-normalization将每一个D维的特征分别作归一化，L2 normalization又实现什么功能？
  - [NetVLAD: CNN architecture for weakly supervised place recognition](https://arxiv.org/pdf/1511.07247.pdf)

- #TODO  
  - [ACTNET: end-to-end learning of feature activations and multi-stream aggregation for effective instance image retrieval]

---

## Visual localization challenge

- #TODO
  - [2018][CVPR][InLoc: Indoor Visual Localization with Dense Matching and View Synthesis]
  - [2018][CVPR][Benchmarking 6DOF Outdoor Visual Localization in Changing Conditions]

- 苏黎世自动驾驶实验室出品，解决视觉定位问题。论文应该是通提出的HF-Net三维重建点云，然后通过检索方式在线获取图片的相机姿态。应用创新点在于在线实时（Backbone MobileNetv1-v2）。
  - [2019][CVPR][From Coarse to Fine: Robust Hierarchical Localization at Large Scale](http://openaccess.thecvf.com/content_CVPR_2019/papers/Sarlin_From_Coarse_to_Fine_Robust_Hierarchical_Localization_at_Large_Scale_CVPR_2019_paper.pdf)

- #TODO
  - [2019][Visual Localization Using Sparse Semantic 3D Map](https://arxiv.org/pdf/1904.03803.pdf)
  - [2019][CVPR][Understanding the Limitations of CNN-based Absolute Camera Pose Regression]
  