---
# overview/review/survey

- CVPR2017论文，比对SIFT、SIFT-PCA、 DSP-SIFT、 ConvOpt、 DeepDesc 、TFeat、 LIFT等算法的性能。
  - [2017][CVPR][Comparative Evaluation of Hand-Crafted and Learned Local Features](https://demuc.de/papers/schoenberger2017comparative.pdf)

- https://github.com/willard-yuan/awesome-cbir-papers
- https://paperswithcode.com/task/image-retrieval
---

# Dataset

- HPatches:580 image pairs。
  - [A benchmark and evaluation of handcrafted and learned local descriptors](https://github.com/hpatches/hpatches-benchmark)
  
- [Long-team visual localization](https://www.visuallocalization.net/benchmark/)
  - Aachen Day-Night dataset:Reference images:4,328,Query images:922 (824 daytime, 98 nighttime)
- Oxford 5k and Oxf105k
- Paris 6k Par106k

- [Google Landmarks Dataset v1](https://github.com/cvdfoundation/google-landmark)
  - https://www.kaggle.com/google/google-landmarks-dataset

- [Google Landmarks Dataset v2](https://github.com/cvdfoundation/google-landmark)
  - https://drive.google.com/file/d/1d-xOKHTedhUjk5rsNmIInhyRMCUJbPC7/view
    - ● Recognition Training set
    - ○ 4.1M images (3.7x increase)
    - ○ 200K unique landmarks (4x increase)
    - ● Retrieval Index set
    - ○ 762k images (1⁄3 decrease)
    - ○ 101k unique landmarks (6.6x increase)
    - ● Test set
    - ○ 118k images (~same)
    - ○ 1.2k have ground truth (~same)
    - ○ Split into “Public” (1⁄3) and “Private” (2⁄3)
    - ○ Most are “distractor queries”
- [Google Landmark Retrieval 2019](https://www.kaggle.com/c/landmark-retrieval-2019)
  - https://github.com/cvdfoundation/google-landmark
  - https://landmarksworkshop.github.io/CVPRW2019/
  - The goal of the Landmark Recognition 2019 challenge is to recognize a landmark presented in a query image, 
   while the goal of Landmark Retrieval 2019 is to find all images showing that landmark. 

- https://image-matching-workshop.github.io/leaderboard/

# Paper

- match一篇经典论文。
  - [CVPR][2012][Three things everyone should know to improve object retrieval](https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf)
 
- ICCV2017论文，Google提出Google-Landmarks Dataset数据集和DELF(DEep Local Feature)算法。
Google-Landmarks包含100万图像，11万检索图像,图片分布在187个国家，4872城市。DELF基于CNN计算实现大规模数据检索，包括特征提取和匹配。
backbone基于ResNet50(ImageNet预训练),图像首先预处理(输入图像分辨率中心裁剪和缩放，用于训练)，训练过程包括两个阶段：Descriptor 
Fine-tuning和Attention-based训练。模型训练集只需要分类的标注，没有像素级的标注。
为了提高图像检索效率，使用PCA把特征维度减少到40。
  - google基于CNN的匹配三年连发三篇定会文章，果然高产。
  - 疑问：论文描述训练过程， 有裁剪和缩放到224*224，有裁剪和缩放到720*720，以何为准？分别对应两阶段训练的策略。
  - Descriptor Fine-tuning和Attention-based是怎样的过程？
  - [github](https://github.com/tensorflow/models/tree/master/research/delf)
  - [2017][ICCV][Large-Scale Image Retrieval with Attentive Deep Local Features](https://arxiv.org/pdf/1612.06321.pdf)
  - [2018][CVPR][Revisiting Oxford and Paris: Large-Scale Image Retrieval Benchmarking](https://arxiv.org/pdf/1803.11285.pdf)

- Google出品DELF升级版，基于检测和检索融合的图像匹配方法。论文主要有三个贡献：
1.标注Google Landmark Boxes dataset(86k images)数据集，相当于图像的目标检测框真值。
2.训练检测器实现高效区域检索：网络模型backbone输出局部特征和目标检测框。
3.借助于区域特征融合R-ASMK(regional aggregated selective match kernel)，实现图像鉴别：第一阶段生成VLAD描述子，第二阶段基于求和池化和归一化。
论文提出的模型以大欺小，基于Google Landmarks dataset数据训练的模型DELF-GLD，在ROxford 和RParis 数据集实现state-of-art水平。

  - DELF在线计算平台[https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/tf_hub_delf_module.ipynb#scrollTo=mVaKXT3cMSib]
  - [2019][CVPR][Detect-to-Retrieve: Efficient Regional Aggregation for Image Search](https://arxiv.org/pdf/1812.01584.pdf) 
-
  - [ACTNET: end-to-end learning of feature activations and multi-stream aggregation for effective instance image retrieval](https://arxiv.org/pdf/1907.05794.pdf)

- 巴黎多芬纳大学,微软等联合提出D2-Net，基于CNN提取描述子特征。传统提取特征的SIFT等是先检测关键点再提取描述子方式(detect-then-describe)，特征是稀疏的。
而论文提出detect-and-describe：先基于backbone提取特征，在local descriptor维度计算soft-NMS,在channel维度计算ratio-to-max，最后归一化计算图像像素对应的描述子，
提取的特征是稠密的，不受光照/低纹理特征影响。训练集基于MegaDepth数据集，可提取匹配对，不需要标注。在测试阶段使用多尺度detection和参考SIFT算法修订描述子，提高匹配算法的鲁棒性。
论文实验在HPatches验证匹配性能，验证三维重建能力和Day-night数据集验证定位能力，表现state-of-art性能。
  另外特征D2-Net提取与Loss不容易理解，可参考github源码。

  - 疑问：开源代码代码中训练使用的是model.py中的SoftDetectionModule，测试使用的是model_test.py的HardDetectionModule，流程不太一样，待解决。
  - 目前开源代码给的backbone是VGG19，ResNet50/ResNet101是否有更好的表现？特征提取时可以用GPU实现，匹配需要用CPU，对模型加速？
  - Aachen Day-Night localization dataset数据集图片在5K+左右量级。
  - [2019][CVPR][D2-Net: A Trainable CNN for Joint Description and Detection of Local Features](https://arxiv.org/pdf/1905.03561v1.pdf)
  - [Supplementary Material](https://dsmn.ml/files/d2-net/d2-net-supp.pdf)[github](https://github.com/mihaidusmanu/d2-net)
  
-
D2-Net和R2D2都依赖dense gt correspondences, 比如D2-Net依赖MegaDepth，而R2D2是用EpicFlow自己插值出来的，获取成本和精度都是麻烦事。  
  
- Landmarks CVPR19 workshop论文。Google Landmark检索第一名和识别第三名。看这篇论文用了很多tricks,Google-Landmarks-v2数据清洗，backbone集成6个模型：FishNet-150,ResNet-101和ResNetXt101等，
损失函数结合人脸识别的cosine-based softmax losses。cosine annealing,mean-pooling (GeM)，finetuning,Discriminative-Reranking等。从实验数据看，Ensemble 6 models仅提升1%，但是实时性应该打折扣。

  - [2019][Large-scale Landmark Retrieval/Recognition under a Noisy and Diverse Dataset](https://arxiv.org/pdf/1906.04087v2.pdf)
  - https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution
  - https://drive.google.com/file/d/1QmC4UKRhIXNW-sa8jxV5b7I6QbPKvJsi/view
  
- NIPS2019论文，浙江大学提出，主要解决在不同视角图像的匹配关系。论文不是在旋转图像上直接提取特征，而是包含两个分支：直接在旋转图像的特征金字塔提取特征和在变化图像的特征金字塔提取特征，再经过分组卷积和
Bilinear Pool，提取像素的描述符。模型需要和特征检测器(Superpoint/DoG/LF-Net)配合，不是end-to-end方式。如论文使用的评测数据集HPSequences和SUN3D数据量都不足1K,对比试验也仅仅基于SIFT和GeoDesc，实验数据不具有代表性，但是提出对图像/特征均进行映射变变换，具有参考意义。
  - [NIPS][2019][GIFT: Learning Transformation-Invariant Dense Visual Descriptors via Group CNNs](https://arxiv.org/pdf/1911.05932.pdf) 
  -  


#待记录
LF-Net、SuperPoint 
Visual Localization Using Sparse Semantic 3D Map
Global: GeM pooling [Radenovic et al., PAMI’18], 
CVPR’19], or simply embeddings before classifier