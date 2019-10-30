---
# overview/review/survey

- CVPR2017论文，比对SIFT、SIFT-PCA、 DSP-SIFT、 ConvOpt、 DeepDesc 、TFeat、 LIFT等算法的性能。
  - [2017][CVPR][Comparative Evaluation of Hand-Crafted and Learned Local Features](https://demuc.de/papers/schoenberger2017comparative.pdf)

---

# Dataset

- [Long-team visual localization](https://www.visuallocalization.net/benchmark/)

- [Google Landmark Retrieval 2019](https://www.kaggle.com/c/landmark-retrieval-2019)(https://github.com/cvdfoundation/google-landmark)


# paper
 
- DELF

  - [Large-Scale Image Retrieval with Attentive Deep Local Features](https://arxiv.org/pdf/1612.06321.pdf)

-
  - [ACTNET: end-to-end learning of feature activations and multi-stream aggregation for effective instance image retrieval](https://arxiv.org/pdf/1907.05794.pdf)

- 巴黎多芬纳大学等联合提出D2-Net，基于CNN提取描述子。传统提取特征的SIFT等是先检测关键点再提取描述子方式(detect-then-describe)，特征是稀疏的。
而论文提出detect-and-describe：先基于backbone提取特征，在local descriptor维度计算soft-NMS,在channel维度计算ratio-to-max，最后归一化计算图像像素对应的描述子，
提取的特征是稠密的，不受光照/低纹理特征影响。训练集基于MegaDepth数据集，可提取匹配对，不需要标注。在测试阶段使用多尺度detection和参考SIFT算法修订描述子，提高匹配算法的鲁棒性。
论文实验在HPatches验证匹配性能，验证三维重建能力和Day-night数据集验证定位能力，表现state-of-art性能。
  另外特征D2-Net提取与Loss不容易理解，可参考github源码。

  - 疑问：开源代码代码中训练使用的是model.py中的SoftDetectionModule，测试使用的是model_test.py的HardDetectionModule，流程不太一样，待解决。
  - [2019][CVPR][D2-Net: A Trainable CNN for Joint Description and Detection of Local Features](https://arxiv.org/pdf/1905.03561v1.pdf)
  - [Supplementary Material](https://dsmn.ml/files/d2-net/d2-net-supp.pdf)[github](https://github.com/mihaidusmanu/d2-net)
  
  
-  1st place in the Google Landmark Retrieval 2019 challenge;3rd place in the Google Landmark Recognition 2019 challenge

  - [2019][Large-scale Landmark Retrieval/Recognition under a Noisy and Diverse Dataset](https://arxiv.org/pdf/1906.04087v2.pdf)



#待记录
LF-Net、SuperPoint 
Visual Localization Using Sparse Semantic 3D Map