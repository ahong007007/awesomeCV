# human pose estimation

## Table of Contents

- [datatset](#datatset)
- [awesome_blog](#awesome_blog)
- [2D_Human_Pose_Estimation](#2D_Human_Pose_Estimation)
- [3D_Human_Pose_Estimation](#3D_Human_Pose_Estimation)
- [Person_Image_Generation](#Person_Image_Generation)

## datatset

Stereo Hand Pose Tracking Benchmark (STB)

the Rendered Hand Pose Dataset (RHD)

LSP, FashionPose, PASCAL Person Layout,J-HMDB, MSCOCO, MPII, AI Challenger

## survey

- [An Overview of Human Pose Estimation with Deep Learning](https://medium.com/beyondminds/an-overview-of-human-pose-estimation-with-deep-learning-d49eb656739b)
- [A 2019 guide to Human Pose Estimation with Deep Learning](https://nanonets.com/blog/human-pose-estimation-2d-guide/)

- 单目姿态估计综述。
  - [Monocular Human Pose Estimation: A Survey of Deep Learning-based Methods](https://arxiv.org/pdf/2006.01423.pdf)

## 2D_Human_Pose_Estimation

- top-down
  - CPM
  - Hourglass
  - CPN
  - Simple Baselines
  - HRNet
  - MSPN
- bottom-up
  - openpose
  - Hourglass+Associative Embedding
  - HigherHRNet

- CVPR2019论文，微软和中科大联合提出。一般CNN是先降低网络的特征分辨率再恢复高分辨率，容易损失特征。
作者提出保存高分辨率的网络架构，降低降采样的频率和多尺度融合。保存的特征分辨率高意味着运算量大，不能做到实时性。
COCO，MPII test，PoseTrack2017取得较高的准确率，但是实时性避而不谈。
  - 图像分类、目标检测、语义分割以及视频理解领域都需要高分辨率的特征提高精度，这也是FPN的初衷。
  - 准确率和实时性要平衡。
  - Deep High-Resolution Representation Learning for Human Pose Estimation. [pdf](http://cn.arxiv.org/pdf/1902.09212.pdf)

- 中科院大学黄骏杰，朱政等提出，认为姿态估计存在两个问题：flip导致原图和翻转图像之间姿态估计结果没有对齐，encoding-decoding方法存在较大的统计误差。
论文提出Unbiased Data Processing，针对两个问题提出的解决方法分别是使用单位长度去度量图像的大小，以及在理想情况下无统计误差的编码解码方法。
  - 论文中公式很多，一不小心陷入，再难脱身。
  - flip没有对齐，分类，检测，分割都有可能存在类似的问题？
  - backbone基于HRNet，直接提高1AP，而模型增加的计算量较少。
  -[2019][The Devil is in the Details: Delving into Unbiased Data Processing for Human Pose Estimation](https://128.84.21.199/pdf/1911.07524.pdf)

- HRNet是Top-down结构。HigherHRNet是Bottom-up,主要解决scale variation和small person的问题。
  - 采用Image pyramid，推图像上采样，生成的feature map融合：Heatmap Aggregation。
  - 特征金字塔在经典检测和分割应用较多，用其他领域的idear对本领域的更新，属于组合创新吧。
  - [2020][CVPR][HigherHRNet: Scale-Aware Representation Learning for Bottom-Up Human Pose Estimation](https://arxiv.org/pdf/1908.10357.pdf)
  - <https://github.com/HRNet/Higher-HRNet-Human-Pose-Estimation>

- 三星研究院作品，MPII state-of-art,论文比较水，在HourGlass更新skip connection(待权重的跳跃连接),concatenation方式采用先拼接再卷积，相当于拼接时增加权重。
  - 特征的融合，比如FPN等都可以采用带权重的特征融合方式。
  - [Toward fast and accurate human pose estimation via soft-gated skip connections](https://arxiv.org/pdf/2002.11098v1.pdf)

## 3D_Human_Pose_Estimation

- [A 2019 guide to 3D Human Pose Estimation](https://blog.nanonets.com/human-pose-estimation-3d-guide/)

- CVPR2019论文，南洋理工大学提出，two-stacked hourglass network提取特征，Graph CNN在RGB图像重建
包括3D手势和姿态的3D mesh，合成3D meshes and 3D poses数据集用于训练，借助于depth map fine-tuning.
最终模型在Nvidia GTX 1080 GPU可以运行50fps。
  - 合成数据集训练：heat-map loss+3D mesh loss+3D pose loss
  - 真实数据集训练：heat-map loss+depth map loss+pseudo-ground truth loss
  - 3D Hand Shape and Pose Estimation from a Single RGB Image.[pdf](https://arxiv.org/pdf/1903.00812.pdf)

## Person_Image_Generation

- CVPR2019 oral论文，华中科大微软联合提出，实现根据image和pose，生成Target pose对应的图片，论文主要是给ReID数据增广，但是其
商业价值应用更广泛：生成某人的某些pose，动画特效，尬舞机之类。论文模型基于GAN架构，生成器（输入人物image和pose，生成target pose对应的图像）,
判别器有两个：纹理判别器和形状判别器，判断任务图像和姿态是否真实。姿态迁移是由生成器完成。论文主要介绍生成器的构造，既Pose-Attentional模块的构造。

  - [Progressive Pose Attention Transfer for Person Image Generation](https://arxiv.org/pdf/1904.03349.pdf)
