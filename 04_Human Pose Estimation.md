
# 数据集

Stereo Hand Pose Tracking Benchmark (STB)

the Rendered Hand Pose Dataset (RHD)

# datatset
# 2D
LSP, FashionPose, PASCAL Person Layout,J-HMDB, MSCOCO, MPII, AI Challenger



# Human Pose Estimation


1、CVPR2019论文，微软和中科大联合提出。一般CNN是先降低网络的特征分辨率再恢复高分辨率，容易损失特征。
作者提出保存高分辨率的网络架构，降低降采样的频率和多尺度融合。保存的特征分辨率高意味着运算量大，不能做到实时性。
COCO，MPII test，PoseTrack2017取得较高的准确率，但是实时性避而不谈。

图像分类、目标检测、语义分割以及视频理解领域都需要高分辨率的特征提高精度，这也是FPN的初衷。
准确率和实时性要平衡。

Deep High-Resolution Representation Learning for Human Pose Estimation. [pdf](http://cn.arxiv.org/pdf/1902.09212.pdf)


# Hand Pose Estimation

## [2019年3月]

1.CVPR2019论文，南洋理工大学提出，two-stacked hourglass network提取特征，Graph CNN在RGB图像重建
包括3D手势和姿态的3D mesh，合成3D meshes and 3D poses数据集用于训练，借助于depth map fine-tuning.
最终模型在Nvidia GTX 1080 GPU可以运行50fps。

合成数据集训练：heat-map loss+3D mesh loss+3D pose loss

真实数据集训练：heat-map loss+depth map loss+pseudo-ground truth loss

3D Hand Shape and Pose Estimation from a Single RGB Image.[pdf](https://arxiv.org/pdf/1903.00812.pdf)

# 待处理

