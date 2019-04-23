# Point cloud

## RGB-D
1、CVPR2019论文，出自于大名鼎鼎李飞飞组，提出模型，一个用于估计RGB-D图像中已知目标6D姿态的通用框架
（类似于视频处理的two-stream，分别处理RGB图像和深度图像,DenseFusion融合两路特征）。在YCB-Video
和LineMOD数据集验证测试。

论文中6自由度指李群SE(3)（包括旋转和平移），目标是求相机的运动姿态。

DenseFusion: 6D Object Pose Estimation by Iterative Dense Fusion. [pdf](https://arxiv.org/pdf/1901.04780.pdf)

# 3D object detection

Facebook何凯明等人提出的直接基于点云的3D目标检测模型(无image输入，话说何凯明开始多领域作战)。点云一般是稀疏性，直接
做检测具有较高难度。论文基于PointNet++,提出VoteNet，由eep point set networks 和 Hough voting组成。论文在ScanNet和
SUN RGB-D具有良好表现。 CNN在3D object classification ,3D object detection和3D semantic segmentation均已有所表现，
下一个战场应该是3D Instance Segmentation.

[Deep Hough Voting for 3D Object Detection in Point Clouds](https://arxiv.org/pdf/1904.09664.pdf)

