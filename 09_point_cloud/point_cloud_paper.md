# Point cloud


<h3> Keywords </h3>

__`dat.`__: dataset &emsp; | &emsp; __`cls.`__: classification &emsp; | &emsp; __`rel.`__: retrieval &emsp; | &emsp; __`seg.`__: segmentation     
__`det.`__: detection &emsp; | &emsp; __`tra.`__: tracking &emsp; | &emsp; __`pos.`__: pose &emsp; | &emsp; __`dep.`__: depth     
__`reg.`__: registration &emsp; | &emsp; __`rec.`__: reconstruction &emsp; | &emsp; __`auto`__: autonomous driving     
__`oth.`__: other, including normal-related, correspondence, mapping, matching, alignment, compression, generative model...

Statistics: :fire: code is available or the paper is very important

---
# survey/review/overview

  - [2017][a review of point clouds segmentation and classification algorithms](https://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XLII-2-W3/339/2017/isprs-archives-XLII-2-W3-339-2017.pdf)

- 论文从点云采集方式(image-derived,LiDAR,RGB-D,SAR)，点云数据集，点云分割方法（edge-based,regin growing,model fitting,unsupervised clustering-based），点云语义分割方法等，是个人认为是点云语义分割入门材料。

  - [2019][A Review of Point Cloud Semantic Segmentation](https://arxiv.org/pdf/1908.08854.pdf)


---

## RGB-D
1、CVPR2019论文，出自于大名鼎鼎李飞飞组，提出模型，一个用于估计RGB-D图像中已知目标6D姿态的通用框架
（类似于视频处理的two-stream，分别处理RGB图像和深度图像,DenseFusion融合两路特征）。在YCB-Video
和LineMOD数据集验证测试。

论文中6自由度指李群SE(3)（包括旋转和平移），目标是求相机的运动姿态。

DenseFusion: 6D Object Pose Estimation by Iterative Dense Fusion. [pdf](https://arxiv.org/pdf/1901.04780.pdf)

---

# pointcloud registration

- ICCV2017论文，在学习open3d时做实验看到，主要是对彩色点云对齐。

  [ICCV2017][Colored Point Cloud Registration Revisited](http://openaccess.thecvf.com/content_ICCV_2017/papers/Park_Colored_Point_Cloud_ICCV_2017_paper.pdf)
    


# 3D object detection

- CVPR2017论文。

  - [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](http://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf) :star: :star: :star: :star:


- PointNet++论文。

 -- [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/pdf/1706.02413.pdf)

- Facebook何凯明等人提出VoteNet,直接基于点云的3D目标检测模型(无image输入，话说何凯明开始多领域作战)。点云一般是稀疏性，直接
做检测具有较高难度。论文基于PointNet++,提出VoteNet，由deep point set networks 和 Hough voting组成。论文在ScanNet和
SUN RGB-D具有良好表现。 CNN在3D object classification ,3D object detection和3D semantic segmentation均已有所表现，
下一个战场应该是3D Instance Segmentation.

  - [Deep Hough Voting for 3D Object Detection in Point Clouds](https://arxiv.org/pdf/1904.09664.pdf)

- 
  - [Going Deeper with Point Networks](https://arxiv.org/pdf/1907.00960.pdf)
  
- RGB-D Image Analysis and Processing,chapter 3

  - [RGB-D image-based Object Detection: from Traditional Methods to Deep Learning Techniques](https://arxiv.org/pdf/1907.09236.pdf)

# 待阅读
pointnet point++ ,VoteNet层次理解
pvnet,SqueezeSeg ，20190723分享
https://zhuanlan.zhihu.com/p/44809266
## Papers 

[Efficient Processing of Large 3D Point Clouds](https://www.researchgate.net/publication/233792575_Efficient_Processing_of_Large_3D_Point_Clouds) Jan Elseberg, Dorit Borrmann, Andreas N̈uchtre, Proceedings of the XXIII International Symposium on Information, Communication and Automation Technologies (ICAT '11), 2011 


