# Backbone

1、Res2Net,南开大学提出。计算机视觉的主题是提取更好的特征表示，多尺度特征提取是图像分类，识别，检测，分割的重要手段，
FPN/ResNet/ResNeXt/DLA/DenseNet等模型都在构造各种提高性能的连接，赏心悦目的美学结构，终极目标应该是何凯明等人提出的
网络随机生成器。Res2Net的基本结构很容易理解，基本单元拆分Res2Net为分组卷积和SENet，显著降低计算量同时提高准确率。论文
在分类，检测，语义分割，实体分割，显著性分割等领域均做了充分的实验，比如Res2Net-50相比ResNet-50，在ImageNet数据集
top-1分类误差降低0.93%，而FLOPs降低69%。期待源码以及更多领域提高性能和实时性。

Res2Net: A New Multi-scale Backbone Architecture.[pdf](https://arxiv.org/pdf/1904.01169.pdf)

# Detection
## NMS系列去重算法

2017----Soft-NMS----Improving Object Detection With One Line of Code

2018----Softer-NMS- Rethinking Bounding Box Regression for Accurate Object Detection

2018----IoUNet----Acquisition of Localization Confidence for Accurate Object Detection

2018----CVPR----Improving Object Localization with Fitness NMS and Bounded IoU Loss

2018----NIPS----Sequential Context Encoding for Duplicate Removal

M2Det: A Single-Shot Object Detector based on Multi-Level Feature Pyramid Network [PDF](https://arxiv.org/pdf/1811.04533.pdf) [Github](https://github.com/qijiezhao/M2Det)

Grid R-CNN [PDF](https://arxiv.org/pdf/1811.12030.pdf)
 
## 人脸检测

1、天津大学、武汉大学、腾讯AI实验室提出的人脸检测模型，主要针对移动端设计（backbone MobileNet v2）
在高通845上达到140fps的实时性。论文主要提出一个解决类别不均衡问题（侧脸、正脸、抬头、低头、表情、遮挡等各种类型）。

但是MobileNet v2的检测框架应该没有这么快，并且论文的预测特征点和3D旋转角时，使用全连接网络，计算量大，
应该更耗时才对。论文只给出特征点预测，但是一张图片N个人，如何区分特征点属于何人？论文没有告知如何计算检测框。
论文应该很多细节没有讲述清晰，应该是idea分拆，写成连续剧的节奏。

PFLD:A Practical Facial Landmark Detector.[pdf](https://arxiv.org/pdf/1902.10859.pdf)

## [2019年3月]

1、国防科技大学和旷视科技联合提出，典型的RPN+FPN架构，backbone基于SNet,增加Context Enhancement
Module(FPN多尺度分辨率特征融合)和spatial attention module（RPN->1x1卷积实现空间注意力模型），
实验结果相对于MobileNetV2-SSDLite速度和精度均有提高。

ThunderNet: Towards Real-time Generic Object Detection.[PDF](https://arxiv.org/pdf/1903.11752.pdf)

2、商汤和香港中文大学联合提出，ICLR2019论文，实在没看懂啥意思。

Feature Intertwiner for Object Detection. [PDF](https://arxiv.org/pdf/1903.11851.pdf)