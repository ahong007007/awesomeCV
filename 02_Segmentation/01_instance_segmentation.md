# Instance Segmentation

---

## survey/overview/review

---

## Instance segmentation

- Mask R-CNN是何凯明在实体分割的奠基之作，实体分割基本延续Mask R-CNN的框架,也是延续Faster RCNN框架，增加Mask分支，实现多任务的学习。
框架包括ResNet-FPN+Fast RCNN+Mask。
  - backbone:ResNet-FPN.FPN特征提取利器，包括从上往下，横向和从下往上三个方向，实现各个层级的特征融合，具有较强的语义信息和空间信息。
  conv2，conv3，conv4和conv5对应原图的stride分别是{4,8,16,32}，conv1因为占用内参较多，没有使用。（#TODO使用会不会提高准确率性能？）
  - Fast RCNN:RPN生成特征金字塔[P2,P3,P4,P5],对应生成多个region proposal。根据RoI在图像中的面积计算对应所在特征层。（训练时有面积，测试时如何确定在那一个特征层）
  - RoIpooling升级到RoIAlign：每一对RPN的边界xywh取整，且对均分方格内通过双线性插值计算采样点。
  - 增加Mask预测分支，经过FCN后输出特征分辨率28*28*80。loss函数包括检测和分割分支之和。分割的损失函数只在检测该类别时输出。（#TODOsigmod如何输出？）
  - [Mask R-CNN](https://arxiv.org/pdf/1703.06870.pdf)

- Ross Girshick，何凯明等人提出TensorMask，解决密集滑动窗口的目标实体分割。从论文的图2的效果看，TensorMask
可能不如Mask R-CNN，也许作者挑选图错误或图的说明错误。

  - [TensorMask: A Foundation for Dense Object Segmentation](https://arxiv.org/pdf/1903.12174.pdf)

- CVPR2019论文，华中科技大学和地平线联合提出。Motivation来自于Mask RCNN的有classification和classification score，但是Mask
没有score，导致的mask quality不匹配（引入Mask IoU计算，避免detection IoU相同而mask无法优化的问题）。论文在Mask RCNN基础上增加
MaskIoU Head，代码也是facebook 开源框架maskrcnn_benchmark基础上直接修订，简单有效。论文的Ablation study实验证明：a.MaskIoU的框架最有效方式，
b.target category训练方式。个人感觉MaskIoU和score不是线性相关，应该还有很多坑可以填。

  - [Mask Scoring R-CNN](https://arxiv.org/pdf/1903.00241.pdf)
  
- CVPR2019论文，香港中文大学，商汤等联合提出，1st in the COCO 2018 Challenge Object Detection Task。实体分割是目标检测和语义分割结合，
论文提出从Mask RCNN->Cascade Mask R-CNN->Hybrid Task Cascade，集成很多trick。论文发现Cascade Mask R-CNN提升检测3.5%但是分割仅仅提升1.2%，不对等在于语义分割没有融合，于是提出混合式的并行和穿行支路。Cascade Mask R-CNN+Interleaved Execution+Mask information flow+Semantic Feature Fusion组成Hybrid Task Cascade框架，
  - 另外tricks包括DCN,SyncBN，multi-scale train，SENet-154，GA-RPN，multi-scale /flip testing，ensemble提升到49%AP。
  - 论文还研究了ASPP,PAFPN，DCN，PrRoIPool，SoftNMS，检测和语义分割模型都能来个大杂烩。
  - [2019][Hybrid Task Cascade for Instance Segmentation](https://arxiv.org/pdf/1901.07518.pdf) :star::star::star::star::star:
  - 论文作者陈恺详解来龙去脉<https://zhuanlan.zhihu.com/p/57629509>

- ICCV2019论文
  - [InstaBoost: Boosting Instance Segmentation via Probability Map Guided Copy-Pasting](https://arxiv.org/pdf/1908.07801v1.pdf)

- 何凯明团队新作，针对语义分割和实体分割提出图像渲染方法。语义分割的特征分辨率一般为1/8图像，Mask RCNN是28*28,这些特征分辨率比较低，上采样过程中目标边缘过于平滑，丢失细节信息。
  - [2019][PointRend: Image Segmentation as Rendering](https://arxiv.org/pdf/1912.08193.pdf)

- 阿德雷得大学，字节跳动联合提出实例分割方法：整体框架类似于YOLO，bottom-up学习像素属于同一个实例的办法(DenseRePoints,polygen,SSAP)。
  - [2019][SOLO: Segmenting Objects by Locations](https://arxiv.org/pdf/1912.04488.pdf)

-阿德雷得大学，华为等联合提出，基于Top down 和bottom Up方的改进。
  -[2020][BlendMask: Top-Down Meets Bottom-Up for Instance Segmentation](https://arxiv.org/pdf/2001.00309.pdf)

---

## Panoptic Segmentation

- [2019][Panoptic-DeepLab:A Simple, Strong, and Fast Baseline for Bottom-Up Panoptic Segmentation](https://arxiv.org/pdf/1911.10194.pdf)