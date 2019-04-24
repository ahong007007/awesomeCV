# tracking


1.SiamFC，ECCV2016论文，牛津大学Luca Bertinetto等提出，深度学习方法在目标跟踪领域的破冰之作。使用函数f(z,x)来比较模板图像z域候选图像x的相似度，
相似度越高，则得分越高。首个基于深度特征却又能保持实时性的跟踪方案，跟踪速度在GPU上达到了86fps（帧每秒），而且其性能超过了绝大多数实时跟踪器。

缺点：

a.在真实世界存在多个目标干扰，遮挡，以及移位等因素，feature map会有多个相应，主要通过高斯窗滤波干扰目标。

b.训练出的网络主要关注外观特征而无视语义信息，容易造成背景干扰。

Fully-Convolutional Siamese Networks for Object Tracking.[pdf](https://arxiv.org/pdf/1606.09549.pdf)

2.SiamRPN CVPR2018论文，商汤，北航和清华共同提出，实时性到160fps(backbone采用AlexNet)。论文提出模型包括两个子网络：Siamese subnetwork （特征提取）
和region proposal subnetwork（包括分类和检测分支）。换个角度，跟踪当做的单样本检测任务，就是把第一帧的BBox视为检测的样例，在其余帧里面检测与它相似的目标。

High Performance Visual Tracking with Siamese Region Proposal Network.[pdf](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_High_Performance_Visual_CVPR_2018_paper.pdf)


3、CVPR2019论文，中科院自动化所（王强）和牛津大学提出SiamMask,目标跟踪和视频分割结合的多任务学习网络。VOT2015以后数据集难度不断增加，
目标预测从轴对齐矩形框到旋转框，其实是mask的一种近视。直接生成mask，可以获取更高精确度的旋转矩形框，这是论文的初衷。

缺点：

a.Siammask的mask预测分支采用SharpMask语义分割模型，精度可使用替代模型提高。

b.目前tracking没有专门处理消失问题（object traker如果从当前画面离开或完全遮挡），特别的，siammask挺容易受到具有语义的distractor影响。
Fast Online Object Tracking and Segmentation: A Unifying Approach.[pdf](https://arxiv.org/pdf/1812.05050.pdf)


4.CVPR2019论文，中国科学院大学和微软联合提出，主要解决Siamese网络架构一般使用较浅的网络架构（比如alexnet）。分析影响神经网络的三个因素
：padding,receptive field size和stride，并提出针对性改善的Cropping-Inside Residual (CIR)单元。模型在OTB和VOT等数据集取得
state-of-art，达到论文提出的改变神经网络deeper和wider的目标。

Deeper and Wider Siamese Networks for Real-Time Visual Tracking.[pdf](https://arxiv.org/pdf/1901.01660.pdf)

5.CVPR2019论文，中科院自动化所和商汤联合提出，同样解决Siamese网络架构一般使用较浅的网络架构（比如alexnet）。
性能：论文提出的SiamRPN++在VOT2018取得性能最优的同时，速度在NVIDIA Titan Xp GPU 35fps,MobileNetv2保持性能的同时可运行70fps.
改进措施：
a.论文同样发现padding对特征提取有损伤,降低padding和stride的影响。（stride降低同时使用空洞卷积提高感受野）。
b.模型基于ResNet架构，并提出层级级联SiamRPN block用于协方差计算，多层次特征图预测目标的相似性。
c.提出Depthwise Cross Correlation (DW-XCorr)，大幅度降低参数和稳定模型训练。
孪生网络和目标跟踪进入深度学习的深度网络时代。

SiamRPN++: Evolution of Siamese Visual Tracking with Very Deep Networks.[pdf](https://arxiv.org/pdf/1901.01660.pdf)


6.商汤，北航等联合提出，多目标跟踪（MOT）框架，可以学会充分利用长期和短期线索来处理MOT场景中的复杂情况。针对短期匹配，使用Siamese-RPN，
长期匹配和矫正，使用ReID。视频检测，分割，跟踪等均可以使用这种机制。

Multi-Object Tracking with Multiple Cues and Switcher-Aware Classification.[pdf](https://arxiv.org/pdf/1901.06129.pdf)

7、哈尔滨工业大学和华为联合提出STAIN(Siamese Attentional Keypoint Network)。一般视频追踪基于discriminative correlation filters和Siamese network， 
而论文提出在三个方面改进：backbone network, attentional mechanism 和detection component。backbone network基于hourglass network设计，cross-attentional
改进空间和时序注意力机制，检测模型基于华为提出的corner point和centroid point。

[Siamese Attentional Keypoint Network for High Performance Visual Tracking](https://arxiv.org/pdf/1904.10128.pdf)
# 待更新

Martin大神新作，需要仔细研读

[Learning Discriminative Model Prediction for Tracking](https://arxiv.org/pdf/1904.07220v1.pdf)


4、Siamese Cascaded Region Proposal Networks for Real-Time Visual Tracking(CRPN,目标跟踪）
作者：Heng Fan, Haibin Ling
论文链接：https://arxiv.org/pdf/1812.06148.pdf


5、LaSOT: A High-quality Benchmark for Large-scale Single Object Tracking(目标跟踪）
作者：Heng Fan, Liting Lin, Fan Yang, Peng Chu, Ge Deng, Sijia Yu, Hexin Bai, Yong Xu, Chunyuan Liao, Haibin Ling
论文链接：https://arxiv.org/pdf/1809.07845.pdf
project链接：https://cis.temple.edu/lasot/


6、Leveraging Shape Completion for 3D Siamese Tracking
作者：Silvio Giancola, Jesus Zarzar, Bernard Ghanem
论文链接：https://arxiv.org/abs/1903.01784


7、Cross-Classification Clustering: An Efficient Multi-Object Tracking Technique for 3-D Instance Segmentation in Connectomics（多目标跟踪)
作者：Yaron Meirovitch, Lu Mi, Hayk Saribekyan, Alexander Matveev, David Rolnick, Casimir Wierzynski, Nir Shavit
论文链接：https://arxiv.org/abs/1812.01157


8、Multiview 2D/3D Rigid Registration via a Point-Of-Interest Network for Tracking and Triangulation (POINT^2)
作者：Haofu Liao, Wei-An Lin, Jiarui Zhang, Jingdan Zhang, Jiebo Luo, S. Kevin Zhou
论文链接：https://arxiv.org/abs/1903.03896



## trade off

illumination, deformation,occlusion and motion,speed

#待合并

Graph Convolutional Tracking

http://nlpr-web.ia.ac.cn/mmc/homepage/jygao/gct_cvpr2019.html#


Learning Discriminative Model Prediction for Tracking.[pdf](https://128.84.21.199/pdf/1904.07220.pdf)




