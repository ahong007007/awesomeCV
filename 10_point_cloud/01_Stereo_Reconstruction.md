# Multi-View Stereo Reconstruction

<h1>
```diff
- Recent papers (from 2017)
```
</h1>

<h3> Keywords </h3>

__`Mon.`__: Monocular &emsp;

Statistics: :fire: code is available & stars >= 100 &emsp;|&emsp; :star: citation >= 50

--

- [survey](#survey)
- [reconstruction](#reconstruction)
- [dynamic-reconstruction](#dynamic-reconstruction)
- [Stereo-Matching](#Stereo-Matching)
- [Depth-Estimation](#Depth-Estimation)
- [Multi-View-Stereo](#Multi-View-Stereo)
- [Benchmark](#Benchmark)

---

## survey

survey/overview/review

- 主要介绍基于图片恢复深度信息的综述。

  - [2019][A Survey on Deep Learning Architectures for Image-based Depth Reconstruction](https://arxiv.org/pdf/1906.06113.pdf) :star: :star: :star: :star:

- 本文主要介绍基于image(单目标物体)的三维重建方法，可以看做indoor场景的子集。

  - [2019][Image-based 3D Object Reconstruction: State-of-the-Art and Trends in the Deep Learning Era](https://arxiv.org/pdf/1906.06543v3.pdf):star: :star: :star: :star:

- 介绍Depth Estimation的一篇博文。
  - [2019][Research Guide for Depth Estimation with Deep Learning](https://heartbeat.fritz.ai/research-guide-for-depth-estimation-with-deep-learning-1a02a439b834)

- [State of the Art on 3D Reconstruction with RGB-D Cameras](http://zollhoefer.com/papers/EG18_RecoSTAR/paper.pdf)

- Robust Vision Challenge
  - <http://robustvision.net/index.php>
  - <https://github.com/tsattler/visuallocalizationbenchmark>
  - <https://paperswithcode.com/task/3d-reconstruction/latest#code>

---

## reconstruction

### indoor

- 微软在2011年提出的KinectFusion，基于RGB-D相机三维重建的开山之作，首次实现实时稠密的三维重建。
  - KinectFusion 算法使用固定体积的网格模型表示重建的三维场景,重建范围有限。
  - 没有回环检测和回环优化，相机累计误差大。
  - [2011][KinectFusion: Real-Time Dense Surface Mapping and Tracking](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ismar2011.pdf)
  - [2016][KinectFusion: Real-time 3D Reconstruction and Interaction Using a Moving Depth Camera](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/kinectfusion-uist-comp.pdf)

- 大名鼎鼎的Thomas Whelan，在KinectFusion基础上改进，位姿估计结合了ICP和直接法，融合了回环检测和回环优化，且使用deformation graph做非刚性变换，使得回环位置可以对齐。
  - [2012][Kintinuous: Spatially Extended KinectFusion](https://www.ri.cmu.edu/pub_files/2012/7/Whelan12rssw.pdf)

- BundleFusion，输入为RGB+depth，数据流首先需要做帧与帧之间的对应关系匹配，做全局位姿优化，对整体做漂移校正。
  - 匹配关系使用sparse-then-dense。先使用稀疏的SIFT特征点来进行比较粗糙的配准，然后使用稠密的几何和光度连续性进行更加细致的配准。
  - 位姿优化方使用分层的local-to-global。
  - [BundleFusion: Real-time Globally Consistent 3D Reconstruction using On-the-fly Surface Re-integration](https://arxiv.org/pdf/1604.01093.pdf)

### outdoor

- 香港科技大学朱思语(现在入职阿里巴巴),目标是解决城市级别百万级图像全局重建问题，采用的手段基于分而治之的思想。
  - [2018][Very Large-Scale Global SfM by Distributed Motion Averaging](https://zpascal.net/cvpr2018/Zhu_Very_Large-Scale_Global_CVPR_2018_paper.pdf)

---

## dynamic-reconstruction

- 第一篇动态场景三维重建论文。
  -首先将每帧获取到的动态目标通过某种变换，转换到一个canonical 空间中，即在该空间中创建一个静态的物体表面模型；
基于每帧的olumetric warp field,将canonical 空间模型还原到live frame。
  - [2015][CVPR][DynamicFusion: Reconstruction and Tracking of Non-rigid Scenes in Real-Time](https://rse-lab.cs.washington.edu/papers/dynamic-fusion-cvpr-2015.pdf)

- 德国亚琛工业大学提出的dynamic object reconstructions，实现Object Tracking, Segmentation and dynamic object Fusion的融合。
  - [Track to Reconstruct and Reconstruct to Track](https://arxiv.org/pdf/1910.00130v2.pdf)
  - <https://github.com/tobiasfshr/MOTSFusion>

---

## Stereo-Matching

- [2018][CVPR][Learning for Disparity Estimation Through Feature Constancy](https://arxiv.org/pdf/1712.01039.pdf)

- stereo reconstruction is decomposed into three important steps: feature extraction (for matching cost
computation), matching cost aggregation and disparity prediction.本文主要介绍实现Stereo Matching。

  -[2019][GA-Net: Guided Aggregation Net for End-to-end Stereo Matching](https://arxiv.org/pdf/1904.06587.pdf)
  
--

## Depth-Estimation

- [2018[CVPR][MegaDepth: Learning Single-View Depth Prediction from Internet Photos](https://arxiv.org/pdf/1804.00607.pdf)  
  
- [2019][SelfVIO: Self-Supervised Deep Monocular Visual-Inertial Odometry and Depth Estimation](https://arxiv.org/pdf/1911.09968.pdf)

- 深度估计综述，包括各种深度学习算法，这几年少有的综述。
  - [A Survey on Deep Learning Techniques for Stereo-based Depth Estimation](https://arxiv.org/pdf/2006.02535.pdf)

---

## Multi-View-Stereo

- 传统multi-view stereo(MVS)利用空间几何基本原理，旨在利用多张影像(影像及对应的相机几何)恢复出三维场景，基本假设是Lambertian反射，既目标表面不吸收任何入射光，在自然场景如玻璃，低纹理特征等场景导致重建失败。香港科技大学和深圳altizure团队提出的基于Deep learning的高精度高效率的三维重建网络MVSNet(ECCV2018 oral)，
包括如特征提取，差分单应矩阵，基于方差的多视觉相识度度量，Depth map估计与修正，实现三维点云生成网络。
  - 缺点：过于耗费内存，难以应用到大尺度场景的问题。
  - 在室内数据集DTU和简单室外场景数据集Tanks and Temples验证，没有室外自然场景的尝试。

  - [2018 ECCV][MVSNet: Depth Inference for Unstructured Multi-view Stereo](https://arxiv.org/pdf/1804.02505.pdf)

- CVPR2019论文，香港大学提出的MVSNet升级改进版。MVSNet过于耗费内存在于Cost Volume Regularization对所有的3D volumes同时进行。论文提出的R-MVSnet引入了循环神经网络架构，依序地在深度方向通过GRU单元正则化2D feature map.
R-MVSNet实现可学习的Depth map的估计，其他非学习模块包括图像预处理，Depth map，滤波与融合。
  - R-MVSNet在可学习的MVS进一步尝试，在室内数据DTUE、TH3D以及简单室外数据集Tanks and Temples,优于传统MVS框架OpenMVS,COLMAP等，离真实的室外场景三维重建更进一步。

  - [2019][CVPR][Recurrent MVSNet for High-resolution Multi-view Stereo Depth Inference](https://arxiv.org/pdf/1902.10556.pdf)

- 清华大学与香港科技大学提出，基于Point-MVSNet解决MVS问题。不同于MVSNet/R-MVSNet基于cost volume的方式，Point-MVSNet直接预测深度图和点云图，包括Coarse Depth Prediction Network,Refned Predictions和Final prediction，
Coarse到refined修订。论文在DTU和Tanks and Temples数据集取得state-of-art水平。
  - 论文开源，实现主要针对简单重建，以及scan9这样简单数据集。

  - [2019][ICCV][Point-Based Multi-View Stereo Network](https://arxiv.org/pdf/1908.04422.pdf)[github](https://github.com/callmeray/PointMVSNet)
  
- Andrew教授项目组，基于单目视觉+语义分割的室内语义三维重建方法。

  -[2019][CVPR][SceneCode: Monocular Dense Semantic Reconstruction using Learned Encoded Scene Representations](https://zpascal.net/cvpr2019/Zhi_SceneCode_Monocular_Dense_Semantic_Reconstruction_Using_Learned_Encoded_Scene_Representations_CVPR_2019_paper.pdf)
  
- 北大深圳研究生院，香港大学联合提出PVA-MVSNet，融合多尺度深度图，多指标约束增强点云重建。feature network and differentiable homography,coarse-to-fine depth map这三个过程类似前人工作MVSNet/PointMVSNet系列。重要改进是
多尺度深度估计的融合，提高准确率同时提升效率。
  - [Pyramid Multi-view Stereo Net with Self-adaptive View Aggregation](https://arxiv.org/pdf/1912.03001v1.pdf)
  - <https://github.com/yhw-yhw/PVAMVSNet>

- 上海大学科技大学提出的Fast-MVSNet.
  - [Fast-MVSNet: Sparse-to-Dense Multi-View Stereo With Learned Propagation and Gauss-Newton Refinement](https://arxiv.org/pdf/2003.13017.pdf)
  - <https://github.com/svip-lab/FastMVSNet>

---

- 传统方法是基于多视角几何计算PnP,RANSAC算法高效但是迭代复杂度高。基于深度学习的方法计算PnP是近些年的一个方向。
耶路撒冷希伯来大学和Google联合提出的方法，在效率上优于RANSAC，但是准确度稍有差距，有待进一步提高。

  - [PnP-Net: A hybrid Perspective-n-Point Network](https://arxiv.org/pdf/2003.04626.pdf)

---

## Benchmark

- Stereo benchmark: ETH3D
  - [2016][IJCV][Large-scale data for multiple-view stereopsis. International Journal of Computer Vision]

- Tanks and temple.
  - [2017][TOG][Tanks and temples: Benchmarking large-scale scene reconstruction]. ACM Transactions on Graphics (TOG) (2017)

- [2017][CVPR][A Multi-View Stereo Benchmark with High-Resolution Images and Multi-Camera Videos](http://www.cvlibs.net/publications/Schoeps2017CVPR.pdf)

- BlendedMVS is a large-scale MVS dataset for generalized multi-view stereo networks.The dataset contains over 17k MVS training samples (113个场景)covering a variety of scenes, including architectures, sculptures and small objects.
  - [2019][BlendedMVS: A Large-scale Dataset for Generalized Multi-view Stereo Networks](https://arxiv.org/pdf/1911.10127v1.pdf)  
  - <https://github.com/YoYo000/BlendedMVS>

## 待记录

 PoseNet，VINet，Perspective Transformer Net，SfMNet，CNN-SLAM，SurfaceNet，3D-R2N2，MVSNet,DeepMVS
 MVDepthNet、DeMoN、DPSNet、MaskMVS，双目的有PSMNet
 sparse reconstruction
 dense reconstruction
