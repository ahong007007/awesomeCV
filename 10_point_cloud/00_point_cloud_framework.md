# awesome-point-cloud-processing

## survey/overview/review

- 深度学习如何处理3维数据的综述论文。论文对3维数据表示分为两类： Euclidean 和non-Euclidean。Euclidean具有griddded array数据结构，allows for a global parametrization and a common system of coordinates，具有平移不变性。
non-Euclidean没有griddded array的数据属性.
  - [A survey on Deep Learning Advances on Different 3D Data Representations](https://arxiv.org/pdf/1808.01462.pdf):star: :star: :star: :star:
  
- 国防科大提出点云综述。
  - [Deep Learning for 3D Point Clouds: A Survey](https://arxiv.org/pdf/1912.12033.pdf)

- 厦门大学等综述点云与深度学习。
  -[Review: deep learning on 3D point clouds](https://arxiv.org/pdf/2001.06280.pdf)

## Library

- 旷视，中科院自动化所联合提出CBGS：class-balanced采样解决类别不平衡，multi-group head解决同时多目标检测，改善网络模型架构
，损失函数，训练策略。通用点云目标检测框架Det3D兼容KITTI/Lyft数据集，包含Point Pillars/PointNet++等model zoo,提高点云研发效率。
  - 论文在CVPR2019自动驾驶workshop 点云目标检测获得第一名。论文提出的3点点云处理框架Det3D复用mmcv,mmdetection等代码，取得STOA性能。
  - 缺点：Multi-group Head是每一类一组。因为 nuScenes 3D Detection Challenge是10类别，论文框架是10组。当数据集有80或跟更多类别，需要N个Head? 模型
  的可扩展能力较弱。
  - [Class-balanced Grouping and Sampling for Point Cloud 3D Object Detection](https://arxiv.org/pdf/1908.09492.pdf)
  - https://github.com/poodarchu/Det3D
  
- 英伟达发布点云加速框架Kaolin。  
  - [Kaolin: A PyTorch Library for Accelerating 3D Deep Learning Research](https://arxiv.org/pdf/1911.05063v2.pdf)

## Leadboard

- <https://www.nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Any>

---

## Track Advancement of SLAM

 帝国理工的Andrew J.Davison 教授
 ETHz、UZH的Davide. Scaramuzza教授

## Tutorials

[Data Structures for Large 3D Point Cloud Processing](http://www7.informatik.uni-wuerzburg.de/mitarbeiter/nuechter/tutorial2014). Data Structures for Large 3D Point Cloud Processing Tutorial at the 13th International Conference on Intelligent Autonomous Systems

[INF555 Geometric Modeling: Digital Representation
and Analysis of Shapes: lecture 7](http://www.enseignement.polytechnique.fr/informatique/INF555/Slides/lecture7.pdf). 

[3D Deep Learning on Point Cloud Data](http://graphics.stanford.edu/courses/cs468-17-spring/LectureSlides/L16%20-%203d%20deep%20learning%20on%20point%20cloud%20(analysis)%20and%20joint%20embedding.pdf)

## SfM

- 使用LF-Net进行三维重建"，在OpenMVG中集成特征检测算法.
  - https://www.jianshu.com/p/f198c7b39193

VisualSFM: A Visual Structure from Motion System
Colmap: a general-purpose Structure-from-Motion (SfM) and Multi-View Stereo (MVS) pipeline with a graphical and command-line interface.
Bundler: Structure from Motion (SfM) for Unordered Image Collections
CMVS: Clustering Views for Multi-view Stereo
MVE: a complete end-to-end pipeline for image-based geometry reconstruction.
MVS-Texturing: 3D Reconstruction Texturing
OpenMVG: open Multiple View Geometry
OpenMVS: open Multi-View Stereo reconstruction library

## Libraries

- [**PCL - Point Cloud Library**](http://pointclouds.org/) is a standalone, large scale, open project for 2D/3D image and point cloud processing.
- [**3DTK - The 3D Toolkit**](http://slam6d.sourceforge.net/) provides algorithms and methods to process 3D point clouds. 
- [**PDAL - Point Data Abstraction Library**](http://www.pdal.io/) is a C++ BSD library for translating and manipulating point cloud data.
- [**libLAS**](http://www.liblas.org/) is a C/C++ library for reading and writing the very common LAS LiDAR format. 
- [**entwine**](https://github.com/connormanning/entwine/) is a data organization library for massive point clouds, designed to conquer datasets of hundreds of billions of points as well as desktop-scale point clouds.
- [**PotreeConverter**](https://github.com/potree/PotreeConverter) is another data organisation library, generating data for use in the Potree web viewer.
- [**lidR**](https://github.com/Jean-Romain/lidR) R package for Airborne LiDAR Data Manipulation and Visualization for Forestry Applications.

## Software (Open Source)

- [**Paraview**](http://www.paraview.org/). Open-source, multi-platform data analysis and visualization application. 
- [**MeshLab**](http://meshlab.sourceforge.net/). Open source, portable, and extensible system for the processing and editing of unstructured 3D triangular meshes
- [**CloudCompare**](http://www.danielgm.net/cc/). 3D point cloud and mesh processing software 
Open Source Project
- [**OpenFlipper**](http://www.openflipper.org/). An Open Source Geometry Processing and Rendering Framework
- [**PotreeDesktop**](https://github.com/potree/PotreeDesktop). A desktop/portable version of the web-based point cloud viewer [**Potree**](https://github.com/potree/potree)

## Servers

- [**LOPoCS**](https://oslandia.github.io/lopocs/) is a point cloud server written in Python
- [**Greyhound**](https://github.com/hobu/greyhound) is a server designed to deliver points from Entwine octrees

## Web-based point cloud viewers

- [**Potree**](https://github.com/potree/potree) is a web-based octree viewer written in Javascript.

## Conferences

- [**International LiDAR Mapping Forum**](https://www.lidarmap.org/) International LiDAR Mapping Forum (ILMF)
- [**3D-ARCH**](http://www.3d-arch.org/) is a series of international workshops to discuss steps and processes for smart 3D reconstruction, modelling, accessing and understanding of digital environments from multiple data sources.

## Community

- [**Laser Scanning Forum**](https://www.laserscanningforum.com/forum/) Laser Scanning Forum

## github 

paper: [https://github.com/Yochengliu/awesome-point-cloud-analysis.git]


## 4. Dataset

Note that some of these datasets don't provide point cloud data, which means you need some toolboxes to convert data from mesh or RGB-D images.

### Shape understanding

- **ModelNet** [[pdf]](https://people.csail.mit.edu/khosla/papers/cvpr2015_wu.pdf) [[Project]](http://modelnet.cs.princeton.edu/)
- **ShapeNet** [[pdf]](http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v2-old/shapenet/tex/TechnicalReport/main.pdf) [[Project]](https://www.shapenet.org/)

### Indoor scenes

- **2D-3D-S** [[pdf]](http://buildingparser.stanford.edu/images/2D-3D-S_2017.pdf) [[Project]](http://buildingparser.stanford.edu/dataset.html)
- **ScanNet** [[pdf]](https://arxiv.org/pdf/1702.04405.pdf) [[Project]](http://www.scan-net.org/)
- **SUN RGB-D** [[pdf]](http://rgbd.cs.princeton.edu/paper.pdf) [[Project]](http://rgbd.cs.princeton.edu/)

### Autonomous driving (Lidar point cloud)

- **KITTI** [[pdf]](http://www.cvlibs.net/publications/Geiger2013IJRR.pdf) [[Project]](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)
- **nuScenes** [[pdf]](https://arxiv.org/pdf/1903.11027.pdf) [[Project]](https://www.nuscenes.org/)
- **Waymo Open dataset** [[Project]](https://waymo.com/open/)



- [[KITTI](http://www.cvlibs.net/datasets/kitti/)] The KITTI Vision Benchmark Suite. [__`det.`__]
- [[ModelNet](http://modelnet.cs.princeton.edu/)] The Princeton ModelNet . [__`cls.`__]
- [[ShapeNet](https://www.shapenet.org/)]  A collaborative dataset between researchers at Princeton, Stanford and TTIC. [__`seg.`__]
- [[PartNet](https://shapenet.org/download/parts)] The PartNet dataset provides fine grained part annotation of objects in ShapeNetCore. [__`seg.`__]
- [[PartNet](http://kevinkaixu.net/projects/partnet.html)] PartNet benchmark from Nanjing University and National University of Defense Technology. [__`seg.`__]
- [[S3DIS](http://buildingparser.stanford.edu/dataset.html#Download)] The Stanford Large-Scale 3D Indoor Spaces Dataset. [__`seg.`__]
- [[**ScanNet**](http://www.scan-net.org/)] Richly-annotated 3D Reconstructions of Indoor Scenes. [__`cls.`__ __`seg.`__]
- [[Stanford 3D](https://graphics.stanford.edu/data/3Dscanrep/)] The Stanford 3D Scanning Repository. [__`reg.`__]
- [[UWA Dataset](http://staffhome.ecm.uwa.edu.au/~00053650/databases.html)] . [__`cls.`__ __`seg.`__ __`reg.`__]
- [[Princeton Shape Benchmark](http://shape.cs.princeton.edu/benchmark/)] The Princeton Shape Benchmark.
- [[SYDNEY URBAN OBJECTS DATASET](http://www.acfr.usyd.edu.au/papers/SydneyUrbanObjectsDataset.shtml)] This dataset contains a variety of common urban road objects scanned with a Velodyne HDL-64E LIDAR, collected in the CBD of Sydney, Australia. There are 631 individual scans of objects across classes of vehicles, pedestrians, signs and trees. [__`cls.`__ __`match.`__]
- [[ASL Datasets Repository(ETH)](https://projects.asl.ethz.ch/datasets/doku.php?id=home)] This site is dedicated to provide datasets for the Robotics community with the aim to facilitate result evaluations and comparisons. [__`cls.`__ __`match.`__ __`reg.`__ __`det`__]
- [[**Large-Scale Point Cloud Classification Benchmark(ETH)**](http://www.semantic3d.net/)] This benchmark closes the gap and provides a large labelled 3D point cloud data set of natural scenes with over 4 billion points in total. [__`seg.`__]
- [[Robotic 3D Scan Repository](http://asrl.utias.utoronto.ca/datasets/3dmap/)] The Canadian Planetary Emulation Terrain 3D Mapping Dataset is a collection of three-dimensional laser scans gathered at two unique planetary analogue rover test facilities in Canada.  
- [[Radish](http://radish.sourceforge.net/)] The Robotics Data Set Repository (Radish for short) provides a collection of standard robotics data sets.
- [[IQmulus & TerraMobilita Contest](http://data.ign.fr/benchmarks/UrbanAnalysis/#)] The database contains 3D MLS data from a dense urban environment in Paris (France), composed of 300 million points. The acquisition was made in January 2013. [__`cls.`__ __`seg.`__ __`det.`__]
- [[Oakland 3-D Point Cloud Dataset](http://www.cs.cmu.edu/~vmr/datasets/oakland_3d/cvpr09/doc/)] This repository contains labeled 3-D point cloud laser data collected from a moving platform in a urban environment.
- [[Robotic 3D Scan Repository](http://kos.informatik.uni-osnabrueck.de/3Dscans/)] This repository provides 3D point clouds from robotic experiments，log files of robot runs and standard 3D data sets for the robotics community.
- [[Ford Campus Vision and Lidar Data Set](http://robots.engin.umich.edu/SoftwareData/Ford)] The dataset is collected by an autonomous ground vehicle testbed, based upon a modified Ford F-250 pickup truck. 
- [[The Stanford Track Collection](https://cs.stanford.edu/people/teichman/stc/)] This dataset contains about 14,000 labeled tracks of objects as observed in natural street scenes by a Velodyne HDL-64E S2 LIDAR.
- [[PASCAL3D+](http://cvgl.stanford.edu/projects/pascal3d.html)] Beyond PASCAL: A Benchmark for 3D Object Detection in the Wild. [__`pos.`__ __`det.`__]
- [[3D MNIST](https://www.kaggle.com/daavoo/3d-mnist)] The aim of this dataset is to provide a simple way to get started with 3D computer vision problems such as 3D shape recognition. [__`cls.`__]
- [[WAD](http://wad.ai/)] This dataset is provided by Baidu Inc.
- [[nuScenes](https://d3u7q4379vrm7e.cloudfront.net/object-detection)] The nuScenes dataset is a large-scale autonomous driving dataset.
- [[PreSIL](https://uwaterloo.ca/waterloo-intelligent-systems-engineering-lab/projects/precise-synthetic-image-and-lidar-presil-dataset-autonomous)] Depth information, semantic segmentation (images), point-wise segmentation (point clouds), ground point labels (point clouds), and detailed annotations for all vehicles and people. [[paper](https://arxiv.org/abs/1905.00160)] [__`det.`__ __`aut.`__]
- [[3D Match](http://3dmatch.cs.princeton.edu/)] Keypoint Matching Benchmark, Geometric Registration Benchmark, RGB-D Reconstruction Datasets. [__`reg.`__ __`rec.`__ __`oth.`__]
- [[BLVD](https://github.com/VCCIV/BLVD)] (a) 3D detection, (b) 4D tracking, (c) 5D interactive event recognition and (d) 5D intention prediction. 
- [[ICRA 2019 paper](https://arxiv.org/abs/1903.06405v1)] [__`det.`__ __`tra.`__ __`aut.`__ __`oth.`__]
- [[PedX](https://arxiv.org/abs/1809.03605)] 3D Pose Estimation of Pedestrians, more than 5,000 pairs of high-resolution (12MP) stereo images and LiDAR data along with providing 2D and 3D labels of pedestrians. 
- [[ICRA 2019 paper](https://arxiv.org/abs/1809.03605)][__`pos.`__ __`aut.`__]
- [[H3D](https://usa.honda-ri.com/H3D)]Full-surround 3D multi-object detection and tracking dataset. 
- [[ICRA 2019 paper](https://arxiv.org/abs/1903.01568)] [__`det.`__ __`tra.`__ __`aut.`__]
- [[Argoverse BY ARGO AI](https://www.argoverse.org/)] Two public datasets (3D Tracking and Motion Forecasting) supported by highly detailed maps to test, experiment, and teach self-driving vehicles how to understand the world around them.
- [[CVPR 2019 paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Chang_Argoverse_3D_Tracking_and_Forecasting_With_Rich_Maps_CVPR_2019_paper.html)][__`tra.`__ __`aut.`__]
- [[Matterport3D](https://niessner.github.io/Matterport/)] RGB-D: 10,800 panoramic views from 194,400 RGB-D images. Annotations: surface reconstructions, camera poses, and 2D and 3D semantic segmentations. Keypoint matching, view overlap prediction, normal prediction from color, semantic segmentation, and scene classification. [[3DV 2017 paper](https://arxiv.org/abs/1709.06158)] [[code](https://github.com/niessner/Matterport)] [[blog](https://matterport.com/blog/2017/09/20/announcing-matterport3d-research-dataset/)]
- [[SynthCity](https://arxiv.org/abs/1907.04758)] SynthCity is a 367.9M point synthetic full colour Mobile Laser Scanning point cloud. Nine categories. [__`seg.`__ __`aut.`__]
- [[Lyft Level 5](https://level5.lyft.com/dataset/?source=post_page)] Include high quality, human-labelled 3D bounding boxes of traffic agents, an underlying HD spatial semantic map. [__`det.`__ __`seg.`__ __`aut.`__]
- [[SemanticKITTI](http://semantic-kitti.org)] Sequential Semantic Segmentation, 28 classes, for autonomous driving. All sequences of KITTI odometry labeled. 
- [[ICCV 2019 paper](https://arxiv.org/abs/1904.01416)][__`seg.`__ __`oth.`__ __`aut.`__]
- [[NPM3D](http://npm3d.fr/paris-lille-3d)] The Paris-Lille-3D  has been produced by a Mobile Laser System (MLS) in two different cities in France (Paris and Lille).[__`seg.`__] 
- [[The Waymo Open Dataset](https://waymo.com/open/)] The Waymo Open Dataset is comprised of high resolution sensor data collected by Waymo self-driving cars in a wide variety of conditions.[__`det.`__]
- [**ICCV2019** [ScanObjectNN](https://hkust-vgd.github.io/scanobjectnn/)]The dataset contains ~15,000 objects that are categorized into 15 categories with 2902 unique object instances.the first on real-world data.[__`cls.`__]