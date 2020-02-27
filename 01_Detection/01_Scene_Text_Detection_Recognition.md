# Scene Text Detection & Recognition

---

- [survey](#survey)
- [Scene_Text_Detection](#Scene_Text_Detection)
- [Scene_Text_Recognition](#Scene_Text_Recognition)
- [DataSet](#DataSet)

---

## survey

- <https://paperswithcode.com/task/scene-text-recognition>
- <https://paperswithcode.com/task/scene-text-detection>

---

## Scene_Text_Detection

- 商汤和香港中文大学联合提出的场景文字检测模型。基于Mask RCNN框架，不同点在于mask分支，
mask rcnn预测的是每个像素是否前景和背景，既{0,1}，而论文提出的Pyramid Mask Text Detector.
(PMTD) 预测是[0,1]（中心点接近1，边缘点接近0，可是和立体图形有什么关系？）。论文图1显示标注
框会导致训练错误，结论是要改变标注方式？根据图1b的解释，应该是mask分支修订矩形检测框，可以直接根据mask图像也可以生成任意检测框。
论文最难解释的是plane clustering部分。
论文的实验在ICDAR 2013，2015和2017 MLT均有测试，根据实验结果state-of-art.

  - [Pyramid Mask Text Detector](https://arxiv.org/pdf/1903.11800.pdf)[1903.11]

- CVPR2019论文，南京大学提出，Progressive Scale Expansion Network (PSENet),基于语义分割的方法解决密集字符黏连的问题。在FPN拼接特征图基础上，人为缩放ground truth，计算不同kernel（文字块的核心）的语义分割图。论文有两个重要的超参数：number of scales n，缩放的数目，minimal scale m缩放的尺度。依次求连通域、渐进扩展算法合并各分割图，得到最终的实例分割。训练损失函数采用 dice coefficient计算相似性，OHEM分离正负样本。

  如果语义分割图一开始就是黏连一起，如果保证最小的kernel情况下字符串不黏连？ OHEM如何实现？

  - [Shape Robust Text Detection with Progressive Scale Expansion Network](https://arxiv.org/pdf/1806.02559.pdf)[1806.02]

Look More Than Once: An Accurate Detector for Text of Arbitrary Shapes

## Scene_Text_Recognition

- 论文提出一个框架模型，包括Spatial Transformer Network，Feature extraction，Sequence modeling，predictor，每个
阶段采用主流的方法，共2×3×2×2= 24种实现方式，从准确率最高的反推，应该是(默认已经检测或分割后的文字区域)STN+Backbone+BiLSTM+
Attention模型可以取得最佳效果（没有考虑实时性）。再次证明学好排列组合的重要性。

  BiLSTM=编码从后到前+从前向后信息(文字具有前后相关性)，Attention模块主要解决的是特征向量和输入图像中
  对应的目标区域准确对齐(Index 1)，其实使用商汤的PMTD预测文本行中心位置即可，节省计算资源。

  - [What is wrong with scene text recognition model comparisons? dataset and model analysis](https://128.84.21.199/pdf/1904.01906.pdf)[1904.01]


## Index

1、ICCV2017----Focusing Attention: Towards Accurate Text Recognition in Natural Images[pdf](https://arxiv.org/pdf/1709.02054.pdf)[1709.02]

## Datasets

- 本论文主要介绍ICDAR2019 Robust Reading Challenge on Arbitrary-Shaped Text（RRC-ArT）的进展，包括 i)scene text detection, ii)scene text recognition, and iii) scene text spotting(同时包括检测和识别。)
  - [2019.09][ICDAR2019 Robust Reading Challenge on Arbitrary-Shaped Text (RRC-ArT)](https://arxiv.org/pdf/1909.07145.pdf)

- 介绍ICDAR 2019 LSVT(Large-scale Street View Text)数据集，任务，评估方法和竞赛结果摘要。
  - [2019.09][ICDAR 2019 Competition on Large-scale Street View Text with Partial Labeling -RRC-LSVT](https://arxiv.org/pdf/1909.07741.pdf)

- 百度开源C-SVT(Chinese Street View Text)中文街景地图数据集,包含3万真实标注的自然场景数据，40万部分标注数据集。
  - [ICCV2019][Chinese Street View Text: Large-scale Chinese Text Reading with Partially Supervised Learning](http://openaccess.thecvf.com/content_ICCV_2019/papers/Sun_Chinese_Street_View_Text_Large-Scale_Chinese_Text_Reading_With_Partially_ICCV_2019_paper.pdf)

| Dataset (Year) | Image Num (train/test) | Text Num (train/test) | Orientation| Language| Characteristics | Detec/Recog Task |
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
|End2End|====|====|====|====|====|====|
| [ICDAR03 (2003)](http://www.iapr-tc11.org/mediawiki/index.php?title=ICDAR_2003_Robust_Reading_Competitions) | 509 (258/251) | 2276 (1110/1156) | Horizontal | En | - | ✓/✓ |
| [ICDAR13 Scene Text(2013)](http://dagdata.cvc.uab.es/icdar2013competition/) | 462 (229/233) | - (848/1095) | Horizontal | En | - | ✓/✓ |
| [ICDAR15 Incidental Text(2015)](http://rrc.cvc.uab.es/?ch=4&com=introduction) | 1500 (1000/500) | - (-/-) | Multi-Oriented | En |  Blur, Small, Defocused | ✓/✓ |
| [ICDAR17 / RCTW (2017)](http://rctw.vlrlab.net/dataset/) | 12263 (8034/4229) | - (-/-) | Multi-Oriented | Cn | - | ✓/✓ |
| [Total-Text (2017)](https://github.com/cs-chan/Total-Text-Dataset) | 1555 (1255/300) | - (-/-) | Multi-Oriented,  Curved | En, Cn | Irregular polygon label | ✓/✓ |
| [SVT (2010)](http://www.iapr-tc11.org/mediawiki/index.php?title=The_Street_View_Text_Dataset) | 350 (100/250) | 904 (257/647) | Horizontal| En| - | ✓/✓ |
| [KAIST (2010)](http://www.iapr-tc11.org/mediawiki/index.php?title=KAIST_Scene_Text_Database) | 3000 (-/-) | 5000 (-/-) | Horizontal| En, Ko| Distorted | ✓/✓ |
| [NEOCR (2011)](http://www.iapr-tc11.org/mediawiki/index.php?title=NEOCR:_Natural_Environment_OCR_Dataset) | 659 (-/-) | 5238 (-/-) | Multi-oriented| 8 langs| - | ✓/✓ |
| [CUTE (2014)](http://cs-chan.com/downloads_CUTE80_dataset.html) | 80 (-/80) | - (-/-) | Curved | En | - | ✓/✓ |
| [CTW (2017)](https://ctwdataset.github.io) |  32K ( 25K/6K) |  1M ( 812K/205K) | Multi-Oriented | Cn |  Fine-grained annotation | ✓/✓ |
| [CASIA-10K (2018)](https://github.com/Jyouhou/SceneTextPapers/blob/master/datasets/CASIA-10K.md) | 10K (7K/3K) | - (-/-) | Multi-Oriented | Cn |  | ✓/✓ |
|Detection Only|====|====|====|====|====|====|
| [OSTD (2011)](http://media-lab.ccny.cuny.edu/wordpress/cyi/www/project_scenetextdetection.html) | 89 (-/-) | 218 (-/-) | Multi-oriented| En| - | ✓/- |
| [MSRA-TD500 (2012)](http://www.iapr-tc11.org/mediawiki/index.php/MSRA_Text_Detection_500_Database_(MSRA-TD500)) | 500 (300/200) | 1719 (1068/651) |  Multi-Oriented | En, Cn |  Long text | ✓/- |
| [HUST-TR400 (2014)](http://mclab.eic.hust.edu.cn/UpLoadFiles/dataset/HUST-TR400.zip) | 400 (400/-) | - (-/-) |  Multi-Oriented | En, Cn |  Long text | ✓/- |
| [ICDAR17 / RRC-MLT (2017)](http://rrc.cvc.uab.es/?ch=8) | 18000 (9000/9000) | - (-/-) | Multi-Oriented |  9 langs | - | ✓/- |
| [CTW1500 (2017)](https://github.com/Yuliang-Liu/Curve-Text-Detector) | 1500 (1000/500) | - (-/-) | Multi-Oriented,  Curved | En | Bounding box with _14_ vertexes | ✓/- |
|Recognition Only|====|====|====|====|====|====|
| [Char74k (2009)](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/) | 74107 (-/-) | 74107 (-/-) | Horizontal| En, Kannada | Character label | -/✓ |
| [IIIT 5K-Word (2012)](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html) | 5000 (-/-) | 5000 (2000/3000) | Horizontal| -| cropped | -/✓ |
| [SVHN (2010)](http://www.iapr-tc11.org/mediawiki/index.php?title=The_Street_View_House_Numbers_(SVHN)_Dataset) | - (-/-) | 600000 (-/-) | Horizontal| -| House number digits | -/✓ |
| [SVTP (2013)](https://github.com/Jyouhou/SceneTextPapers/blob/master/datasets/svt-p.zip) | 639 (-/639) | - (-/-) |  | En | Distorted | -/✓ |