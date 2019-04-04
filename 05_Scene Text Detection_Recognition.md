
# Scene Text Detection
1、商汤和香港中文大学联合提出的场景文字检测模型。基于Mask RCNN框架，不同点在于mask分支，
mask rcnn预测的是每个像素是否前景和背景，既{0,1}，而论文提出的Pyramid Mask Text Detector 
(PMTD) 预测是[0,1]（中心点接近1，边缘点接近0，可是和立体图形有什么关系？）。论文图1显示标注
框会导致训练错误，结论是要改变标注方式？根据图1b的解释，应该是mask分支修订矩形检测框，可以直接根据mask图像也可以生成任意检测框。
论文最难解释的是plane clustering部分。
论文的实验在ICDAR 2013，2015和2017 MLT均有测试，根据实验结果state-of-art. 

Pyramid Mask Text Detector.[PDF](https://arxiv.org/pdf/1903.11800.pdf)


2、CVPR2019论文，基于语义分割的方法解决密集字符黏连的问题。在FPN拼接特征图基础上计算不同kernel（文字块的核心）的语义分割图。
论文有两个重要的超参数：number of scales n，缩放的数目，minimal scale m缩放的尺度。
依次求连通域、渐进扩展算法合并各分割图，得到最终的实例分割。

如果语义分割图一开始就是黏连一起，如果保证最小的kernel情况下字符串不黏连？ 

Shape Robust Text Detection with Progressive Scale Expansion Network.[pdf](https://arxiv.org/pdf/1806.02559.pdf)


# Scene Text Recognition

论文提出一个框架模型，包括Spatial Transformer Network，Feature extraction，Sequence modeling，Sequence modeling，每个
阶段采用主流的方法，共2×3×2×2= 24种实现方式，从准确率最高的反推，应该是(默认已经检测或分割后的文字区域)STN+Backbone+BiLSTM+
Attention模型可以取得最佳效果（没有考虑实时性）。再次证明学好排列组合的重要性。

BiLSTM=编码从后到前+从前向后信息(文字具有前后相关性)，Attention模块主要解决的是特征向量和输入图像中
对应的目标区域准确对齐(Index 1)，其实使用商汤的PMTD预测文本行中心位置即可，节省计算资源。

What is wrong with scene text recognition model comparisons? dataset and model analysis.[pdf](https://128.84.21.199/pdf/1904.01906.pdf)


# Index

1、ICCV2017----Focusing Attention: Towards Accurate Text Recognition in Natural Images[pdf](https://arxiv.org/pdf/1709.02054.pdf)
