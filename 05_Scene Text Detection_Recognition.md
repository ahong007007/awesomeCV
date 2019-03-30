#[2019年3月]
1、商汤和香港中文大学联合提出的场景文字检测模型。基于Mask RCNN框架，不同点在于mask分支，
mask rcnn预测的是每个像素是否前景和背景，既{0,1}，而论文提出的Pyramid Mask Text Detector 
(PMTD) 预测是[0,1]（中心点接近1，边缘点接近0，可是和立体图形有什么关系？）。论文图1显示标注
框会导致训练错误，结论是要改变标注方式？根据图1b的解释，应该是mask分支修订矩形检测框，可以直接根据mask图像也可以生成任意检测框。
论文最难解释的是plane clustering部分。
论文的实验在ICDAR 2013，2015和2017 MLT均有测试，根据实验结果state-of-art. 

Pyramid Mask Text Detector.[PDF](https://arxiv.org/pdf/1903.11800.pdf)
