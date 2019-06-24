# survey/overview/review

Single image super-resolution (SISR)是从低分辨率图像恢复高分辨率图像。


  - [Deep Learning for Single Image Super-Resolution: A Brief Review](https://arxiv.org/pdf/1808.03344.pdf)

# Super-Resolution

- ECCV2014论文，香港中文大学汤晓鸥组提出SRCNN(Super-Resolution Convolutional Neural Network)，深度学习应用在超分辨率重构(SISR)上的开山之作。从信息熵角度理解，从低分辨率到高分辨率有无线的对应
关系，所以依赖样本的先验知识(example-based，sparse-coding-based等)。论文提出让CNN 逼近sparse-coding-based，流程包括Patch extraction and representation
，Non-linear mapping，Reconstruction（每个模块仅对应一个卷积层，故参数空间为{W1，W2,W3,B1,B2,B3}),使用的训练集timofte数据集共91张图片（直接在CPU训练，Intel CPU 3.10 GHz and 16 GB memory），损失函数MSE，评价函数PSNR。


  - [Learning a Deep Convolutional Network for Image Super-Resolution](http://personal.ie.cuhk.edu.hk/~ccloy/files/eccv_2014_deepresolution.pdf)


- CVPR2019论文，中科大，自动化所，旷视等联合提出Meta-SR, 单一模型解决任意尺度的 super-resolution。模型包括：Feature Learning Module和MetaUpscale Module。Feature Learning Module
可使用RDN，EDSR，通用的特征提取模块(类似ResNet/DenseNet)；Meta-Upscale学习不同比率下上采样权重。可使用的损失函数包括L1,L2正则化。Super-Resolution是否可以理解为如何语义分割的上采样过程？super-resolution的backbone，upsample,loss函数都可借鉴分类/分割的设计，遍地都是机会啊。

  - [Meta-SR: A Magnification-Arbitrary Network for Super-Resolution](https://arxiv.org/pdf/1903.00875.pdf)
  
  

# Underexposed Photo Enhancement


  
 - CVPR2019论文，香港中文大学等提出。论文假设自然图像的光照图有着相对简单的先验，让网络模型去学习image-to-illumination mapping，实现retinex的图像增强。另外论文提出损失函数是Reconstruction Loss(L2)+Smoothness Loss+Color Loss。论文在MIT-Adobe FiveK( 5,000 raw images)之外标注3,000训练集训练模型(什么样的模型用这么少的训练集？)backbone使用VGG16,Titan X Pascal GPU训练40 epochs。

   - [Underexposed Photo Enhancement using Deep Illumination Estimation](http://jiaya.me/papers/photoenhance_cvpr19.pdf)
  

## Evaluation Metrics
PSNR（人眼的感受并不完全一致） and SSIM
## dataset
MIT-Adobe FiveK
