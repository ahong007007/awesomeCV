#　State of the art


## 框架总结 

  - 1.手动设计网络结构 -> NAS搜索； 
  - 2.固定感受野 -> 引入空间注意力做感受野自动调节；
  - 3.效果提升不上去 -> 换个思路做实时分割来对比结果；
  - 4.自监督太热门 -> 引入弱监督 （GAN, 知识蒸馏, ...） + trick = 差不多的score；
  - 5.DNN太枯燥，融入点传统视觉的方法搞成end-to-end训练；
  - 6.CNN太单调，配合GCN搞点悬念；
  - 7.嫌2D太low逼，转3D点云分割；在3D卷积中引入2D的各种相似结构。

## 结构类总结
  - A+B，A+B+C，A+B+C+D，...
  - 1.注意力机制：spatial/Temporal attention，Hard/Soft attention.
    SE ~ Non-local ~ CcNet ~ GC-Net ~ Gate ~ CBAM ~ Dual Attention ~ Spatial Attention ~ Channel Attention ~ ... 
  【只要你能熟练的掌握加法、乘法、并行、串行四大法则，外加知道一点基本矩阵运算规则（如：HW * WH = HH）和sigmoid/softmax操作，那么你就能随意的生成很多种注意力机制】 
  - 2.卷积结构：Residual block ~ Bottle-neck block ~ Split-Attention block ~ Depthwise separable convolution ~ Recurrent convolution ~ Group convolution  ~ Octave convolution ~ Ghost convolution 
  - 3.可形变卷积：deformable conv,dilated conv
  - 4.多尺度模块：ASPP ~ PPM ~ DCM ~ DenseASPP ~ FPA ~ OCNet ~ MPM... 【好好把ASPP和PPM这两个模块理解一下，搞多/减少几条分支，并联改成串联或者串并联结合，每个分支搞点加权，再结合点注意力或者替换卷积又可以组装上百种新结构出来了】 
  - 5.损失函数：Focal loss ~ Dice loss ~ BCE loss ~ Wetight loss ~ Boundary loss ~ Lovász-Softmax loss ~ TopK loss ~ Hausdorff distance(HD) loss ~ Sensitivity-Specificity (SS) loss ~ Distance penalized CE loss ~ Colour-aware Loss...
  - 6.池化结构：Max pooling ~ Average pooling ~ Random pooling ~ Strip Pooling ~ Mixed Pooling ~...
  - 7.归一化模块：Batch Normalization ~Layer Normalization ~ Instance Normalization ~ Group Normalization ~ Switchable Normalization ~ Filter Response Normalization...
  - 8.骨干网络：LeNet ~ ResNet ~ DenseNet ~ VGGNet ~ GoogLeNet ~ Res2Net ~ ResNeXt ~ InceptionNet ~ SqueezeNet ~ ShuffleNet ~ SENet ~ DPNet ~ MobileNet ~NasNet ~ DetNet ~ EfficientNet ~ ...
  - 9.缩放策略：model dimensions(width, depth and resolution)

## 训练类总结

  - 1.学习衰减策略：StepLR ~ MultiStepLR ~ ExponentialLR ~ CosineAnnealingLR ~ ReduceLROnPlateau ~...
  - 2.优化算法：BGD ~ SGD ~ Adam ~ RMSProp ~ Lookahead ~...
  - 3.正则化方法：dropout,label smoothing,stochastic depth,dropblock  
  - 4.数据增强：水平翻转、垂直翻转、旋转、平移、缩放、裁剪、擦除、反射变换 ~ 亮度、对比度、饱和度、色彩抖动、对比度变换 ~ 锐化、直方图均衡、Gamma增强、PCA白化、高斯噪声、GAN,Mixup

## 目标检测类
  
  - Neck:从直筒型的SSD,STDN到FPN,PANet,Bi-FPN,NAS-FPN,ASFF,i-FPN：本质是增加不同level的feature map感受野，增加小目标/大目标的语义信息.
  - loss&IOU函数:
  - RoIpooling:
  - NMS:

## 语义分割&实例分割
  - loss函数

# backbone
FPN->ResNet->ResNeXt->DLA->DenseNet->Res2Net

不同的架构变化，不同层级的特征分辨率融合，分组卷积，1x1卷积，channel shuffle,终极目标随机网络连接。


# training tricks

## check 

1.Use appropriate logging and meaningful variable names.
 
2.Make sure your network is wired correctly.

3.Implement data augmentation techniques. (mixup,)

4.Use weight initilization and regularization for all layers. 

5.Make sure the regularization terms are not overwhelming the other terms in the loss function.

6.Try overfitting a small dataset.

7.While overfitting the small dataset mentioned above, find a reasonable learning rate. 

8.Perform gradient checks.


## loss function

1、类别不平衡问题。对类别较少样本或hard样本增加系数。

## Loss Value not Improving

1.Make sure you are using an appropriate loss and optimizing the correct tensor.

损失函数类别总结

2.Use a decent optimizer. 

优化总结

3.Make sure your variables are training. 

4.Adjust the initial learning rate and implement an appropriate learning rate schedule. 

5.Make sure you are not overfitting.

学习率 vs 训练步数的曲线，如果像是抛物线，可能就过拟合了。

## Variable Not Training

计算各个层参数的张量范数，把范数为常亮的归结为没有在训练。

1.Make sure your gradient updates are not vanishing.

2.Make sure your ReLus are firing.

## Vanishing/Exploding Gradients

1.Consider using better weight initialization. 

2.Consider changing your activation functions. 

3.If using a recurrent neural net consider using LSTM blocks.

## Overfitting

1.Implement data augmentation techniques.

2.Implement dropout. 

3.Increase regularization.

4.Implement Batch normalization. 

5.Implement validation-based early stopping. 

6.If everything else fails, using a smaller network.

7.增加训练数据集，增加各类别样本，保证类别的平衡。

## more

1.增加超参数搜索范围

## Index 

1.[Troubleshooting Convolutional Neural Networks](https://gist.github.com/zeyademam/0f60821a0d36ea44eef496633b4430fc#before-troubleshooting)

The Best and Most Current of Modern Natural Language Processing
https://medium.com/huggingface/the-best-and-most-current-of-modern-natural-language-processing-5055f409a1d1
test