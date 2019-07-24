

# Data Augmentation

- Google大脑出品。论文提出的数据增强方式是训练过程常用的技巧：Color operations（Equalize, Contrast, Brightness），Geometric operations（e.g., Rotate,ShearX, TranslationY）
Bounding box operations（BBox Only Equalize,BBox Only Rotate, BBox Only FlipLR），硬生地设计(22×6×6)^2×5 ≈ 9.6×10^28的搜索空间(当然可以再增加)，延续NAS的设计思路（强化学习+RNN），
让神经网络选择数据增强的方式和过程。
    1、图像增强的方式没有什么亮点，但是9.6×10^28的搜索空间，想想都头大。
    2、不仅仅目标检测，其他分类，分割等计算机视觉任务都可以通过NAS-Data Augmentation训练模型？
    3、The RNN controller is trained over 20K augmentation policies. The search employed 400 TPU’s over 48 hours,土豪就是这么任性。
    4、Google最近很多论文都是基于NAS实现，NAS-FPN -> MobileNet v3-> EfficientNet -> NAS Data Augmentation，在EfficientNet时Google的调参就是满满的异类(initial learning rate 0.256 that decays by 0.97 every 2.4 epochs).
    Google不如一鼓作气让NAS给模型调参，真正实现AutoML,也能解放调参侠的工作量。

  - [Learning Data Augmentation Strategies for Object Detection](https://arxiv.org/pdf/1906.11172.pdf)[2019.06]

- 韩国kakaobrain作品。搜索空间包括autocontrast,cutout，把数据集分成K-fold，每个fold使用超参数（p是否使用增强的概率,λ数据增强的程度）并行训练，K-fold排序top-N策略组合。实验部分ResNet-200在Imagenet性能优于谷歌Augmentation,但是数据数据没有谷歌丰富，在目标检测数据集也有良好表现。

  - [Fast AutoAugment](https://arxiv.org/pdf/1905.00397.pdf)


# Super-Resolution 

- 小米AI团队团队提出的超分辨率模型。

  - [Fast, Accurate and Lightweight Super-Resolution with Neural Architecture Search](https://arxiv.org/pdf/1901.07261.pdf)[2019.01]


  - [Architecture Search for Image Inpainting]
  
  