Rethinking the Value of Network Pruning[1810.05270] 

channel pruning with LASSO-based channel selection [PDF](https://arxiv.org/pdf/1707.06168.pdf)

Discrimination-aware Channel Pruning for Deep Neural Networks[PDF](https://arxiv.org/pdf/1810.11809.pdf)

# Quantization

##  mixed precision Quantization


- 加州大学伯克利分校AI研究中心提出quantizing different layers with different bit-widths。

[mixed precisoin quantization of convnets via differentiable neural architecture search](https://arxiv.org/pdf/1812.00090.pdf)
 

- MIT提出量化策略HAQ。常规量化策略忽略了不同的硬件架构和所有卷积层统一量化标准。提出基于强化学习的自适应量化bit位，根据不同硬件平台latency, energy和storage
设计，不同硬件平台使用不同的量化策略。
  mixed precision是基于硬件平台/IP核的发展和支持：苹果A12，NVIDIA Turing GPU，以及BISMO和BitFusion等，限制模型的应用范围。

  - [HAQ: Hardware-Aware Automated Quantization](https://arxiv.org/pdf/1811.08886.pdf)
  

# pruning

## weight pruning

## filter pruning

- CVPR2019论文，电子信息技术研究院和华为等提出。论文提出filter pruning的一种维度norm-based，Filter Pruning via Geometric Median (FPGM) 实现the most replaceable contribution
替代其他工作的relatively less contribution。每一层求滤波器的中位数并pruning，实现论文优化目标。在算法伪代码中，每一个epoch 按照pruning rate Pi 对滤波器prunging一次。在imagenet 数据集ResNet50
FLOPS减少53.5%,TOP1准确率降低1.32%。 

  - [Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration](https://arxiv.org/pdf/1811.00250.pdf)

## channel pruning
- 清华大学和旷视科技提出，基于MobileNet V1/V2 网络的自动化通道剪枝，相比AMC和NetAdapt有提升

  - [MetaPruning: Meta Learning for Automatic Neural Network Channel Pruning](https://arxiv.org/pdf/1903.10258.pdf)


- 伊利诺伊大学厄巴纳-香槟分校提出的以及channel select算法，论文对mobilenetv1/2 MNasNet 性能提高，推断延迟降低。

  - [Network Slimming by Slimmable Networks:Towards One-Shot Architecture Search for Channel Numbers](https://arxiv.org/pdf/1903.11728.pdf)


- ICLR2019论文。

  - [Slimmable Neural Networks](https://arxiv.org/pdf/1812.08928.pdf)

- CVPR2019论文，厦门大学和北航等联合提出，基于GAN的思路，对filter channel等剪枝。在Imagenet评测，Param减少60%,但是Top-1 降低7%。思路较为新颖，但是性能略差。

[Towards Optimal Structured CNN Pruning via Generative Adversarial Learning](https://arxiv.org/pdf/1903.09291.pdf)


# framework

- ECCV2018,韩松提出AMC(AutoML for Model Compression) 

  - [AMC: AutoML for Model Compression and Acceleration on Mobile Devices](https://arxiv.org/pdf/1802.03494.pdf)

- google和微软提出CADNN，对标TVM tensorflow-lite等移动端压缩框架。论文仅有框架之前性能的对比，没有具体算法说明。

  -- [26ms Inference Time for ResNet-50: Towards Real-Time Execution of all DNNs on Smartphone](https://arxiv.org/pdf/1905.00571.pdf)

# people work

- [MIT 韩松](https://songhan.mit.edu/publications/)

# awesome

[EfficientDNNs](https://github.com/MingSun-Tse/EfficientDNNs)

# 待记录


[T-Net: Parametrizing Fully Convolutional Nets with a Single High-Order Tensor](https://arxiv.org/pdf/1904.02698.pdf)

Snapshot Distillation: Teacher-Student Optimization in One GenerationBinary Ensemble





