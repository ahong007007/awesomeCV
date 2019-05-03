[1810.05270] Rethinking the Value of Network Pruning

channel pruning with LASSO-based channel selection [PDF](https://arxiv.org/pdf/1707.06168.pdf)

Discrimination-aware Channel Pruning for Deep Neural Networks[PDF](https://arxiv.org/pdf/1810.11809.pdf)


# Quantization

- MIT提出量化策略HAQ。常规量化策略忽略了不同的硬件架构和所有卷积层统一量化标准。提出基于强化学习的自适应量化bit位，根据不同硬件平台latency, energy和storage
设计，不同硬件平台使用不同的量化策略。

  - [HAQ: Hardware-Aware Automated Quantization](https://arxiv.org/pdf/1811.08886.pdf)


## pruning


- 清华大学和旷视科技提出，基于MobileNet V1/V2 网络的自动化通道剪枝，相比AMC和NetAdapt有提升

  - [MetaPruning: Meta Learning for Automatic Neural Network Channel Pruning](https://arxiv.org/pdf/1903.10258.pdf)


- 伊利诺伊大学厄巴纳-香槟分校提出的以及channel select算法，论文对mobilenetv1/2 MNasNet 性能提高，推断延迟降低。

  - [Network Slimming by Slimmable Networks:Towards One-Shot Architecture Search for Channel Numbers](https://arxiv.org/pdf/1903.11728.pdf)


- ICLR2019论文。

  - [Slimmable Neural Networks](https://arxiv.org/pdf/1812.08928.pdf)


# framework

- 韩松提出AMC(AutoML for Model Compression) 

  - [AMC: AutoML for Model Compression and Acceleration on Mobile Devices](https://arxiv.org/pdf/1802.03494.pdf)

- ICML2019论文，google和微软提出CADNN，对标TVM tensorflow-lite等移动端压缩框架。论文仅有框架之前性能的对比，没有具体算法说明。

  -- [26ms Inference Time for ResNet-50: Towards Real-Time Execution of all DNNs on Smartphone](https://arxiv.org/pdf/1905.00571.pdf)

