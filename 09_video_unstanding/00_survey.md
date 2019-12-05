# video understanding/video analysis

# survey
 - 澳大利亚麦觉理大学提出，行为识别论文综述,包括Vison Based和Sensor Based两个方向。
  - [Different Approaches for Human Activity Recognition– A Survey](https://arxiv.org/pdf/1906.05074.pdf)

# dataset
- 美图联合清华大学开源的视频分类数据集，规模较大且类别丰富。COIN 数据集采用分层结构，
即第一层是领域（Domain）、第二层是任务（Task）、第三层是步骤（Step），其中包含与
日常生活相关的 11827 个视频，涉及交通工具、电器维修、和家具装修等 12 个领域的 180
 个任务，共 778 个步骤。

  - [2019][COIN: A Large-scale Dataset for Comprehensive Instructional Video Analysis](https://arxiv.org/pdf/1903.02874.pdf)

- Charades Dataset
9848 videos of 157 classes (7985 training and 1863 testing videos). Each video is ∼30 seconds.

- Moments in Time (MiT) Dataset.
The Moments in Time (MiT) datasetvis a large-scale video classification dataset with more than 800K videos (∼3 seconds per video).


# training

- FackbookRoss Girshick，Kaiming He等提出快速训练视频理解模型的方法Multigrid:从固定的mini-batch，到根据时间和空间分辨率调整的动态mini-batch，加速视频理解模的训练，同时实现准确率提升
0.8%。论文实验模型基于I3D, nonlocal, SlowFast等，数据集包括Kinetics, Something-Something,Charades), 实现4.5倍加速训练，从128GPU到1GPU完成视频模型训练成为可行。

  - [2019][A Multigrid Method for Efficiently Training Video Models](https://arxiv.org/pdf/1912.00998.pdf)
