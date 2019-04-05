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


