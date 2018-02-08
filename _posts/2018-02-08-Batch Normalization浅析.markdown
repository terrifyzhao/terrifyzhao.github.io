---
layout: post
title: 'Batch Normalization浅析'
subtitle: 'Batch Normalization算法讲解'
date: 2018-02-08
categories: 神经网络
cover: ''
tags: 机器学习 神经网络 trick
postPatterns: 'overlappingCircles'
---


## 背景问题

自从2012年以来，CNN网络模型取得了非常大的进步，而这些进步的推动条件往往就是模型深度的增加。从AlexNet的几层，到VGG和GoogleNet的十几层，甚至到ResNet的上百层，网络模型不断加深，取得的效果也越来越好，然而网络越深往往就越难以训练。我们知道，CNN网络在训练的过程中，前一层的参数变化影响着后面层的变化，而且这种影响会随着网络深度的增加而不断放大。

在CNN训练时，绝大多数都采用mini-batch使用随机梯度下降算法进行训练，那么随着输入数据的不断变化，以及网络中参数不断调整，网络的各层输入数据的分布则会不断变化，那么各层在训练的过程中就需要不断的改变以适应这种新的数据分布，从而造成网络训练困难，难以拟合的问题。 举个例子，比如10个batch，前8个要求z方向+0.1，后两个要求z方向-0.4，最终有效梯度为0.8-0.4*2=0，这种竞争消耗着效率。

对于深层模型，越底层，越到训练后期，这种batch梯度之间的方向竞争会变得厉害（这是统计上的，某个时刻不一定成立），类似饱和。深层模型都用不着坐等梯度衰减，光是batch方向感就乱了训练效率。

BN算法解决的就是这样的问题，他通过对每一层的输入进行归一化，保证每层的输入数据分布是稳定的，从而达到加速训练的目的。

## Batch Normalization的优势

1、可以选择比较大的初始学习率，让你的训练速度飙涨。当然这个算法即使你选择了较小的学习率，也比以前的收敛速度快，因为它具有快速训练收敛的特性

2、具有一定的regularization 作用, 可以减少 Dropout 的使用，dropout的作用是防止overfitting, 实验发现, BN 可以reduce overfitting

3、降低L2权重衰减系数

4、不再需要使用使用局部响应归一化(LRN)层了（LRN实际效果并不理想）

## 为什么要归一化
开始讲解算法前，先来思考一个问题：我们知道在神经网络训练开始前，都要对输入数据做一个归一化处理，那么为什么需要归一化呢？

我们先看个例子，假设为预测房价，自变量为面积与房间数，因变量为房价。
那么可以得到的公式为：

y = w<sub>1</sub>x<sub>1</sub> + w<sub>2</sub>x<sub>2</sub>

其中，x<sub>1</sub>为面积，w<sub>1</sub>为面积的系数，x<sub>2</sub>为房间数，w<sub>2</sub>为房间数的系数

我们假定单位面积的房价是10w，每个房间的额外费用是500，最终

y = 100000x<sub>1</sub> + 500x<sub>2</sub>

x1的轻微变化就会对结果产生巨大的影响，其次，在BP过程中，x2的系数相对较小，收敛过程就会变慢。如果我们把系数都变为同一个范围，那么x1与x2对结果的影响就是一样的，收敛速度就会变快。

未归一化：
<img src="https://raw.githubusercontent.com/terrifyzhao/terrifyzhao.github.io/master/assets/img/2018-02-08-Batch%20Normalization%E6%B5%85%E6%9E%90/normalization1.jpg" with=200, height=200>

归一化：
<img src="https://raw.githubusercontent.com/terrifyzhao/terrifyzhao.github.io/master/assets/img/2018-02-08-Batch%20Normalization%E6%B5%85%E6%9E%90/normalization2.jpg" with=200, height=200>

## Batch Normalization算法讲解



