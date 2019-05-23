---
layout: post
title: '你真的懂学习率了吗'
subtitle: '学习率Learning Rate进阶讲解'
date: 2019-05-23
categories: 机器学习
cover: 'https://raw.githubusercontent.com/terrifyzhao/terrifyzhao.github.io/master/assets/img/2019-05-20-%E6%96%87%E6%9C%AC%E5%8C%B9%E9%85%8D%E6%A8%A1%E5%9E%8B%E4%B9%8BESIM/cover.jpg'
tags: NLP
---

## **前言**
对于刚刚接触深度学习的的童鞋来说，对学习率只有一个很基础的认知，当学习率过大的时候会导致模型难以收敛，过小的时候会收敛速度过慢，但其实学习率是一个十分重要的参数，合理的学习率才能让模型收敛到最小点而非鞍点。本文后续内容将会给大家简单回顾下什么是学习率，并介绍如何改变学习率并设置一个合理的学习率。

## **什么是学习率**
首先我们简单回顾下什么是学习率，在梯度下降的过程中更新权重时的超参数，即下面公式中的$\alpha$

$$
\theta = \theta - \alpha\frac{\partial}{\partial \theta}J(\theta)
$$

学习率越低，损失函数的变化速度就越慢，容易过拟合。虽然使用低学习率可以确保我们不会错过任何局部极小值，但也意味着我们将花费更长的时间来进行收敛，特别是在被困在局部最优点的时候。而学习率过高容易发生梯度爆炸，loss振动幅度较大，模型难以收敛。下图是不同学习率的loss变化，因此，选择一个合适的学习率是十分重要的。

![](https://raw.githubusercontent.com/terrifyzhao/terrifyzhao.github.io/master/assets/img/2019-05-23-%E5%AD%A6%E4%B9%A0%E7%8E%87Learning%20rate/pic1.jpg)

## **如何设置初始学习率**
通常来说，初始学习率以 0.01 ~ 0.001 为宜，但这也只是经验之谈，这里为大家介绍一种较为科学的设置方法。该方法是Leslie N. Smith 在2015年的一篇论文[Cyclical Learning Rates for Training Neural Networks](https://link.jianshu.com/?t=https://arxiv.org/abs/1506.01186)中的3.3节提出来的一个非常棒的方法来找初始学习率。该方法很简单，首先设置一个十分小的学习率，在每个epoch之后增大学习率，并记录好每个epoch的oss，迭代的epoch越多，那被检验的学习率就越多，最后将不同学习率对应的loss进行对比，