---
layout: post
title: 'Boosted Tree'
subtitle: 'Boosted Tree原理详解'
date: 2018-02-28
categories: 机器学习
cover: 'https://raw.githubusercontent.com/terrifyzhao/terrifyzhao.github.io/master/assets/img/2018-02-28-%E5%86%B3%E7%AD%96%E6%A0%91/cover.jpeg'
tags: 机器学习
---

## 前言

在上一篇文章中，我们讲述了树模型的原理与Boosted Tree的简单介绍，本文将会参考[Tianqi Chen的论文](https://homes.cs.washington.edu/~tqchen/pdf/BoostedTree.pdf)深入讲解Boosted Tree。

## Boosted Tree回顾

Boosted Tree是一族可以将弱学习器提升为强学习器的算法。这族算法的工作机制类似：先从初始训练集训练处一个基本学习器，再根据基学习器的表现对训练样本分布进行调整，使得先前基学习器做错的训练样本在后续可以受到更多的关注，然后基于调整后的样本分布来训练下一个基学习器。如此重复进行，直到基学习器的数量达到预先制定的值T，最终将这T个基学习器进行加权组合。

那每一个基学习器到底是怎么调整的呢，我们接着看下文。

## Boosted Tree原理

开始之前，我们确认下Objective function，其中$L(Θ)$是损失项，$Ω(Θ)$是正则化项

$$Obj(Θ) = L(Θ) + Ω(Θ) $$

此外，我们令损失函数为

$$L = \sum_{i=1}^{n} l(y_i,\hat{y_i})$$

接下来，我们以一个例子来理解下文。我们有一堆用户信息，我们想要根据这些信息来判断用户是否打游戏，结果是一个实数域的值，值越大，用于越可能喜欢打游戏。我们分别看下一棵树的情况和Boosted Tree的情况。

<img src="https://raw.githubusercontent.com/terrifyzhao/terrifyzhao.github.io/master/assets/img/2018-06-15-Boosted%20Tree/bt1.jpg"/>





