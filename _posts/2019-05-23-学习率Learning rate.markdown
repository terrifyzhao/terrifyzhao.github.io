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
