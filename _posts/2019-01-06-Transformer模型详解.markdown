---
layout: post
title: 'Transformer模型详解'
subtitle: 'Transformer模型详解'
date: 2019-01-06
categories: NLP
cover: 'https://raw.githubusercontent.com/terrifyzhao/terrifyzhao.github.io/master/assets/img/2018-11-29-%E4%BD%BF%E7%94%A8BERT%E5%81%9A%E4%B8%AD%E6%96%87%E6%96%87%E6%9C%AC%E7%9B%B8%E4%BC%BC%E5%BA%A6%E8%AE%A1%E7%AE%97/cover.jpeg'
tags: NLP
---

## 简介

[Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)是一篇Google提出的将Attention思想发挥到极致的论文。这篇论文中提出一个全新的模型，叫 Transformer，抛弃了以往深度学习任务里面使用到的 CNN 和 RNN ，目前大热的Bert就是基于Transformer构建的，这个模型广泛应用于NLP领域，例如机器翻译，问答系统，文本摘要和语音识别等等方向。

## Transformer结构

和Attention模型一样，Transformer模型中也采用了 encoer-decoder 架构。但其结构相比于Attention更加复杂，论文中encoder层由6个encoder堆叠在一起，decoder层也一样。

![](https://raw.githubusercontent.com/terrifyzhao/terrifyzhao.github.io/master/assets/img/2019-01-06-Transformer%E6%A8%A1%E5%9E%8B%E8%AF%A6%E8%A7%A3/pic1.png)

每一个encoder和decoder的内部结构如下图

![](https://raw.githubusercontent.com/terrifyzhao/terrifyzhao.github.io/master/assets/img/2019-01-06-Transformer%E6%A8%A1%E5%9E%8B%E8%AF%A6%E8%A7%A3/pic2.png)

对于encoder，包含两层，一个self-attention层和一个前馈神经网络，self-attention能帮助当前节点不仅仅只关注当前的词，从而能获取到上下文的语义。

decoder也包含encoder提到的两层网络，但是在这两层中间还有一层attention层，帮助当前节点获取到当前需要关注的重点内容。

现在我们知道了模型的主要组件，接下来我们看下模型的内部细节。首先，模型需要对输入的数据进行一个embedding操作，也可以理解为类似w2c的操作，enmbedding结束之后，输入到encoder层，self-attention处理完数据后把数据送给前馈神经网络，前馈神经网络的计算可以并行，得到的输出会输入到下一个encoder。

![](https://raw.githubusercontent.com/terrifyzhao/terrifyzhao.github.io/master/assets/img/2019-01-06-Transformer%E6%A8%A1%E5%9E%8B%E8%AF%A6%E8%A7%A3/pic3.png)

## self-attention
接下来我们详细看一下self-attention，其思想和attention类似，但是self-attention是Transformer用来将其他相关单词的“理解”转换成我们正在处理的单词的一种思路，我们看个例子：
```The animal didn't cross the street because it was too tired```
这里的it到底代表的是animal还是street呢，对于我们来说能很简单的判断出来，但是对于机器来说，是很难判断的，self-attention就能够让机器把it和animal联系起来，接下来我们看下详细的处理过程。

1、首先，self-attention会计算出三个新的向量，我们把这三个向量分别称为Query、Key、Value，这三个向量是用embedding向量与一个矩阵（这个矩阵是随机初始化的，在BP的过程中会一直进行更新。）相乘得到的结果，注意，这三个向量的维度是低于embedding维度的。

![](https://raw.githubusercontent.com/terrifyzhao/terrifyzhao.github.io/master/assets/img/2019-01-06-Transformer%E6%A8%A1%E5%9E%8B%E8%AF%A6%E8%A7%A3/pic4.png)

那么Query、Key、Value这三个向量又是什么呢？这三个向量对于attention来说很重要，当你理解了下文后，你将会明白这三个向量扮演者什么的角色。

2、计算self-attention的分数值，


