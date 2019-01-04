---
layout: post
title: 'Attention模型详解'
subtitle: 'Attention模型详解'
date: 2019-01-04
categories: NLP
cover: 'https://raw.githubusercontent.com/terrifyzhao/terrifyzhao.github.io/master/assets/img/2018-11-29-%E4%BD%BF%E7%94%A8BERT%E5%81%9A%E4%B8%AD%E6%96%87%E6%96%87%E6%9C%AC%E7%9B%B8%E4%BC%BC%E5%BA%A6%E8%AE%A1%E7%AE%97/cover.jpeg'
tags: NLP
---

## 简介

相信做NLP的同学对这个机制不会很陌生，它在**Attention is all you need**可以说是大放异彩，在machine translation任务中，帮助深度模型在性能上有了很大的提升，输出了当时最好的state-of-art model。当然该模型除了attention机制外，还用了很多有用的trick，以帮助提升模型性能。但是不能否认的时，这个模型的核心就是attention，attention是**一种能让模型对重要信息重点关注并充分学习吸收的技术**，它不算是一个完整的模型，应当是一种技术，能够作用于任何序列模型中。


## Seq2Seq

在开始讲解Attention之前，我们先简单回顾一下Seq2Seq模型，传统的机器翻译基本都是基于Seq2Seq模型来做的，该模型分为encoder层与decoder层，并均为RNN或RNN的变体构成，如下图所示

![](https://raw.githubusercontent.com/terrifyzhao/terrifyzhao.github.io/master/assets/img/2019-01-04-Attention%E6%A8%A1%E5%9E%8B%E8%AF%A6%E8%A7%A3/pic1.gif)

在encode阶段，第一个节点输入一个词，之后的节点输入的是下一个词与前一个节点的hidden state，最终encoder会输出一个context，这个context又作为decoder的输入，每经过一个decoder的节点就输出一个翻译后的词，并把decoder的hidden state作为下一层的输入。改模型对于短文本的翻译来说效果很好，但是其也存在一定的缺点，如果文本稍长一些，就很容易丢失文本的一些信息，为了解决这个问题，Attention应运而生。

## Attention

Attention，正如其名，注意力，该模型在decode阶段，会选择最适合当前节点的context作为输入。Attention与传统的Seq2Seq模型主要有以下两点不同。

+ encoder提供了更多的数据给到decoder，encoder会把所有的节点的hidden state提供给decoder，而不仅仅只是encoder最后一个节点的hidden state

![](https://raw.githubusercontent.com/terrifyzhao/terrifyzhao.github.io/master/assets/img/2019-01-04-Attention%E6%A8%A1%E5%9E%8B%E8%AF%A6%E8%A7%A3/pic2.gif)

+ Second, an attention decoder does an extra step before producing its output. In order to focus on the parts of the input that are relevant to this decoding time step, the decoder does the following:

1.  Look at the set of encoder hidden states it received – each encoder hidden states is most associated with a certain word in the input sentence
2.  Give each hidden states a score (let’s ignore how the scoring is done for now)
3.  Multiply each hidden states by its softmaxed score, thus amplifying hidden states with high scores, and drowning out hidden states with low scores

+ decoder并不是直接把所有encoder提供的hidden state作为输入，而是采取一种选择机制，把最符合当前位置的hidden state选出来，具体的步骤如下
  + 确定哪一个hidden state与当前节点关系最为密切
  + 计算每一个hidden state的分数值（具体怎么计算我们下文讲解）
  + 对每个分数值做一个softmax的计算，这能让相关性高的hidden state的分数值更大，相关性低的hidden state的分数值更低

![](https://raw.githubusercontent.com/terrifyzhao/terrifyzhao.github.io/master/assets/img/2019-01-04-Attention%E6%A8%A1%E5%9E%8B%E8%AF%A6%E8%A7%A3/pic3.gif)


