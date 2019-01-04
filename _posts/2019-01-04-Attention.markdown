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

在encode阶段，第一个节点输入一个词，之后的节点输入的是下一个词与前一个节点的hidden state，最终encode层会输出一个context，这个context又作为decode层的输入，每经过一个decode的节点就输出一个翻译后的词，并把decode的hidden state作为下一层的输入。改模型对于短文本的翻译来说效果很好，但是其也存在一定的缺点，如果文本稍长一些，就很容易丢失文本的一些信息，为了解决这个问题，Attention应运而生。

## Attention
The context vector turned out to be a bottleneck for these types of models. It made it challenging for the models to deal with long sentences. A solution was proposed in [Bahdanau et al., 2014](https://arxiv.org/abs/1409.0473) and [Luong et al., 2015](https://arxiv.org/abs/1508.04025). These papers introduced and refined a technique called “Attention”, which highly improved the quality of machine translation systems. Attention allows the model to focus on the relevant parts of the input sequence as needed.

Let’s continue looking at attention models at this high level of abstraction. An attention model differs from a classic sequence-to-sequence model in two main ways:

First, the encoder passes a lot more data to the decoder. Instead of passing the last hidden state of the encoding stage, the encoder passes _all_ the hidden states to the decoder:

Attention，正如其名，注意力，该模型在decode阶段，会选择

![](https://raw.githubusercontent.com/terrifyzhao/terrifyzhao.github.io/master/assets/img/2019-01-04-Attention%E6%A8%A1%E5%9E%8B%E8%AF%A6%E8%A7%A3/pic2.gif)





