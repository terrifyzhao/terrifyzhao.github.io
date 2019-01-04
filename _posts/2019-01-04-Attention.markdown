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

传统的机器翻译基本都是基于seq2seq模型来做的，该模型分为encoder层与decoder层，并均为RNN构成，如下图所示

<iframe  height=500  width=500  src="http://ww4.sinaimg.cn/mw690/e75a115bgw1f3rrbzv1m8g209v0diqv7.gif">
