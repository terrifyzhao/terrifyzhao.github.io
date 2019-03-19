---
layout: post
title: '新型初始化方法Fixup，不用归一化Normalization也能进行深度训练'
subtitle: 'Fixup Initialization'
date: 2019-03-12
categories: NLP
cover: 'https://raw.githubusercontent.com/terrifyzhao/terrifyzhao.github.io/master/assets/img/2019-02-26-Rasa%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%9702/cover.jpg'
tags: NLP
---


## **简介**

Normalization是目前用来解决梯度消失、梯度爆炸的主流方法，Normalization不仅能加速收敛，还能使用较大的学习率。近期，有一种新的方法提出，即Fixup Initialization，对输入的数据进行一个预先的缩放处理，该方法能有效取代Normalization，并且在10000层的残缺网络上进行了验证，效果能和Normalization媲美，并且在图像分类和机器翻译领域都有超越Normalization的表现。

首先我们来看两个问题

+ 如果没有normalization能否正常训练一个深度残缺网络
+ 如果没有normalization训练残缺网络是否可以采用一样大的学习率，是否收敛的速度会一样快活着说更快

本篇论文的作者给出的答案是**YES**，并围绕以下4个点进行了解释。


## **为什么normalization能帮助训练**

如果不采用normalization很有可能引起梯度消失与梯度爆炸，这是为什么呢，我们来看下其中的细节。我们设残缺网络的残缺模块是$\lbrace F_1,...F_L \rbrace$，输入是$X_0$，那么经过残缺模块得到的输出可以表示为：

$$X_l = X_0 + \sum_{i=0}^{l-1}F_i(X_i)$$

残缺网络的输出值的方差，会随着网络的深度呈指数级增长。我们设每一个$X_l$的方差是$Var[X_l]$，为了简化我们假设残缺模块的初始化权重的均值是0即$E[F_l(X_l)|X_l]=0$，因为$X_{l+1}=X_l+F_l(X_l)$，所以有$Var[X_{l+1}]=E[Var[F_l(X_l)|X_l]]+Var(X_l)$，ResNet的结构能阻碍$X_l$随着深度的增加方差越来越小。因为只要$E[Var[F_l(X_l)|X_l]]>0$那么$Var[X_l]<Var[X_{l+1}]$。目前还有一些初始化的方法能让残缺模块的输出值的方差和输入值的方差很类似，因此$Var[X_{l+1}]\approx2Var[X_l]$，这就会导致输出值的方差随着网络深度的增加呈指数级的增加，最终训练的时候导致梯度爆炸。

权重最终的值是根据交叉熵损失函数经过反向传播来确定最终的下界的，也就是说，如果最后的logits的值如果方差较大，那么就会引起梯度爆炸。

Fixup的作者采用了齐次方程的性质，来避免了上面提到的问题，首先来看下两个正齐次方程（positively homogeneous functions）的定义：

+ 对于$X \in R^m$，当$\alpha > 0$的时候有$f(\alpha X)=\alpha f(X)$称为正齐次方程
+ 设$\theta =\lbrace \theta_i \rbrace_{i \in S}$是$f(x)$的参数，$\theta_{ph} =\lbrace \theta_i \rbrace_{i \in S,ph \subset S}$，我们称$\theta_{ph}$为正齐次集，对于任意一个$\alpha >0$有$f(X;\alpha\theta_{ph})= \alpha f(X;\theta_{ph})$

p.h.（齐次）方程其实在神经网络中很常见，例如没有偏差的全连接层，卷积层，池化层等等。

对于一个分类问题假设目标是$c$个分类，损失函数采用的是交叉熵损失，我们用$f$来表示神经网络softmax层的函数表达式，交叉熵定义为$l(z,y)^\Delta_= - y^T(z-logsumexp(z))$，并提出了以下两个假设：

+ $f$是网络模块的序列结构即$f(X_0)=f_L(f_{L-1(...f_1(X_0))})并且偏差是0$

+ 全连接层的权重是从0均值的对称分布中采样得到的。

根据以上的假设作者得到了以下两个结论

结论1.根据第一个假设可以得到：

$$\parallel \frac {\alpha l}{\alpha x_{i-1}} \parallel \ge \frac{l(z,y)-H(p)}{\parallel x_{i-1} \parallel}$$

其中$p$表示的是softmax的概率值$H$表示的是香浓熵。因为$H(p)$是$log(c)$的上界并且$\parallel x_{i-1} \parallel$是其下界，损失函数的值的增加将会造成梯度范数很大，我们的第二个结论证明了一种用p.h.集来找下界的方法。

结论2.根据假设1我们有：

$$\parallel \frac{\alpha l_{avg}}{\alpha \theta_{ph}} \parallel \ge \frac {1}{M \parallel \theta_{ph} \parallel}  \sum ^M_{m=1}l(z^{m},y^{m})-H(p^{(m)}) ^\Delta_= G(\theta_{ph})$$

并且根据假设1与假设2我们有：

$$EG(\theta_{ph}) \ge \frac{E[max_{i \in [c]}Z_i]-log(c)}{\parallel \theta_{ph} \parallel}$$


未完待续。。。







