---
layout: post
title: 'Transformer源码解读'
subtitle: 'Transformer源码解读'
date: 2019-01-11
categories: NLP
cover: 'https://raw.githubusercontent.com/terrifyzhao/terrifyzhao.github.io/master/assets/img/2019-01-11-BERT%E5%AE%8C%E5%85%A8%E6%8C%87%E5%8D%97/cover.jpg'
tags: NLP
---



之前我们一起了解了attention、transformer的原理，本文将会基于github的一个[transformer](https://github.com/Kyubyong/transformer)开源代码进行代码分析讲解，该代码相比于Google提供的[tensor2tensor/transformer](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py)代码更简单，也更容易理解。

这里简单说一下代码怎么运行：
*   下载数据集 [IWSLT 2016 German–English parallel corpus](https://wit3.fbk.eu/download.php?release=2016-01&type=texts&slang=de&tlang=en) 并解压到 `corpora/` 文件夹。
*   在`hyperparams.py`文件中调整超参数，可不调。
*   执行`prepro.py` 文件，把数据做一个预处理，结果会保存在`preprocessed` 文件夹下。
*   执行 `train.py` 即可开始训练模型。

该代码主要包括已经几个文件：

* `hyperparams.py` 该文件包含所有需要用到的参数
* `prepro.py` 该文件生成源语言和目标语言的词汇文件。
* `data_load.py` 该文件包含所有关于加载数据以及批量化数据的函数。
* `modules.py` 该文件具体实现编码器和解码器网络
* `train.py` 训练模型的代码，定义了模型，损失函数以及训练和保存模型的过程
* `eval.py` 评估模型的效果







