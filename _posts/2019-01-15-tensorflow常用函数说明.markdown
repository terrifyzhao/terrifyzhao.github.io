---
layout: post
title: TensorFlow常用函数说明'
subtitle: 'TensorFlow常用函数说明'
date: 2019-01-15
categories: TensorFlow
cover: 'https://raw.githubusercontent.com/terrifyzhao/terrifyzhao.github.io/master/assets/img/2019-01-15-tensorflow%E5%B8%B8%E7%94%A8%E5%87%BD%E6%95%B0%E8%AF%B4%E6%98%8E/cover.jpg'
tags: NLP
---

`tf.concat()`
组合两个张量，axis表示是把哪个维度进行组合即直接把对应维度相加

```
a = np.arange(6).reshape(2, 3)
b = np.arange(6).reshape(2, 3)
print(a.shape)
print(b.shape)
c = tf.concat((tf.convert_to_tensor(a), tf.convert_to_tensor(b)), 0)
print(c)
d = tf.concat((tf.convert_to_tensor(a), tf.convert_to_tensor(b)), 1)
print(d)

out：
(2, 3)
(2, 3)
Tensor("concat:0", shape=(4, 3), dtype=int32)
Tensor("concat_1:0", shape=(2, 6), dtype=int32)
```

`tf.expand_dims()`
扩展一个维度

```
a = tf.range(10)
print(a)
b = tf.expand_dims(a, 0)
print(b)

out：
Tensor("range:0", shape=(10,), dtype=int32)
Tensor("ExpandDims:0", shape=(1, 10), dtype=int32)
```



`tf.tile()`
张量扩展，如果现有一个形状如[`width`, `height`]的张量，需要得到一个基于原张量的，形状如[`batch_size`,`width`,`height`]的张量，其中每一个batch的内容都和原张量一模一样
```
a = tf.expand_dims(tf.range(10), 0)
print(a)
b = tf.tile(a, [32, 1])
print(b)

out：
Tensor("ExpandDims:0", shape=(1, 10), dtype=int32)
Tensor("Tile:0", shape=(32, 10), dtype=int32)
```

`tf.linalg.LinearOperatorLowerTriangular()`
给张量设置一个全是0的上三角
```
a = np.arange(1, 10).reshape(3, 3)
a = tf.convert_to_tensor(a, tf.float32)
b = tf.linalg.LinearOperatorLowerTriangular(a).to_dense()
with tf.Session() as sess:
    c = sess.run(b)
    print(c)

out:
[[1. 0. 0.]
 [4. 5. 0.]
 [7. 8. 9.]]
```

`tf.where(tf.equal(a, 0),b,c)`
a中为0的位置取b的值，不为0的位置取c的值
```
a = [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
b = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
c = [[2, 2, 2], [2, 2, 2], [2, 2, 2]]
d = tf.where(tf.equal(a, 0), b, c)
with tf.Session() as sess:
    print(sess.run(d))

out:
[[2 2 1]
 [1 2 1]
 [1 1 2]]
```


