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

我们这里直接从`train.py`的main函数开始看起。

首先作者从数据集中读取了相应的数据来构建词典，词典的顺序是根据词频决定的，后文的词输入的时候就是采用词的index。
```
# Load vocabulary 
de2idx, idx2de = load_de_vocab()
en2idx, idx2en = load_en_vocab()
```

首先作者构建了一个计算流图，训练过程中每一个epoch都对模型进行一次保存，其中tqdm是一个进度条，用于显示训练的进度。
```
# Construct graph g = Graph()
print("Graph loaded")

# Start session 
sv = tf.train.Supervisor(graph=g.graph,
                         logdir=hp.logdir,
                         save_model_secs=0)
with sv.managed_session() as sess:
    for epoch in range(1, hp.num_epochs + 1):
        if sv.should_stop():
            break
    for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
            sess.run(g.train_op)

    gs = sess.run(g.global_step)
    sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))

print('Done')
```

接下来看下Graph的构造方法，首先是读取数据，该项目的任务是德语翻译英语，这里的x代表的是德语，y表示的是英语，num_batch表示的是batch的轮数，x与y都是对应的词在词典中的index表示成的向量，这里需要注意一个参数，max_len，该参数表示的是词的最大个数，举个例子，假设我们设置max_len为10，如果一个句子分成了的词少于10，那么后面两个词就用0来填充，如果超过了10个词，就把后面多余的舍去。这里x与y的维度是(N, T)，N表示每个batch数据的条数即batch_size，T表示max_len。

```
with self.graph.as_default():
    if is_training:
        self.x, self.y, self.num_batch = get_batch_data()  # (N, T)
    else:  # inference
        self.x = tf.placeholder(tf.int32, shape=(None, hp.maxlen))
        self.y = tf.placeholder(tf.int32, shape=(None, hp.maxlen))
```

定义decoder层的输入，作者把y的值的最后一列即&lt;/S&gt;去除了，并在第一列添加了2，这里的2表示的是开始符&lt;S&gt;，维度依旧是(N, T)
```
# define decoder inputs 
self.decoder_inputs = tf.concat((tf.ones_like(self.y[:, :1]) * 2, self.y[:, :-1]), -1)  # 2:<S>
```

接下来就是encoder层了。首先是embedding层，然后是positional encoding，接下来是dropout层，最后是6个Multihead Attention与Feed Forward，我们分别展开来说。

```
with tf.variable_scope("encoder"):
    ## Embedding
    self.enc = embedding(self.x,
                         vocab_size=len(de2idx),
                         num_units=hp.hidden_units,
                         scale=True,
                         scope="enc_embed")
  
    ## Positional Encoding
    if hp.sinusoid:
          self.enc += positional_encoding(self.x,
                                          num_units=hp.hidden_units,
                                          zero_pad=False,
                                          scale=False,
                                          scope="enc_pe")
     else:
          self.enc += embedding(
              tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0), [tf.shape(self.x)[0], 1]),
              vocab_size=hp.maxlen,
              num_units=hp.hidden_units,
              zero_pad=False,
              scale=False,
              scope="enc_pe")
  
     ## Dropout
     self.enc = tf.layers.dropout(self.enc,
     rate=hp.dropout_rate,
     training=tf.convert_to_tensor(is_training))
    
     ## Blocks
     for i in range(hp.num_blocks):
         with tf.variable_scope("num_blocks_{}".format(i)):
              ### Multihead Attention
              self.enc = multihead_attention(queries=self.enc,
                                             keys=self.enc,
                                             num_units=hp.hidden_units,
                                             num_heads=hp.num_heads,
                                             dropout_rate=hp.dropout_rate,
                                             is_training=is_training,
                                             causality=False)
    
               ### Feed Forward
               self.enc = feedforward(self.enc, num_units=[4 * hp.hidden_units, hp.hidden_units])
```

先看下embedding，这里说一下其中的两个参数，num_units表示的是embedding之后的结果的维度，scale表示的是是否对结果进行缩放。
```
self.enc = embedding(self.x,
                     vocab_size=len(de2idx),
                     num_units=hp.hidden_units,
                     scale=True,
                     scope='enc_embed')
```

看下embedding的具体实现，首先初始化一个lookup_table，zero_pad这个参数默认是0，作者把初始化的lookup_table的第一行全部设为了0，我的理解是因为inputs的第一个维度在上文的时候设置成了开始符&lt;S&gt;，没必要计算这个符号的embedding，接下来就是embedding的操作了，这里直接调用了tensorflow的`tf.nn.embedding_lookup`方法，其中传入了两个参数lookup_table与inputs，其中lookup_table就是嵌入矩阵，这个矩阵的维度是（vocab_size, num_units），
inputs是输入数据的index构成的向量，维度是(N, T)，在embedding过程中，inputs首先会做一个one-hot的处理，之后维度变成了(N, T, vocab_size)，embedding的过程其实就是把inputs与lookup_table做一个矩阵乘法，所以最后得到的结果维度是(N, T，num_units)，最后作者通过scale这个参数来判断是否对结果进行缩放，缩放的值是num_units的开方，这里为什么要缩放我也不清楚，希望清楚的读者能给我留言。
```
with tf.variable_scope(scope, reuse=reuse):
    lookup_table = tf.get_variable('lookup_table',
                                   dtype=tf.float32,
                                   shape=[vocab_size, num_units],
                                   initializer=tf.contrib.layers.xavier_initializer())
    if zero_pad:
        lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                  lookup_table[1:, :]), 0)
    outputs = tf.nn.embedding_lookup(lookup_table, inputs)

    if scale:
        outputs = outputs * (num_units ** 0.5) 

return outputs
```

接下来是positional_encoding，作者这里给出了两种方案，第一种方案positional_encoding是和论文中保持一致的，第二种方案也是embedding操作，不过输入数据做了修改，上文提到的embedding输入的是词典中词的index，这里输入的是max_len的index，这里采用了`tf.range()`方法，获取到了max_len的index，并用`tf.tile()`方法，扩展到了每一个batch的大小，所以输入数据的维度也是（N, T），embedding后的结果和词的embedding的结果进行了相加。

```
## Positional Encoding 
if hp.sinusoid:
    self.enc += positional_encoding(self.x,
                                    num_units=hp.hidden_units,
                                    zero_pad=False,
                                    scale=False,
                                    scope="enc_pe")
else:
    self.enc += embedding(
        tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0), [tf.shape(self.x)[0], 1]),
        vocab_size=hp.maxlen,
        num_units=hp.hidden_units,
        zero_pad=False,
        scale=False,
        scope="enc_pe")
```

我们再看下positional_encoding这个方法。和position的embedding类似，只是嵌入矩阵的初始化的内容进行了进一步的改进，首先position_ind这个变量获得了max_len的下标之后作为输入内容，lookup_table是由正弦余弦编码决定的，具体的原理在上一篇的博客[#Transformer](https://terrifyzhao.github.io/2019/01/11/Transformer%E6%A8%A1%E5%9E%8B%E8%AF%A6%E8%A7%A3.html)中有详细介绍。

```
N, T = inputs.get_shape().as_list()
with tf.variable_scope(scope, reuse=reuse):
    position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])

    # First part of the PE function: sin and cos argument
    position_enc = np.array([
        [pos / np.power(10000, 2.*i/num_units) for i in range(num_units)]
        for pos in range(T)])

    # Second part, apply the cosine to even columns and sin to odds.
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1   
    
    # Convert to a tensor  
    lookup_table = tf.convert_to_tensor(position_enc)

    if zero_pad:
        lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                  lookup_table[1:, :]), 0)
    outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

    if scale:
        outputs = outputs * num_units**0.5    return outputs
```

embedding结束之后是dropout层，比较简单，不再多提。

然后是6层的multihead_attention与feedforward了，我们先看multihead_attention。这里的代码和原paper不太一样，作者把512维的embedding值分成了8份，每一份作为一个attention来进行计算，我对这里的代码进行了改版，改成了和paper一样的模式，我的代码这里的multi-head没有做并发的处理，只是一个简单的循环，所以速度会比较慢，大家可以把这里进行一个并发的改进提高速度。

这一块的代码略长，但是如果知道原理的话很好理解，我这里把它分成多个部分来讲，先看下参数，其中queries与keys都是上文处理好的embedding的值，num_units表示的是attention的大小，num_heads表示attention的个数，还有一个很关键的参数causality，这个参数决定了是否采用Sequence Mask。

```
def multihead_attention(queries, 
                        keys, 
                        num_units=None, 
                        num_heads=8, 
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention", 
                        reuse=None):
```

接下来我们看代码，首先是mask部分，Padding Mask主要是把较短的序列补0，`tf.reduce_sum()`把最后一个维度的值全部相加，即把embedding的值相加，`tf.abs()`是取绝对值，取绝对值是为了`tf.sign()`做准备，该函数会把大于0的值置为1，小于0的置为-1，等于0的置为0，我们下面只用到0和1所以这里要取绝对值。接下来调用`tf.tile()`方法，把0和1扩展到对应的维度，接下来要说一下`tf.where()`这个方法，该方法有三个参数，第一个参数是一个布尔类型的张量，后两个值分别是mask与真实值，第一个参数对应位置为1那么该位置的值就取真实值里面的值，如果第一个参数对应位置为0那么该位置的值就取mask里面的值。

```
with tf.variable_scope(scope, reuse=reuse):
    # Set the fall back option for num_units
    if num_units is None:
        num_units = queries.get_shape().as_list[-1]

    key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
    key_masks = tf.tile(tf.expand_dims(key_masks, 2), [1, 1, num_units])
    paddings = tf.ones_like(keys) * (-2 ** 32 + 1)
    outputs = tf.where(tf.equal(key_masks, 0), paddings, keys)
```

然后是Sequence Mask，这里有一个技巧，就是取一个上三角，首先`tf.ones_like()`方法生成了一个（1，T, C）维度的全是1的张量，C表示的是embedding的维度，`tf.linalg.LinearOperatorLowerTriangular()`这个方法把这个全是1的张量的右上角变成了0，再用`tf.tile()`方法把这个mask变回（N, T, C）维度的，最后依旧采用`tf.where()`方法来做过滤。

```
if causality:
    diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
    # mask 把上三角的值全部设置为0
    tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
    masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)                
    paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
    outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)
```


接下来是multi-head的部分，我这里做了一个循环，所以没法并发操作，循环体内很简单，先把输入过了一个全连接层得到Q、K、V，然后把Q与K做矩阵的乘法，得到的结果除以一个常数并做softmax处理，然后和V相乘，每次循环能得到一个head的结果，8次循环结束后，把所有的结果拼接在一起再过一次全连接层，把维度降到一个head的维度，最后就是残缺层和layer normalization层了。
```
  multi_outputs = None for head in range(num_heads):
  Q = tf.layers.dense(outputs, num_units, activation=tf.nn.relu)  # (N, T_q, C)
  K = tf.layers.dense(outputs, num_units, activation=tf.nn.relu)  # (N, T_k, C)
  V = tf.layers.dense(outputs, num_units, activation=tf.nn.relu)  # (N, T_k, C)    
  output = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))
  output = output / (K.get_shape().as_list()[-1] ** 0.5
  output = tf.nn.softmax(output)  # (h*N, T_q, T_k)
  output = tf.matmul(output, V)
  if multi_outputs is None:
      multi_outputs = output
  else:
      multi_outputs = tf.concat((multi_outputs, output), -1)

  outputs = tf.layers.dense(multi_outputs, 512, activation=tf.nn.relu)  # (N, T_k, C) 
  outputs = tf.layers.dropout(outputs, rate=dropout_rate,       
                              training=tf.convert_to_tensor(is_training))

  # Residual connection 
  outputs += queries

  # Normalize 
  outputs = normalize(outputs)   
```

到这里tansformer的关键mutil-head attention的代码就介绍完了，接下来我们继续看前馈神经网络的代码。这里的代码很简单，只是作者用了一个tricks，把全连接层换成了一个卷积核大小为1的卷积层，然后接上残缺层和layer normalization层。
```
def feedforward(inputs,
                num_units=[2048, 512],
                scope="multihead_attention",
                reuse=None):

   with tf.variable_scope(scope, reuse=reuse):
      # Inner layer
      params = {'inputs': inputs, 'filters': num_units[0], 'kernel_size': 1,
                'activation': tf.nn.relu, 'use_bias': True}
      outputs = tf.layers.conv1d(**params)

      # Readout layer
      params = {'inputs': outputs, 'filters': num_units[1], 'kernel_size': 1,
                'activation': None, 'use_bias': True}
      outputs = tf.layers.conv1d(**params)

      # Residual connection
      outputs += inputs

      # Normalize
      outputs = normalize(outputs)

    return outputs
```

这些就是encoder的全部代码了，接下来到了decoder阶段，decoder层和encoder层代码类似，只是多了一个multi-head attention层，该层会把上文提到的`causality`参数设置为True，因为此时decoder不能提前读取未知的信息。

```
# Decoder 
with tf.variable_scope("decoder"):
    ## Embedding
    self.dec = embedding(self.decoder_inputs,
                         vocab_size=len(en2idx),
                         num_units=hp.hidden_units,
                         scale=True,
                         scope="dec_embed")

    ## Positional Encoding
    if hp.sinusoid:
        self.dec += positional_encoding(self.decoder_inputs,
                                        vocab_size=hp.maxlen,
                                        num_units=hp.hidden_units,
                                        zero_pad=False,
                                        scale=False,
                                        scope="dec_pe")
    else:
        self.dec += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.decoder_inputs)[1]), 0),
                    [tf.shape(self.decoder_inputs)[0], 1]),
                    vocab_size=hp.maxlen,
                    num_units=hp.hidden_units,
                    zero_pad=False,
                    scale=False,
                    scope="dec_pe")

    ## Dropout
    self.dec = tf.layers.dropout(self.dec,
    rate=hp.dropout_rate,
    training=tf.convert_to_tensor(is_training))

    ## Blocks
    for i in range(hp.num_blocks):
        with tf.variable_scope("num_blocks_{}".format(i)):
            ## Multihead Attention ( self-attention)
            self.dec = multihead_attention(queries=self.dec,
                                           keys=self.dec,
                                           num_units=hp.head_size,
                                           num_heads=hp.num_heads,
                                           dropout_rate=hp.dropout_rate,
                                           is_training=is_training,
                                           causality=True,
                                           scope="self_attention")

            ## Multihead Attention ( vanilla attention)
            self.dec = multihead_attention(queries=self.dec,
                                           keys=self.enc,
                                           num_units=hp.head_size,
                                           num_heads=hp.num_heads,
                                           dropout_rate=hp.dropout_rate,
                                           is_training=is_training,
                                           causality=False,
                                           scope="vanilla_attention")
                                           
            ## Feed Forward
            self.dec = feedforward(self.dec, num_units=[4 * hp.hidden_units, hp.hidden_units])
```

拿到decoder的结果后，作者对结果进行了一个acc的计算，代码如下。首先把结果经过一个全连接网络，把最后一个维度变为词典的长度，结果的维度是(N, T, len(en2idx))，然后`tf.arg_max()`方法能找到最后一个维度的最大值对应的词在词典中的下标index，最后`self.pres`的维度为(N, T)，表示的是预测的结果，实际的内容是对应的预测结果在词典中的index，然后调用`tf.not_equal()`方法，把不是0的位置全部置为1，因为为0的位置是&lt;pad&gt;，没有实际意义，在计算acc的时候不应该考虑进去，最后把`self.preds`和`self.y`做一个比较，相同的位置置为1，不同的位置置为0，最后乘上`self.istarget`，把&lt;pad&gt;的位置置为0，求和之后除以`self.istarget`的和即为acc，并把acc添加到summry里，

```
self.logits = tf.layers.dense(self.dec, len(en2idx))
self.preds = tf.to_int32(tf.arg_max(self.logits, dimension=-1))
self.istarget = tf.to_float(tf.not_equal(self.y, 0))
self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y)) * self.istarget) / (tf.reduce_sum(self.istarget))
tf.summary.scalar('acc', self.acc)
```

最后，作者定义了一下损失函数和优化方法。在把结果送到softmax之前，作者对label做了一个平滑处理，然后把label和logits一起送到softmax中，拿到loss的值后还做一个平均的操作，该操作的目的主要是要把pad的无意义值给去除，并把mean_loss添加到summary中方便跟踪，最后定义了一个adam的优化方法。

```
if is_training:
    # Loss
    self.y_smoothed = label_smoothing(tf.one_hot(self.y, depth=len(en2idx)))
    self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_smoothed)
    self.mean_loss = tf.reduce_sum(self.loss * self.istarget) / (tf.reduce_sum(self.istarget))

    # Training Scheme
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
    self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)

    # Summary 
    tf.summary.scalar('mean_loss', self.mean_loss)
    self.merged = tf.summary.merge_all()
```


到此tansformer的源码就分析完成了。