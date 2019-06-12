---
layout: post
title: 'tensorflow word2vec源码分析'
subtitle: 'word2vec源码分析'
date: 2019-06-11
categories: NLP
cover: 'https://raw.githubusercontent.com/terrifyzhao/terrifyzhao.github.io/master/assets/img/2019-06-11-tensorflow%20word2vec%E6%BA%90%E7%A0%81%E5%88%86%E6%9E%90/cover.jpg'
tags: NLP
---

## **简介** 

最近在编写文本匹配模型时输入需要传入词向量，并在训练的过程中更新词向量。由于之前都是采用的gensim来生成词向量，词典和嵌入矩阵都不太方便获取到，因此决定采用tensorflow来训练词向量，但是据我在网上的了解，tensorflow训练的词向量整体效果还是不如gensim，gensim的源码我没有看过，如果对此清楚的童鞋请留言，十分感谢。不过在模型的训练阶段，还是会对词向量进行更新，因此即使效果没有达到最佳影响应该也不是特别大，接下来就一起来看下tensorflow的word2vec。

## **源码**

众所周知，word2vec包括两种形式，cbow与skip-gram，cbow是采用context来预测target，skip-gram是采用target来预测context，tensorflow的官方代码采用的是skip-gram，在下文中，我会讲解如何改写成cbow的形式。其次tensorflow的官方代码采用了的负采样的优化方法来加速训练，损失函数为噪声对比估算损失，后文会详细介绍，接下来就一起来看代码。

tensorflow提供了多个版本的word2vec源码，不同的版本核心代码差不多，更多的是速度上的提升，这里我们以basic版本来讲解[word2vec_basic.py](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/examples/tutorials/word2vec/word2vec_basic.py)

[word2vec不同版本代码地址](https://github.com/tensorflow/models/tree/ce03903f516731171633d92a50e2218a4d3303b6/tutorials/embedding)

首先下载带训练的数据，数据是一个没有换行的整段文本，因为是英文，因此根据空格进行分词。

```python
# Step 1: Download the data. 
url = 'http://mattmahoney.net/dc/'     

# pylint: disable=redefined-outer-name
def maybe_download(filename, expected_bytes):
  local_filename = os.path.join(gettempdir(), filename)
    if not os.path.exists(local_filename):
        local_filename, _ = urllib.request.urlretrieve(url + filename,
  local_filename)
    statinfo = os.stat(local_filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify ' + local_filename +
                        '. Can you get to it with a browser?')
    return local_filename

filename = maybe_download('text8.zip', 31344016)

# Read the data into a list of strings. 
def read_data(filename):
  with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

vocabulary = read_data(filename)
print('Data size', len(vocabulary))
```

接下来构建词典，并把一些较为稀少的词替换成UNK。
```python
# Step 2: Build the dictionary and replace rare words with UNK token. 
vocabulary_size = 50000     
def build_dataset(words, n_words):
  count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
  for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
  unk_count += 1
  data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(vocabulary, vocabulary_size)
del vocabulary  
# Hint to reduce memory. 
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
```

接下来是比较重要的一部分，我分成多个部分来讲解。因为是skip-gram模型，所以是用target来预测context。先看三个参数，batch_size表示训练的batch个数，解释num_skips之前我们先说skip_window，skip_window表示的是对于target来说，选择的上下文的单边长度，举个例子，“我的爱好是唱、跳、rap”分词之后的结果是[我，的，爱好，是，唱，跳，rap]，如果这里target选的是爱好，skip_window设置的是2，那么和target相关的词就是[我，的，是，唱]，如果skip_window设置的是1，那么和target相关的词是[的，是]，下文中定义的span参数，就是包括target后的待选集合的长度即`skip_window*2+target` 。再看下batch与labels参数，batch表示的是输入，labels表示的是输出，所以两个参数的第一个维度都是batch_size

```python
data_index = 0
# Step 3: Function to generate a training batch for the skip-gram model. 
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips &lt;= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1 # [ skip_window target skip_window ]
```

下面的代码就是构建训练集了，这里逻辑有点复杂，以注释来讲解。顺便举个例子，还是以[我，的，爱好，是，唱，跳，rap]为例子，假设这几个词在词典中的index为[1,2,3,4,5,6,7]，num_skips为3，skip_window为4，如果target是爱好，那么候选集就是[我，的，爱好，是，唱]对应的index为[1,2,3,4,5]，根据下面的代码，context_words为[1,2,4,5]，words_to_use是随机进行选择num_skips个数，这里假设选到了[1,2,5]三个数，接下来就是遍历words_to_use的数值，并和target组成训练集，最后针对爱好这个词的数据集为batch：[3，3，3]，label：[1，2，5]，然后继续循环for i in range(batch_size // num_skips)

```python
    # 构建一个队列，长度是span
    buffer = collections.deque(maxlen=span)  # pylint: disable=redefined-builtin
    # data_index 表示的是对应的word的下标，从0开始，如果下标+span已经大于数据的长度了，就把index设置为0重新遍历
    if data_index + span > len(data):
        data_index = 0
    # 把数据添加到队列里
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    # 要循环batch_size // num_skips次才能把batch填满，每次填num_skips个数
    for i in range(batch_size // num_skips):
        # 确认context的位置index
        context_words = [w for w in range(span) if w != skip_window]
        # 随机选择num_skips个位置，并返回index
        words_to_use = random.sample(context_words, num_skips)
        # 遍历context的index并与target组合在一起
        for j, context_word in enumerate(words_to_use):
            # batch保存的是输入数据，因为是skip-gram，所以把target放进去，
            batch[i * num_skips + j] = buffer[skip_window]
            # labels保存输出，根据words_to_use中的下标context_word去获取context word
            labels[i * num_skips + j, 0] = buffer[context_word]
        # 如果已经遍历到了数据的尾部，就重新把头的数据添加到队列里，否则buffer添加一个元素，队列中首的数据挤掉，尾部加入新数据，index+1，
        if data_index == len(data):
            buffer.extend(data[0:span])
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]], '-', labels[i, 0],
reverse_dictionary[labels[i, 0]])
```

接下来的代码就是模型的构建了，代码很简单，就是一个embedding层，不过损失函数是nce_loss噪声对比估算损失。为什么不用交叉熵损失呢，主要原因是模型采用了负采样来加速模型训练，负采样其实就是正样本与负样本的极大似然估计，这里不做深入讲解，不清楚的小伙伴建议google一下把这里弄清楚，其中`embeddings `是嵌入矩阵，即我们要求的词向量，`nce_weights`是计算逻辑回归的权重，`nce_biases`是偏移量

```python
# Step 4: Build and train a skip-gram model.   

batch_size = 256 
embedding_size = 128 
skip_window = 1 
num_skips = 2 
num_sampled = 64 

valid_size = 16
valid_window = 100 
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

graph = tf.Graph()

with graph.as_default():

    # Input data.
    with tf.name_scope('inputs'):
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
    with tf.name_scope('embeddings'):
            embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # Construct the variables for the NCE loss
        with tf.name_scope('weights'):
            nce_weights = tf.Variable(
                tf.truncated_normal([vocabulary_size, embedding_size],
        stddev=1.0 / math.sqrt(embedding_size)))
        with tf.name_scope('biases'):
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
```

tensorflow为服采用提供了封装好的损失函数即nce_loss，其api如下
```python
def nce_loss(weights, biases, inputs, labels, num_sampled, num_classes,
             num_true=1,
             sampled_values=None,
             remove_accidental_hits=False,
             partition_strategy="mod",
             name="nce_loss")
```
对应的参数说明如下
*   weight.shape = (N, K)
*   bias.shape = (N)
*   inputs.shape = (batch_size, K)
*   labels.shape = (batch_size, num_true)
*   num_true : 实际的正样本个数
*   num_sampled: 采样出多少个负样本
*   num_classes = N
*   sampled_values: 采样出的负样本，如果是None，就会用不同的sampler去采样
*   remove_accidental_hits: 如果采样时不小心采样到的负样本刚好是正样本，要不要去掉
*   partition_strategy：对weights进行embedding_lookup时并行查表时的策略。TF的embeding_lookup是在CPU里实现的，这里需要考虑多线程查表时的锁的问题。

后续的代码就是loss的编写，随机梯度下降法进行训练，测试代码编写等，不再赘述。
```python
      tf.name_scope('loss'):
            loss = tf.reduce_mean(
                tf.nn.nce_loss(
                    weights=nce_weights,
                    biases=nce_biases,
                    labels=train_labels,
                    inputs=embed,
                    num_sampled=num_sampled,
                    num_classes=vocabulary_size))
  
      # Add the loss value as a scalar to summary.
      tf.summary.scalar('loss', loss)
  
      # Construct the SGD optimizer using a learning rate of 1.0.
      with tf.name_scope('optimizer'):
          optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
  
      # Compute the cosine similarity between minibatch examples and all
      # embeddings.  
      norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
      normalized_embeddings = embeddings / norm
      valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
                                                valid_dataset)
      similarity = tf.matmul(
          valid_embeddings, normalized_embeddings, transpose_b=True)
  
      # Merge all summaries.
      merged = tf.summary.merge_all()
  
      # Add variable initializer.
      init = tf.global_variables_initializer()
  
      # Create a saver.
      saver = tf.train.Saver()
```

最后给出tensorflow官方对于[字词的向量表示法](https://tensorflow.google.cn/tutorials/representation/word2vec)的讲解

官方的代码输入数据是一个整段的文本，如果对于一行一行的文本数据，需要对数据处理部分进行代码的修改，如果您不知道该如何处理，不妨参考下[我的代码](https://github.com/terrifyzhao/text_matching/blob/master/w2v.py)
