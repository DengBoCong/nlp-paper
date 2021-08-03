<font color=#999AAA >提示：阅读论文时进行相关思想、结构、优缺点，内容进行提炼和记录，论文和相关引用会标明出处。</font>

@[TOC](文章目录)

<hr style=" border:solid; width:100px; height:1px;" color=#000000 size=1">

# 前言

> [Language Modeling with Gated Convolutional Networks](https://arxiv.org/pdf/1612.08083.pdf)
> nlp-paper：[NLP相关Paper笔记和代码复现](https://github.com/DengBoCong/nlp-paper)
> nlp-dialogue：[一个开源的全流程对话系统，更新中！](https://github.com/DengBoCong/nlp-dialogue)
> text-similarity：[文本相似度（匹配）计算模型包](https://github.com/DengBoCong/text-similarity)
> 说明：阅读原文时进行相关思想、结构、优缺点，内容进行提炼和记录，原文和相关引用会标明出处，引用之处如有侵权，烦请告知删除。
> 转载请注明：DengBoCong

统计语言模型本质上是在给定前面若干个单词的条件下，通过概率建模来估计单词序列的概率分布，即：
$$P(w_0,...,W_N)=P(w_0)\prod_{i=1}^NP(w_i|w_0,...,w_{i-1})$$
比较典型的方法是N-gram语言模型，即当前位置只和前面N个位置的单词相关，但是这样也会带来问题即，N若小了，语言模型的表达能力不够，N若大了，遇到稀疏性问题时，无法有效的表征上下文。LSTM模型一般会将单词embedding到连续空间，然后输入进LSTM，从而有效的表征上下文，但LSTM的问题在于，作为递归模型，当前状态依赖于上一状态，并行化受到限制，导致运行速度非常慢。

既然RNNs很难通过硬件的并行化进行加速，那么自然而言会尝试使用卷积，卷积网络非常适合这种计算范式，因为所有输入词的计算可以同时进行。所以这篇文章吸收了LSTM的门控机制，将其应用在了卷积结构中，从而使得卷积模型保留非线性能力的同时，能够一定程度上减少梯度消失问题，而使其拥有更加深的结构。

# 结构细节
![在这里插入图片描述](https://img-blog.csdnimg.cn/bd5ca129a79e4a0ebda5c4a6b255d7f3.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70)
首先使用一个lookup table $D^{|v|\times e}$ 对输入序列进行embedding，其中 $|v|$ 表示单词表中的单词数量，$e$ 表示嵌入大小，每个单词 $w_i$ 在词嵌入查找表中都能找到一个向量表示 $E=D_{w_i}$，然后将序列表示E作为卷积层的输入，输入表示为 $X$：
$$h_l(X)=(X*W+b)\otimes \sigma(X*V+c)$$
其主要结构跟原始的CNN并无很大差异，只不过在卷积层引入了门控机制，将卷积层的输出变成了一个没有非线性函数的卷积层输出*经过sigmod非线性激活函数的卷积层输出。其中 $W$ 和 $V$ 是不同的卷积核，卷积核宽度为 $k$，输出通道数为 $n$，$b$ 和 $c$ 是偏置参数。$\sigma(X*V+c)$ 这一部分就是所谓的门控机制GLU，它控制着哪些信息可以传入下一层，这也使得卷积模型可以进行堆叠，以捕获Long Term Memory。

当然，除了GLU，还有一种被称为LSTM-style的门控机制，即GTU：
$$h_l(X)=tanh(X*W+b)\otimes \sigma(X*V+c)$$
不过从梯度的角度对两种门控单元进行分析，会发现GTU理论上会衰减的比较快，因为其梯度公式中包含两个衰减项，而GLU只有一个衰减项，可以较好地减轻梯度弥散，如下：
+ GLU：$\triangledown[X\otimes \sigma(X)]=\triangledown X\otimes \sigma(X)+X\otimes\sigma^{'}(X)\triangledown X$
+ GTU：$\triangledown[tanh(X)\otimes \sigma(X)]=tanh^{'}(X)\triangledown X\otimes \sigma(X)+\sigma^{'}(X)\triangledown X\otimes tanh(X)$

在模型预测时最简单的方法是使用一个 $softmax$ 层进行预测，但对于词量大的词表来说，在计算上显然不够有效率，本文中采用 $adaptive softmax$ ，其为高频词提供更高的容量，为低频词提供更低的容量。这在训练和测试时降低了内存需求以及提供了更快的计算速度。 

![在这里插入图片描述](https://img-blog.csdnimg.cn/7622e85d63d04785af056a6d2a472bea.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70)

# 实验结果
![在这里插入图片描述](https://img-blog.csdnimg.cn/9028821c64e54a4d802fc72ccef3eb91.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70)
Tanh是GTU去掉输出门部分后的模型，将其和GTU比较研究门限影响和贡献，从实验中对比可以看到GLU取得了最优的结果，如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/a7710e3ffdbe4ac1a477eb0965c6bd5c.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70)


# 总结
这里有几个小细节：
+ 论文使用的是宽卷积
>  Specifically, we zero-pad the beginning of the sequence with k−1 elements, assuming the first input element is the beginning of sequence marker which we do not predict and k is the width of the kernel
+ 对于文本长度更大的数据集而言，论文使用了更深的网络结构以获取其Long-Term记忆。

本篇论文做出的贡献如下：
+ 提出一种基于线性门控单元（Gated Linear Units）的卷积网络并将其运用于语言建模。GLU在保持一定非线性能力的同时通过为梯度提供线性的传播路径使得在深度架构中能够有效减少“梯度消失”的问题。
+ 在GBW数据集上证明了该卷积网络性能优于其他语言模型：如LSTMs，并在段落数据集WikiText-103上验证了该模型处理长距离依赖（long-range depenencies）的能力。
+ 证明了GLU比LSTM-style门控具有更高的准确度和更快的收敛速度。