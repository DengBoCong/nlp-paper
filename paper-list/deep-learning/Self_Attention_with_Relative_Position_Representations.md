# 前言

> 标题：Self-Attention with Relative Position Representations\
> 原文链接：[Link](https://arxiv.org/pdf/1803.02155.pdf)\
> Github：[NLP相关Paper笔记和实现](https://github.com/DengBoCong/nlp-paper)\
> 说明：阅读论文时进行相关思想、结构、优缺点，内容进行提炼和记录，论文和相关引用会标明出处，引用之处如有侵权，烦请告知删除。\
> 转载请注明：DengBoCong

# 介绍
了解Transformer的都知道，与递归和卷积神经网络相反，它没有在其结构中显式地建模相对或绝对位置信息，而是它需要在其输入中添加绝对位置的表示，这是一种完全依赖于注意力机制的方法。在本篇论文中，提出了一种替代方法，扩展了自注意机制，可以有效地考虑相对位置或序列元素之间距离的表示。本文描述了该方法的有效实现，并将其转换为可感知到任意图标记输入的相对位置感知自注意力机制的实例，即提出了一种将相对位置表示形式并入Transformer自注意机制的有效方法，残差连接有助于将位置信息传播到更高的层。

循环神经网络（RNN）通常根据时间 $t$ 的输入和先前的隐藏状态 $h_{t-1}$ 计算隐藏状态 $h_t$，直接通过其顺序结构沿时间维度捕获相对位置和绝对位置。非循环模型不必一定要顺序考虑输入元素，因此可能需要显式编码位置信息才能使用序列顺序。

一种常见的方法是使用与输入元素结合的位置编码，以将位置信息公开给模型。这些位置编码可以是位置的确定性函数或学习的表示形式。比如，卷积神经网络捕获每个卷积内核大小内的相对位置，已被证明仍然受益于位置编码。

# 相关
这里我直接将原Transformer的位置编码公式，self-attention以及多头注意力机制图贴出来，方便进行回忆和对比。
+ 位置编码公式
$$PE_{(pos,2i)}=sin(pos/10000^{2i/d_{model}})$$  $$PE_{(pos,2i+1)}=cos(pos/10000^{2i/d_{model}})$$
+ self-attention以及多头注意力机制
![在这里插入图片描述](https://img-blog.csdnimg.cn/202010251715376.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
# 具体结构
本文扩展自注意力以考虑输入元素之间的成对关系，从这个意义上讲，就是将输入建模为标记的(labeled)，有向的( directed)，完全连接的图( fully-connected graph)。在正式进入讲解之前，我们需要回顾一下self-attention。假设我们从多头注意力机制中的一个头输出后的序列是 $x=(x_1, ...,x_n)$，其中$x_i\in\mathbb{R}^{d_x}$，这个时候，我们需要通过attention计算出一个新的序列 $z=(z_1,...,z_n)$，其中 $z_i\in\mathbb{R}^{d_z}$。线性变换的输入元素的加权和计算公式如下：
$$z_i=\sum_{j=1}^na_{ij}(x_jW^V)（1）$$
其中，权重系数 $a_{ij}$是通过softmax计算的：
$$a_{ij}=\frac{exp(e_{ij})}{{\sum}_{k=1}^nexp(e_{ik})}$$
使用兼容函数计算 $e_{ij}$，该函数比较两个输入元素（其中，使用Scaled dot product作为兼容函数计算是很高效的）：
$$e_{ij}=\frac{(x_iW^Q)(x_jW^K)^T}{\sqrt{d_z}}（2）$$

###  Relation-aware自注意力
接下来说说Relation-aware自注意力，沿用上面的 $x$的表示，将输入元素 $x_i$ 和 $x_j$ 之间的edge表示为 $a_{ij}^V,a_{ij}^K\in\mathbb{R}^{d_a}$，学习两个不同的edge表示的出发点是 $a_{ij}^V$ 和 $a_{ij}^K$ 适用于等式如下两个等式，这些表示可以在关注头之间共享，其中 $d_a=d_z$。
首先，我们修改等式（1）将edge信息传播到子层输出：
$$z_i=\sum_{j=1}^na_{ij}(x_jW^V+a_{ij}^V)（3）$$
此扩展对于在任务中非常重要，其中任务由给定的注意头选择的edge类型信息对下游编码器或解码器层有用。同时，还修改了等式（2）确定兼容性时要考虑edge
$$e_{ij}=\frac{(x_iW^Q)(x_jW^K+a_{ij}^K)^T}{\sqrt{d_z}}（4）$$
这里将edge信息通过简单加法合并进表示的主要原因是为了高效实现，这在后面会讲到。

### 相对位置表示
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201025235302530.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)

对于线性序列，edge可以捕获有关输入元素之间相对位置差异的信息。我们考虑的最大相对位置被裁剪为最大绝对值 $k$，因为假设精确的相对位置信息在一定距离之外没有用。剪裁最大距离还使模型能够泛化训练期间看不到的序列长度，因此，考虑 $2k + 1$个唯一的edge标签。
$$a_{ij}^K=w_{clip(j-i,k)}^K$$  $$a_{ij}^V=w_{clip(j-i,k)}^V$$  $$clip(x,k)=max(-k,min(k,x))$$
通过上式就可以学习相对位置的表示$w^K=(w_{-k}^K,...,w_k^K)$和$w^V=(w_{-k}^V,...,w_k^V)$
### 高效实现
这里把等式（4）拆分开如下：
$$e_{ij}=\frac{x_iW^Q(x_jW^K)+x_iW^Q(a_{ij}^K)^T}{\sqrt{d_z}}（4）$$
然后我们就可以通过矩阵并行计算批量进入的数据了。
# 实验结果
使用newstest2014测试集，WMT 2014英语到德语（EN-DE）和英语到法语（EN-FR）翻译任务的实验结果。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201025235412728.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
改变剪切距离 $k$ 的实验结果。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201025235458897.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
相对位置表示 $a_{ij}^V$ 和 $a_{ij}^K$ 的实验结果
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201025235630450.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
# 总结
本文提出了自注意力的扩展，可用于合并序列的相对位置信息，从而提高了机器翻译的性能。论文中的这个思路可以借鉴参考，通过对自注意力的改造，就不需要进行硬位置编码了，但是论文中貌似没有比较硬位置编码和该方法的效果。