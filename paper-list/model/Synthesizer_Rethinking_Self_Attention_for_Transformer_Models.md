# 前言

> [Synthesizer: Rethinking Self-Attention for Transformer Models](https://arxiv.org/pdf/2005.00743.pdf)\
> nlp-paper：[NLP相关Paper笔记和代码复现](https://github.com/DengBoCong/nlp-paper)\
> nlp-dialogue：[一个开源的全流程对话系统，更新中！](https://github.com/DengBoCong/nlp-dialogue)\
> text-similarity：[文本相似度（匹配）计算模型包](https://github.com/DengBoCong/text-similarity)\
> 说明：阅读原文时进行相关思想、结构、优缺点，内容进行提炼和记录，原文和相关引用会标明出处，引用之处如有侵权，烦请告知删除。\
> 转载请注明：DengBoCong

Transformer自从被提出，就火到现在，它的关键在于 query-key-value 的点积注意力，token与token之间被完全连接，能够对远距离的依赖关系进行建模。Transformer在到处都是黑箱的深度学习领域，可以说一个解释性比较强的模型之一了。而作为Transformer核心的组件，Self-Attention被许多人反复研究来研究去，随处可见的资料貌似已经把它解释的很好的，但事实真的这样么？本文对自注意力机制做了一些探索，里边的结果也许会颠覆我们对自注意力的认知。

# 前情提要
首先我们要明白点积的几何意义，两个向量越相似，他们的点积越大，Self-Attention结构就是利用Q，K，V计算点积，Self-Attention就是计算一个向量（可以理解为一个词）与其它向量的点积，即相似性。下面给出其公式：
$$Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V$$
注意了，完整的Self-Attention中，是对同一个 $X\in\mathbb{R}^{n\times d}$ 通过不同的投影矩阵 $W_q,W_k,W_v\in\mathbb{R}^{d\times d'}$得到$Q=XW_q,K=XW_k,V=XW_v$，即：
$$SelfAttention(X)=softmax(\frac{XW_qW_k^TX^T}{\sqrt{d_k}})XW_v$$

点积自注意力提供了强大的建模能力，其基本作用是确定单个 token 相对于序列中所有其他 token 的相对重要性。key、query、value 暗示自注意力模拟一个基于内容的检索过程，过程的核心是 pairwise 的交互。该文对整个过程进行了反思，通过探究如下几个问题：
+ 点自注意力机制是否真的那么重要？
+ 真的需要点注意力机制么？
+ 通过pairwise点积来获得注意力权重是否必要？

# 模型细节
本篇论文本质上是讨论在Transformer结构下，替换自注意力机制的相关研究，如下图所示：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210613233405832.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70)
为了方便后面理解，这里我统一一下注意力权重计算的形式：
$$A=softmax(B)$$  $$Y=AXW_v$$
其中，$A$ 就是在 $B$ 的sequence length维度上进行归一化后，计算出的注意力权重，所以本质上来看，自注意力就是通过一下一个 $n\times n$ 的矩阵 $A$ 和一个 $d\times d'$ 的$W_v$，将原本是 $n\times d$ 的输入矩阵 $X$ ，通过 $AXW_v$ 的计算得到一个 $n\times d'$ 的矩阵的过程。有了上面的理解定义之后，我们来看看Synthesizer Model的几种形式。

+ Dense Synthesizer：本质上我们需要输入 $n\times d$的 $X$ 来得到一个 $n\times n$ 的注意力权重矩阵，所以，如果不使用token-token，一种很平常的方式就是通过Dense来进行维度的变换。论文在实践中稍微将操作做得更复杂了一点，即：
$$B=relu(XW_1 + b_1)W_2+b_2$$
值得一提的是，在这一步操作中，sequence中的每个token互相之间都是独立计算的。
+ Random Synthesizer：既然上面我们是为了得到一个 $n\times n$ 的 $B$，使用了一个比较直观的方式即Dense进行计算，那么为什么我们不能使用一个近乎淳朴的方式，即直接初始化一个随机的 $n\times n$矩阵 $B$ 呢？Random Synthesizer就是这种方式：
$$B=R$$
上面的 $R$ 就是Random的意思， $B$ 是随机初始化得到的，可以选择随训练更新或不更新。据原论文描述，固定形式的Attention首次出现在论文《Fixed Encoder Self-Attention Patterns in Transformer-Based Machine Translation》，不同点是那里的Attention矩阵是由一个函数算出来的，而Google这篇论文则是完全随机初始化的。从形式上看，Random实际上就相当于可分离卷积（Depthwise Separable Convolution）运算。
+ Factorized Models：Dense只需要 $d\times N$ 的参数量，Random需要 $N\times N$ 的参数量，节省了原本 $Q$ 和 $K$ 做投影的参数量。但当sequence的长度比较长时，还是参数会很多，所以这里通过低秩分解来降低参数量。对于Dense和Random都有对应的低秩分解的形式，分别称为Factorized Dense和Factorized Random。Factorized Dense通过Dense的方式，生成两个 $n\times a,n\times b$ 的矩阵 $B1,B2$，其中$ab=n$；然后将 $B_1$ 重复 $b$ 次、然后将 $B_2$ 重复 $a$ 次，得到对应的 $n\times n$矩阵 $\tilde{B}_1,\tilde{B} 2$，最后将它们逐位相乘，合成一个 $n\times n$ 的矩阵：
$$B=\tilde{B}_1\odot\tilde{B}_2$$
至于Factorized Random就很好理解了，本来是一整个 $n\times n$ 的矩阵 $R$，现在变成两个 $n\times k$ 的矩阵 $R_1,R_2$：
$$B=R_1R_2^T$$
+ Mixture of Synthesizers：到目前为止，连同标准的自注意力，我们有5种不同的生成矩阵 $B$ 的方案，它们也可以混合起来，即：
$$B=\sum_{i=1}^Na_iB_i$$
其中，$B_i$ 是不同形式的自注意力矩阵，而 $\sum_{i=1}^Na_i=1$ 是可学习参数。

如下是对于上面讲到的方法和原方法的复杂度对比如下图：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210614120602783.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70)
# 实验
+ 机器翻译
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210614204149371.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70)
+ 摘要对话
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210614204436986.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70)
+ 预训练+微调
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210614204547470.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70)

# 总结
+ 提出了Synthetic Attention，无需点注意力或基于内容的注意力。
+ 提出了Synthesizer，集成了Synthetic Attention的新模型。
+ 随机对齐的可学习矩阵一样表现良好。在一些任务重token-token依赖并不是必须的