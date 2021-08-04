
<font color=#999AAA >提示：阅读论文时进行相关思想、结构、优缺点，内容进行提炼和记录，论文和相关引用会标明出处。</font>

@[TOC](文章目录)

<hr style=" border:solid; width:100px; height:1px;" color=#000000 size=1">

# 前言

> [Pay Less Attention With Lightweight And Dynamic Convolutions](https://arxiv.org/pdf/1901.10430.pdf)
> nlp-paper：[NLP相关Paper笔记和代码复现](https://github.com/DengBoCong/nlp-paper)
> nlp-dialogue：[一个开源的全流程对话系统，更新中！](https://github.com/DengBoCong/nlp-dialogue)
> text-similarity：[文本相似度（匹配）计算模型包](https://github.com/DengBoCong/text-similarity)
> 说明：阅读原文时进行相关思想、结构、优缺点，内容进行提炼和记录，原文和相关引用会标明出处，引用之处如有侵权，烦请告知删除。
> 转载请注明：DengBoCong

Self-Attention能够将每个元素和当前时刻元素进行比较来确定上下文元素的重要性，这也使得它在NLP模型中表现优异。而本篇文章则是将卷积结构与Self-Attention结构进行了比较，通过 实验证明了这样的卷积结构同样有着高效的计算和足以和Self-Attention媲美的效果。本篇文章所述的卷积结构是基于non-separable convolutions和depthwise separable convolutions，不清楚深度可分离卷积的小伙伴可以参考这篇文章：[深度可分离卷积](https://zhuanlan.zhihu.com/p/92134485)

![在这里插入图片描述](https://img-blog.csdnimg.cn/c5d87ac909c848af9cf8bbbceaf2820c.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70)

Self-Attention被定义为基于内容的表示，如上图a中所示，其中通过将当前时刻与上下文中的所有元素进行比较来计算注意力权重，这种不受上下文大小限制的计算比较能力，也被视为Self-Attention的核心能力。当然，这种能力也是需要付出代价的，就是计算复杂度是输入长度的二次方，这也使得在相对较长的文本中进行计算成本变得非常的高。

Dynamic convolutions基于lightweight convolutions 构建的，其每个时刻预测不同的卷积核，也就是说卷积核只学习当前时间步的相关信息，而不是学习全局信息。动态卷积在轻量卷积的基础之上，增加了一个可学习的参数单元来影响时间步的权重，这有点类似于局部Attention，只不过相较来说没有考虑前一时刻的权重信息。
# 结构细节
这篇[文章](https://qiita.com/koreyou/items/328fa92a1d3a7e680376)对几种卷积的关联进行了可视化的阐述。
![在这里插入图片描述](https://img-blog.csdnimg.cn/94f0ccad9b4e4015aedea9801343f04d.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70)
其中的GLU结构可以参考这篇[文章](https://zhuanlan.zhihu.com/p/395977833)。LConv(LightConv)就是基于DepthwiseConv进行计算得到的，如序列中的第 $i$ 个元素和输出通道 $c$ 公式推导如下：
$$DepthwiseConv(X, W, i, c) = \sum_{j=1}^k W_{c,j} \cdot X_{(i+j-\lceil \frac{k+1}{2} \rceil),c}$$
$$LightConv(X,W_{\left \lceil \frac{cH}{d} \right \rceil,:},i,c)=DepthwiseConv(X,softmax(W_{\left \lceil \frac{cH}{d} \right \rceil,:}),i,c)$$
###  Lightweight Convolutions
+ Weight sharing：将 $H$ 个通道的参数进行共享，这将参数数量减少了$\frac{d}{H}$倍。 如上图2所示，常规卷积需要 $d^2\times k$ 个权重，深度可分离卷积则需要 $d\times k$ 个权重，而通过权重共享，轻量卷积只需要 $H \times k$个权重，这大大减少了模型的参数量。
+ Softmax-normalization：对时间维度 $k$ 上的权重$W\in\mathbb{R}^{H\times k}$，使用softmax运算进行归一化：
$$softmax(W)_{h,j}=\frac{expW_{h,j}}{\sum_{j'=1}^{k}expW_{h,j'}}$$
相当于归一化每个词的每一维的的重要性（比self-attention更精细）。
+ Module：图2b展示集成LightConv的模块的网络结构，首先通过Linear层将维度 $d$ 到 $2d$ 的输入投影映射，然后是门控线性单位（GLU），以及LightConv。GLU通过应用sigmoid单元将输入的一半用作门，然后与另一半输入计算逐点乘积。并将大小为 $W^O\in \mathbb{R}^{d\times d}$ 的输出投影应用于LightConv的输出。
+ Regularization：DropConnect是LightConv模块的良好正则化工具。具体来说，以概率 $p$ 丢弃归一化权重 $softmax(W)$ 的每一项，并在训练过程中将其除以 $1-p$。这等于去除了通道内的一些temporal信息。
+ 实现小细节：为了更好的适应原始CUDA计算，将归一化的权重 $W\in\mathbb{R}^{H\times k}$ 复制并扩展到大小为 $BH\times n \times n$ 的带状矩阵，其中 $B$ 是批量大小。然后，将输入整形并转置为大小 $BH\times n\times \frac{d}{H}$ ，并执行批处理矩阵乘法以获取输出。

### Dynamic Convolutions
动态卷积具有随时间而变化的卷积核，这些卷积核随各个时刻的学习函数而变化。由于当前的GPU需要大量内存，因此对于当前的GPU而言，在动态卷积中使用标准卷积是不切实际的。 所以这里通过在LightConv上进行构建来解决此问题，该方法大大减少了参数的数量。

DynamicConv的形式与LightConv相同，但使用的是随时间变化的卷积核，该卷积核使用函数 $f:\mathbb{R}^d\rightarrow\mathbb{R}^{H\times k}$：
$$DynamicConv(X,i,c)=LightConv(X,f(X_i)_{h,:},i,c)$$
我们用具有可学习权重 $W^Q\in\mathbb{R}^{H\times k\times d}$ 的简单线性模块对 $f$ 建模，即$f(X_i)=\sum_{c=1}^dW_{h,j,c}^QX_{i,c}$。与self-attention类似，DynamicConv会随着时间更改分配给上下文元素的权重。但是，DynamicConv的权重并不取决于整个上下文，它们仅是当前时刻的函数。self-attention需要在句子长度中进行二次运算来计算注意力权重，而DynamicConv的动态卷积核的计算随序列长度进行线性缩放。

# 实验结果
![在这里插入图片描述](https://img-blog.csdnimg.cn/fe7623307e8c4ac685518e332ec65b8d.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70)
# 总结
这项研究还是很有价值的，因为它从某种程度上证明了机器翻译可以在没有自注意力（例如 Transformer）的情况下实现高精度（毕竟这些年来自注意力机制被认为是机器翻译必不可少的），而且论文的方法很简单，从计算时间上来说，该方法非常值得推荐的。而且在多个任务中也同样取得了非常好的性能，这篇文章提供了一个视角，为后续的相关研究提供了有力的助力。
