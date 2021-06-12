# 前言

> [Are Pre-trained Convolutions Better than Pre-trained Transformers?](https://arxiv.org/pdf/2105.03322.pdf)
> nlp-paper：[NLP相关Paper笔记和代码复现](https://github.com/DengBoCong/nlp-paper)
> nlp-dialogue：[一个开源的全流程对话系统，更新中！](https://github.com/DengBoCong/nlp-dialogue)
> text-similarity：[文本相似度（匹配）计算模型包](https://github.com/DengBoCong/text-similarity)
> 说明：阅读原文时进行相关思想、结构、优缺点，内容进行提炼和记录，原文和相关引用会标明出处，引用之处如有侵权，烦请告知删除。
> 转载请注明：DengBoCong

Transformer诞生到现在，从NLP领域到CV领域，可以说是两开花。特别是在预训练模型中，BERT相关系列近些年屡屡突破，在各种下游任务中，不仅能提速还有效果上的提升。所以在NLP的相关任务中，提及Transformer和CNN时，Transformer一般都会优先考虑，更何况是在预训练语言模型方面，我以前都没有想过将CNN用在预训练，直到看到这篇文章，才打开了新思路，看来还是我格局小了呀。

正如论文标题一样：Are Pre-trained Convolutions Better than Pre-trained Transformers?这篇文章并没有能够将“CNN预训练优于Transformer预训练”这个结论石锤，不过从某种程度上说，还是将BERT、transformers和大规模预训练模型进行解耦，给我们打开了新世界，接下来我们就一起来品一品这篇文章。

# 前情提要
这篇文章其实围绕三个问题要讨论：
+ 只有类Transformers的结构才适合预训练？
+ 如果使用不同于Transformers结构的模型来进行预训练，是否能够提高收益？
+ 使用卷积进行预训练是否在特定的场景表现更好？

在正式研究和讨论之前，还有几点需要达成共识的，根据以往的研究表明，卷积有着如下的优势：
+ CNN 比 self-attention 快得多：CNN 是线性复杂度，self-attention 是平方复杂度（甚至因此诞生了《轻量 transformers》这个分支领域）。
+ CNN 是按顺序进行的，因此不需要如Transformers那样，需要额外的位置编码。

不过还是需要注意的是，CNN 在单层的感受野大小是有限且固定的，只能通过堆叠层数来增大感受野，而self-attention 在一层就可以捕捉所有 token 之间的关系，这对于捕捉长距离依赖非常关键。同时，self-attention 聚合的权重是与输入 token 相关的，而 CNN 的聚合权重是与输入 token 无关的。

文章涉及到对比卷积的运行速度，我之前写过一篇如何根据FLOPs或MACC去大致的计算模型的速度，感兴趣的可以参考如下：
[教你如何估计各种神经网络的计算量和参数量](https://zhuanlan.zhihu.com/p/342668070)

我们来简单过一遍论文中使用到的卷积：
+ Depthwise Convolutions：深度可分离卷积中，每个通道只被一个卷积核所卷积，这里我们假设输入的张量 $X$ 的维度大小为 $n\times d$，那么深度可分离卷积 $D(X,W_{c:},i,c)$ 可以被定义为：
$$O_{i,c}=\sum_{j-1}^k W_{c,j}\cdot X_{i+j-\left \lceil \frac{k+1}{2} \right \rceil},c$$
其中，$W\in \mathbb{R}^{d\times k}$是可训练参数，$O_{i,c}$是通道 $c$的第$i$ 个位置的输出，输出的shape和输入的shape相同，都是 $n\times d$
+ Lightweight Convolutions：轻量化卷积对深度可分离卷积做了进一步地简化，这里我们令 $L(.)$ 是深度可分离卷积，并做了softmax处理，如下：
$$O_{i,c}^L=\sum_{j-1}^k softmax(W_{\hat{c},j})\cdot X_{i+j-\left \lceil \frac{k+1}{2} \right \rceil},\hat{c}$$
其中，$\hat{c}=\frac{cH}{d}$ ，也就是说每个 $\frac{d}{H}$ 的输出通道共享参数，其中 $H=1$，这是相当于共享所有通道的所有权重。简单来说，（1）相邻通道的卷积核可进行参数共享；（2）卷积核参数在其长度的维度上被 softmax 归一化：
$$softmax(W_{\hat{c},j})=\frac{expW_{\hat{c},j}}{\sum_{j=1}^kexpW_{\hat{c},j'}}$$
其中，$H$ 和 $k$ 分别是卷积核的数量，以及卷积核的长度。
+ Dynamic Convolutions：动态卷积是对轻量化卷积的进一步改进，如下：
$$D_Y=L(X,f(X_i)_{h,:},i,c)$$
其中，$f(.)$ 是线性变换，用于学习卷积核的生成与其顺序输入的不同token $X$ 的关系，其参数可表示为 $W^Q\in \mathbb{R}^{H\times k\times d}$。通过这种方式，使得卷积核 $W$ 的生成与其顺序输入的不同 token $X$ 有关，而不是对整个文本序列固定的卷积核。而且，这里的卷积核参数只与当前被卷积核覆盖的几个 token 相关，而不像 self-attention 那样，需要与全部 token 交互计算。因此整体上，动态卷积还是线性复杂度。

对上面三种卷积想要图形化理解的话，可以参考这篇文章：[論文紹介: Pay Less Attention with Lightweight and Dynamic Convolutions](https://qiita.com/koreyou/items/328fa92a1d3a7e680376)

# 预训练细节
在开始细节前，我们来感受一下原文的描述：
> We implement a Seq2Seq (Sutskever et al., 2014) architecture similar to (Wu et al., 2019). The key difference when compared with Transformer architectures is that we replace the multi-headed selfattention with convolutional blocks. Instead of query-key-value transforms, we use gated linear unit projections following (Wu et al., 2019).

说白了，本文的卷积预训练模型结构依然在模仿基于 transformers 的预训练模型结构，只不过是将其中的 multi-head self-attention 换成了上面说的卷积操作，query-key-value 的结构换成了类似的线性门控（Gated Linear Units）结构。每个 convolution block 的结构如下图所示（图源来自[PAY LESS ATTENTION WITH LIGHTWEIGHT AND DYNAMIC CONVOLUTIONS](https://arxiv.org/pdf/1901.10430.pdf)）：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210612123558454.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70)
我们可以从上图看出，论文所用的卷积是一种“线性门控 + 卷积 + 线性映射”的结构，可以表示为：
$$X^1=W^IX\odot sigmoid(W^SX)$$   $$X^2=ConvBlock(X^2)$$    $$X^3=W^O(X^2)$$
其中，$W^I,W^S,W^O$都是可训练参数，论文中实验了轻量化卷积，动态卷积和空洞卷积。

对于整体的卷积预训练模型，本文也是使用类似 transformers 的方式将 convolution blocks 进行堆叠：
$$X_A=LayerNorm(Conv(X))+X$$    $$X_B=LayerNorm(FFN(X_A))+X_A$$

其中，Conv表示实验的卷积，FFN则是用ReLU的前馈神经网络，看起来是不是简直就是Transformer，哈哈哈。

训练模型时，损失使用的是token-wise的交叉熵，并通过 teacher forcing进行训练，如下：
$$L=\sum_{t=1}^L\sum_{i=1}^nlog(\pi_i^t)+(1-y_i^t)log(1-\pi_i^t)$$
其中，$\pi_i^t$ 在 $t$ 时刻对类别 $i$ 预测值，而 $y_i^t$ 是类别 $i$ 在时刻 $t$ 的真实标签值。

# 实验
模型在 Colossal Cleaned CommonCrawl Corpus (C4) 数据集上进行了预训练。预训练时，模型的 seq2seq 的结构、MLM 任务依然是模拟 transformers；层数、序列长度等参数也与 BART-base 保持了一致。

在实验部分，这篇文章希望探究如下五个问题：

+ 卷积也能在预训练中获益，学到丰富的先验知识吗？
+ 卷积预训练和 transformers 相比，性能怎么样？
+ 卷积预训练和 transformers 相比，有什么优点？会更快吗？
+ 什么情景下，卷积预训练会失败？
+ 不同的卷积模块之间，有很大的差别吗？

下图是在一些下游实验中，几个模型的对比：![在这里插入图片描述](https://img-blog.csdnimg.cn/20210612165543865.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70)
上面的实验可以回答提出的几个问题：
+ 卷积网络也能在预训练中获益，只是不如 transformers 大。
+ 无论是否预训练，卷积的性能优于或与 transformers 一致。
+ 空洞卷积和动态卷积似乎好于轻量化卷积。

作者在实验中发现，与训练卷积结构缺少相互的 attention 结构，因此在需要构建多个事物之间关系的任务上，卷积预训练结构似乎并不适合。另外，卷积预训练模型更快，因此能被运用到更长的序列。随着序列长度的增加，卷积预训练模型的速度优势将更加显著：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210612165939278.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70)

# 总结
这篇论文给我们的主要启发是：预训练改进与模型改进不应该混为一谈，预训练技术本身往往能给各种模型都带来提升，不应该一提到预训练就想到Transformer，也不应该只把预训练跟Transformer结合。这篇论文给出了一个新的视野，但是并不能实质性的影响类Transformer在预训练中作用，毕竟CNN有着根本的缺陷，即无法捕捉足够远的长程依赖，虽然通过膨胀卷积等方式，可以快速增大CNN的感受野，但也只是比较大，不是Transformer理论上的一步到位。其次，如果单纯看提高效率角度，Transformer本身也有很多优化空间，如果只是为了执行效率而转向CNN，那这个理由似乎不那么有说服力。