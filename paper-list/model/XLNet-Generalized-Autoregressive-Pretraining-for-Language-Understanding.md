
# 前言

> 标题：XLNet: Generalized Autoregressive Pretraining for Language Understanding
> 原文链接：[Link](https://arxiv.org/pdf/1906.08237.pdf)
> nlp-paper：[NLP相关Paper笔记和代码复现](https://github.com/DengBoCong/nlp-paper)
> nlp-dialogue：[一个开源的全流程对话系统，更新中！](https://github.com/DengBoCong/nlp-dialogue)
> 说明：阅读原文时进行相关思想、结构、优缺点，内容进行提炼和记录，原文和相关引用会标明出处，引用之处如有侵权，烦请告知删除。
> 转载请注明：DengBoCong

Google发布的XLNet在问答、文本分类、自然语言理解等任务上都大幅超越BERT，XLNet提出一个框架来连接语言建模方法和预训练方法。我们所熟悉的BERT是denoising autoencoding模型，最大的亮点就是能够获取上下文相关的双向特征表示，所以相对于标准语言模型（自回归）的预训练方法相比，基于BERT的预训练方法具有更好的性能，但是这种结构同样使得BERT有着它的缺点：
+ 生成任务表现不佳：预训练过程和生成过程的不一致，导致在生成任务上效果不佳；
+ 采取独立性假设：没有考虑预测[MASK]之间的相关性（位置之间的依赖关系），是对语言模型联合概率的有偏估计（不是密度估计）；
+ 输入噪声[MASK]，造成预训练-精调两阶段之间的差异；
+ 无法适用在文档级别的NLP任务，只适合于句子和段落级别的任务；

鉴于这些利弊，作者提出一种广义自回归预训练方法XLNet，该方法：
+ enables learning bidirectional contexts by maximizing the expected likelihood over all permutations of the factorization orde
+ overcomes the limitations of BERT thanks to its autoregressive formulation

# 前情提要
首先在此之前需要了解一下预训练语言模型的相关联系和背景，这里推荐两篇文章，一篇是邱锡鹏老师的关于NLP预训练模型的总结Paper：[Pre-trained Models for Natural Language Processing: A Survey](https://arxiv.org/pdf/2003.08271.pdf)，我之前对它有写过阅读笔记：[论文阅读笔记：超详细的NLP预训练语言模型总结清单！](https://zhuanlan.zhihu.com/p/352152573)，还有一篇就是：[nlp中的预训练语言模型总结(单向模型、BERT系列模型、XLNet)](https://zhuanlan.zhihu.com/p/76912493)，其中总结的也相当的全面精辟到位。

目前无监督表示学习这一块，自回归（autogression）语言建模和自动编码（autoencoding）无疑是最成功的两个。对于ELMO、GPT等预训练模型都是基于传统的语言模型（自回归语言模型AR），自回归语言模型天然适合处理生成任务，但是无法对双向上下文进行表征，因此人们反而转向自编码思想的研究（如BERT系列模型）。

那AE就完美了嘛？自编码语言模型（AE）虽然可以实现双向上下文进行表征，但是依旧存在不适于生成任务的问题，就和上面说的BERT的缺点一样，以BERT为代表的系列模型：
+ BERT系列模型引入独立性假设，没有考虑预测[MASK]之间的相关性；
+ MLM预训练目标的设置造成预训练过程和生成过程不一致；
+ 预训练时的[MASK]噪声在finetune阶段不会出现，造成两阶段不匹配问题；

对于AE和AR两种模型在各自的方向优点，有什么办法能构建一个模型使得同时具有AR和AE的优点并且没有它们缺点呢？这也是XLNet诞生的初衷，对于XLNet：
+ 不再像传统AR模型中那样使用前向或者反向的固定次序作为输入，XLNet引入排列语言模型，采用排列组合的方式，每个位置的上下文可以由来自左边和右边的token组成。在期望中，每个位置都要学会利用来自所有位置的上下文信息，即，捕获双向上下文信息。
+ 作为一个通用的AR语言模型，XLNet不再使用data corruption，即不再使用特定标识符号[MASK]。因此也就不存在BERT中的预训练和微调的不一致性。同时，自回归在分解预测tokens的联合概率时，天然地使用乘法法则，这消除了BERT中的独立性假设。
+ XLNet在预训练中借鉴了Transformer-XL中的segment recurrence机制的相对编码方案，其性能提升在长文本序列上尤为显著。
+ 由于分解后次序是任意的，而target是不明确的，所以无法直接使用Transformer-XL，论文中提出采用“reparameterize the Transformer(-XL) network”以消除上述的不确定性。

# 排列语言模型
受无序NADE（Neural autoregressive distribution estimation）的想法的启发，提出一个排列组合语言模型，该模型能够保留自回归模型的优点，同时能够捕获双向的上下文信息。例如一个长度为T的序列，其排序组合为T!，如果所有排列组合次序的参数共享，那么模型应该会从左右两个方向的所有位置收集到信息。但是由于遍历 T! 种路径计算量非常大（对于10个词的句子，10!=3628800）。因此实际只能随机的采样 T! 里的部分排列，并求期望；
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210401224809488.png#pic_center)

为了更好的理解，看下面这张图：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210401225106603.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
假设输入的序列是[1,2,3,4], 排列共有4x3x2=24种，选其中的四种分别为[3,2,4,1],[2,4,3,1],[1,4,2,3],[4,3,1,2]。在预测位置3的单词时，第一种排列看不到任何单词，第二种排列能看到[2,4]，第三种排列能看到[1,2,4]，第四种排列能看到[4]，所以预测位置3的单词时，不仅能看到上文[1,2]，也能看到下文的[4]，所以通过这种方式，XLNet模型能同时编码上下文信息。

> PLM的本质就是LM联合概率的多种分解机制的体现，将LM的顺序拆解推广到随机拆解，但是需要保留每个词的原始位置信息（PLM只是语言模型建模方式的因式分解/排列，并不是词的位置信息的重新排列！）

但是有个问题需要注意，上面提出的排列语言模型，在实现过程中，会存在一个问题，举个例子，还是输入序列[1, 2, 3, 4]肯定会有如下的排列[1, 2, 3, 4]，[1,2,4,3]，第一个排列预测位置3，得到如下公式 $P(3|1,2)$，第二个排列预测位置4,得到如下公式 $P(4|1,2)$，这会造成预测出位置3的单词和位置4的单词是一样的，尽管它们所在的位置不同。论文给出具体的公式解释如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021040122582989.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
那怎么解决没有目标(target)位置信息的问题？那就是下面要讲的Two-Stream Self-Attention。

# Two-Stream Self-Attention
除了上述之外，模型的实现过程中还有两点要求
+ 在预测当前单词的时候，只能使用当前单词的位置信息，不能使用单词的内容信息。
+ 在预测其他单词的时候，可以使用当前单词的内容信息

为了满足同时这两个要求，XLNet提出了双流自注意力机制，结构如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210401233224610.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
> 下文用 $g_{z_t}$ 表示，上下文的内容信息用 $x_{z<t}$ 表示，目标的位置信息 $z_t$ ，目标的内容信息 $x_{z_t}$

+ content stream：上面图(a)中，$h_{z_t}^{(m)}\leftarrow Attention(Q=h_{z_t}^{(m-1)},KV=h_{z<t}^{(m-1)};\theta)$，预测其他单词时，使用自己的内容信息 $h_1^{(0)}$，即Content 流主要为 Query 流提供其它词的内容向量，包含位置信息和内容信息
+ query stream：上面图(b)中，$g_{z_t}^{(m)}\leftarrow Attention(Q=g_{z_t}^{(m-1)},KV=h_{z<t}^{(m-1)};\theta)$，预测当前单词时，不能使用当前单词内容信息 $h_1^{(0)}$，Query 流就为了预测当前词，只包含位置信息，不包含词的内容信息；
+ 总流程：上图(c)中，首先，第一层的查询流是随机初始化了一个向量即 $g_i^{(0)}=w$，内容流是采用的词向量即 $h_i^{(0)}=e(x_i)$，self-attention的计算过程中两个流的网络权重是共享的，最后在微调阶段，只需要简单的把query stream移除，只采用content stream即可。

# 集成Transformer-XL
除了上文提到的优化点，作者还将transformer-xl的两个最重要的技术点应用了进来，即相对位置编码与片段循环机制。我们先看下recurrence mechanism（不采用BPTT方式求导）。
+ 前一个segment计算的representation被修复并缓存，以便在模型处理下一个新的segment时作为扩展上下文resume；
+ 最大可能依赖关系长度增加了N倍，其中N表示网络的深度；
+ 解决了上下文碎片问题，为新段前面的token提供了必要的上下文；
+ 由于不需要重复计算，Transformer-XL在语言建模任务的评估期间比vanilla Transformer快1800+倍；

bert的position embedding采用的是绝对位置编码，但是绝对位置编码在transformer-xl中有一个致命的问题，因为没法区分到底是哪一个片段里的，这就导致了一些位置信息的损失，这里被替换为了transformer-xl中的相对位置编码。假设给定一对位置 $i$ 和 $j$ ，如果 $i$ 和 $j$ 是同一个片段里的那么我们令这个片段编码 $s_{ij}=s_{+}$，如果不在一个片段里则令这个片段编码为 $s_{ij}=s_{-}$，这个值是在训练的过程中得到的，也是用来计算attention weight时候用到的，在传统的transformer中attention weight=$Softmax(\frac{Q\cdot K}{d}V)$，在引入相对位置编码后，首先要计算出 $a_{ij}=(q_i+b)^Ts_{sj}$，其中 $b$也是一个需要训练得到的偏执量，最后把得到的 $a_{ij}$与传统的transformer的weight相加从而得到最终的attention weight。

关于相对位置编码更详细的描述可以参考这篇文章：[Transformer改进之相对位置编码(RPE)](https://zhuanlan.zhihu.com/p/105001610)

# 总结

XLNet预训练阶段和BERT差不多，不过去除了Next Sentence Prediction，作者发现该任务对结果的提升并没有太大的影响。输入的值还是 [A, SEP, B, SEP, CLS]的模式，A与B代表的是两个不同的片段。更详细的实现细节可以参考[论文源码](https://github.com/zihangdai/xlnet)。

XLNet的创新点：
+ 仍使用自回归语言模型，未解决双向上下文的问题，引入了排列语言模型
+ 排列语言模型在预测时需要target的位置信息，为此引入了Two-Stream:Content流编码到当前时刻的所有内容，而Query流只能参考之前的历史信息以及当前要预测的位置信息
+ 为了解决计算量大的问题，采取随机采样语言排列+只预测1个句子后面的 $\frac{1}{K}$ 的词
+ 融合Transformer-XL的优点，处理过长文本