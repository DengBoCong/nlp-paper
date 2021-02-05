
# 前言

> 标题：Multi-Cast Attention Networks for Retrieval-based Question Answering and Response Prediction\
> 原文链接：[Link](https://arxiv.org/pdf/1806.00778.pdf)\
> Github：[NLP相关Paper笔记和代码复现](https://github.com/DengBoCong/nlp-paper)\
> 说明：阅读论文时进行相关思想、结构、优缺点，内容进行提炼和记录，论文和相关引用会标明出处，引用之处如有侵权，烦请告知删除。\
> 转载请注明：DengBoCong

在基于检索式对话系统中，总体上可以分为关键的两个部分，分别是文本特征表示和文本相关性建模，核心思想就是通过评分模型，输出一个排序的候选响应列表，并在其中选择最佳响应（也就是所谓的神经排序模型）。说到文本相关性，我们就很容易想到当下非常受欢迎的注意力机制，注意力的关键思想是仅提取对预测有用的最相关信息，在文本数据的上下文中，注意力学习根据文档中的单词和子短语的重要性来对它们进行加权，有关深度学习中注意力机制的相关总结和复现可以看我这一篇文章：[NLP中遇到的各类Attention结构汇总以及代码复现](https://zhuanlan.zhihu.com/p/338193410)。

论文中提到，注意力机制通常情况下用来作为特征提取器，它的行为可以被认为是一种动态的pooling形式，因为它学习选择和组合不同的词来形成最终的文档表示。而本文中换了一种思路，将attention作为一种特征增强方法（也就是说Attention的目的不是组合学习，而是为后续层提供提示特征），这怎么理解呢？不着急，我们接着往下看。

通常情况下，我们只在一个句子上施加一次注意力，然后将学习到的表示传递给后续的预测网络进行学习，在我以前学习和接触的模型结构中，通常只是使用一次Attention。即使在使用多个Attention的结构中，也通常使用串联来融合表示形式，这样做有一个很明显的缺点就是会使得表示大小成倍增大，从而在后续层中产生成本。

使用多种注意力机制可以显著提高性能，比如Co-Attention 和 Intra-Attention（Self-Attention）中，每种Attention都为query-document对提供了不同的视图，可以学习用于预测的高质量表示。例如，在Co-Attention机制中，利用max-pooling基于单词对另一文本序列的最大贡献来提取特征，利用mean-pooling计算其对整个句子的贡献，利用alignment-based pooling将语义相似的子短语对齐在一起。Co-Attention的论文：[DYNAMIC COATTENTION NETWORKS FOR QUESTION ANSWERING](https://arxiv.org/pdf/1611.01604.pdf)


综上所述，论文主要解决两个方面的问题：
+ 消除调用任意k次注意力机制所需架构工程的需要，且不会产生任何后果。
+ 通过多次注意力调用建模多个视图以提高性能，即多播注意力(Multi-Cast Attention)。

类似Multi-head Attention，通过多次投射Co-attention，每次返回一个压缩的标量特征，重新附加到原始的单词表示上。压缩函数可以实现多个注意力调用的可扩展投射，旨在不仅为后续层提供全局知识而且还有跨句子知识的特征。当将这些增强嵌入传递到组合编码器（例如LSTM编码器）时，LSTM可以从该提示中获益并相应地改变其表示学习过程。

# Multi-Cast Attention Networks（MCAN）结构
首先定义网络的输入，即两个序列，分别是query $q$和document $d$，这两个输入基本可以概括QA或响应预测的输入，下图是QA检索结构图：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210204130339123.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
## Input Encoder
document和query输入作为one-hot编码向量。词向量层通过 $W_e\in R^{d\times |V|}$ 参数化，将每个单词转换为密集的单词表示 $w\in R^d$， $V$ 是词汇表中所有单词的集合。每个词嵌入向量都将通过Highway Encoder（[Highway Encoder](https://arxiv.org/pdf/1505.00387.pdf)原文，即通过一种控制穿过神经网络的信息流的闸门机制为神经网络提供通路，让信息穿过后却没有损失，将这种通路称为information highways，也就是说，highway networks主要解决的问题是网络深度加深、梯度信息回流受阻造成网络训练困难的问题）。

highway网络是门控非线性变换层，它控制后续层的信息流。许多工作都采用一种训练过的投影层来代替原始词向量。这不仅节省了计算成本，还减少了可训练参数的数量。本文将此投影层扩展为使用highway编码器，可以解释为数据驱动的词滤波器，它们可以参数化地了解哪些词对于任务具有重要性和重要性。例如，删除通常对预测没有多大贡献的停用词和单词。与自然门控的循环模型类似，highway编码器层控制每个单词流入下一层多少信息。设 $H(.)$ 和 $T(.)$ 是单层的，分别用ReLU和Sigmoid激活函数进行变换。单个highway网络层定义为：
$$y=H(x,W_H)\cdot T(x,W_T)+(1-T(x,W_T))\cdot x$$
其中，$W_H,W_T\in R^{r\times d}$，前一项表示输入信息被转换的部分，后一项原来信息中保留的部分。
## Co-Attention
Co-Attention 是一种成对注意力机制，它能够共同关注文本序列对。本文引入了四种注意力变体，即max-pooling，mean-pooling，alignment-pooling和intra-attention（self attention）。Co-Attention的第一步是学习两个序列中每个单词之间的一个近似（或相似）矩阵。采用以下公式来学习近似矩阵：
$$s_{ij}=F(q_i)^TF(d_j)$$
其中 $F(.)$ 是诸如多层感知器（MLP）之类的函数。Co-Attention形式也可以为 $s_{ij}=q_i^TMd_j$ 和 $s_{ij}=F([q_i;d_j])$

+ 抽取式Co-Attention最常见变体是max-pooling co-attention，基于每个词在其他文本序列上的最大影响来关注每个词:
$$q^{'}=Soft(\underset{col}{max}(s))^Tq \ and \ d^{'}=Soft(\underset{row}{max}(s))^Td$$
其中，$q^{'},d^{'}$ 分别是 $q$ 和 $d$ 的co-attentive表示，$Soft(.)$ 表示softmax操作
+ 和max-pooling co-attention相似，我们也可以取行和列层次的平均池化矩阵：
$$q^{'}=Soft(\underset{col}{mean}(s))^Tq \ and \ d^{'}=Soft(\underset{row}{mean}(s))^Td$$
mean-pooling co-attention则是基于每个词在其他文本上的总体影响来关注每个词（一般是更合适的选择），当然max和mean之间的选择一般可以作为超参数进行实验调整。
+ Alignment-Pooling，Soft alignment-based pooling也被用于学习co-attentive表示。和标准co-attention最主要区别在于，标准co-attention只是学会对重要单词进行加权和评分，而soft alignment重新调整序列对，用下述公式学习co-attention的表示：
$$d_i^{'}:=\sum_{j=1}^{l_q}\frac{exp(s_{ij})}{\sum_{K=1}^{l_q}exp(s_{ik})}q_j\ and \ q_j^{'}:=\sum_{i=1}^{l_q}\frac{exp(s_{ij})}{\sum_{K=1}^{l_d}exp(s_{kj})}d_i$$
其中 $d_i^{'}$ 是 $q$ 中与 $d_i$ 软对齐的子短语。直观地说， $d_i^{'}$ 是 $\{q_j\}_{j=1}^{l_q}$ 上的加权和，在 $q$ 上选择最相关的部分表示 $d_i$。
+  Intra-Attention（Self-Attention）可以学习长期依赖性的表示，这通常被表述为关于其自身的co-attention（或alignment）操作。在这种情况下，本文将intra-attention同时分别应用于document和query。intra-attention函数定义为（为了符号简单，这里将文档和查询称为 $x$ 而不是 $q$ 或 $d$）：
$$x_i^{'}:=\sum_{j=1}^l\frac{exp(s_{ij})}{\sum_{K=1}^lexp(s_{ik})}x_j$$
其中 $x_i^{'}$ 是 $x_j$ 的内部注意力表示。

## Multi-Cast Attention
在上面的讨论，我们很容易发现每种Attention机制都为模型提供了不同的视角，且Attention是通过重新加权或重新调整来改变原始表示，大多数神经架构仅使用一种类型的co-attention或alignment函数 ，这不仅需要调整模型架构，并且可能错过使用co-attention的多种变体带来的好处， 因此，本文模型将每个注意力操作视为word-level特征，然后使用如下几个操作进行特征表示：
+ Casted Attention：设 $x$ 为 $q$ 或 $d$ ，$\bar{x}$ 是应用attention后x的表示，则co-attention操作的注意力特征是：
$$f_c=F_c([\bar{x};x])$$   $$f_m=F_c(\bar{x}\odot x)$$    $$f_s=F_c(\bar{x}-x)$$
其中 $\odot$ 是Hadamard乘积，$[. ; .]$ 是连接运算符。$F_c(.)$ 是用于将特征减少到标量的压缩函数。 通过比较在co-attention之前和之后的表示来模拟co-attention的影响，其中，这里使用多个比较运算符（减法，连接和乘法运算符）是为了捕捉多个视角。
+ Compression Function：上面提到的 $F_c(.)$ 的压缩函数原理简单而直观，即作者不希望使用高维向量来膨胀后续图层，从而产生后续图层的参数开销，因此本文研究了三种压缩函数的用法，它们能够将 $n$ 维向量减少到标量：
   + Sum（SM）函数是一个非参数化函数，它对整个向量求和，并输出标量：
$$F(x)=\sum_i^nx_i,\forall x_i\in x$$
   + Neural Network（NN）是一个全连接层，按如下方式转换每个n维特征向量：
$$F(x)=ReLU(W_c(x)+b_c)$$
    + Factorization Machines（FM），即因子分解机是一种通用机器学习技术，接受实值特征向量 $x\in R^n$ 并返回标量输出
$$F(x)=w_0+\sum_{i=1}^nw_ix_i+\sum_{i=1}^n\sum_{j=i+1}^n<v_i,v_j>x_ix_j$$
FM是表达模型，使用分解参数捕获特征之间的成对相互作用， $k$ 是FM模型的因子数。

> 注意到，模型不会在多个注意力投射中共享参数，因为每个注意力都旨在为不同的视角建模。实验分别在MCAN（SM），MCAN（NN）和MCAN（FM）下展示了上述变体的结果。
+ Multi-Cast：架构背后的关键思想是促进 $k$ 个注意力投射，每个投射都用一个实值注意力特征来增强原始词向量，对于每个query-document对，应用Co-Attention with mean-pooling，Co-Attention with max-Pooling和Co-Attention with alignment-pooling。 此外，将Intra-Attention分别单独应用于query和document。 每个注意力投射产生三个标量（每个单词），它们与词向量连接在一起。最终的投射特征向量是 $z\in R^{12}$。 因此，对于每个单词 $w_i$，新的表示成为 $[w_i;z_i]$ 。

## LSTM Encoder
将带有casted attetnion的单词表示 $\bar{w_1},\bar{w_2},...\bar{w_l}$ 传递到序列encoder层，采用标准的LSTM编码器：
$$h_i=LSTM(u,i),\forall_i\in [1,...l]$$
其中 $l$ 代表序列的最大长度，LSTM在document和query之间共享权重，关键思想是LSTM encoder通过使用非线性变换作为门控函数来学习表示序列依赖性的表示。因此，在该层之前引入attention作为特征的关键思想是它为LSTM encoder提供了带有信息的提示，例如长期和全局句子知识和句子对（document 和 query）之间的信息。

Pooling Operation：最后，在每个句子的隐藏状态 $\{h_1...h_l\}$ 上应用池化函数，将序列转换为固定维度的表示：$h=MeanMax[h_1...h_l]$。采用MeanMax pooling，它将mean pooling和max pooling的结果连接在一起。我们发现这样比单独使用max pooling或mean pooling更好。

## Prediction Layer和Optimization
最后，给定 document-query 对的固定维度表示，将它们的连接传递到一个两层 $h$ 维highway网络中，模型的最终预测层计算如下：
$$y_{out}=H_2(H_1([x_q;x_d;x_q\odot x_d;x_q-x_d]))$$
其中，$H_1(.),H_2(.)$ 是具有ReLU激活的highway网络层，然后将输出传递到最终线性softmax层：
$$y_{pred}=softmax(W_F\cdot y_{out}+b_F)$$
然后用带有L2范式的标准多分类交叉熵loss函数训练：
$$J(\theta)=-\sum_{i=1}^N[y_ilog\hat{y}_i+(1-y_i)log(1-\hat{y}_i)]+\lambda||\theta||_{L2}$$
其中，$\theta$ 是网络的参数，$\hat{y}$ 是网络的输出，$\lambda$ 是 $||\theta||_{L2}$ L2正则化的权重。

# 实验结果
首先需要知道实验的目的，如下：
+ 论文提出的方法能否在问题回答和对话建模任务上达到最先进的性能？ 与完善的baseline相比有哪些相对改进？
+ 模型结构设计对性能有什么影响？使用LSTM来对casted特征进行学习是否有必要？  co-attention的变体是否都对整体模型性能有所贡献？
+ 能否解释所提出模型的内部运作方式？ 我们可以解释casted attention特征吗？

## Dialogue Prediction
在这个任务中，评估模型是否能够成功预测对话中的下一个回复。使用Ubuntu对话语料库进行实验，其中训练集由一百万个message-response对组成，正负样本1：1。召回率@ k（Rn@K），表示在 $n$ 个response候选的前 $k$ 个结果中是否存在ground truth，论文使用了四个评估指标分别是 $R2@ 1$，$R10@1$，$R10@2$ 和 $R10@5$。

相关参数设置：MCAN中的LSTM encoder的 $d=100$ ，使用学习率 $3\times 10^{-4}$ 的Adam优化器对MCAN进行优化，处理词嵌入层以外的所有层都使用采样率为 $0.2$ 的dropout，序列最大为 $50$，同时使用预训练的 GloVe 作为词嵌入模型。

所有指标的改善比KEHNN好 $5％-9％$。 比AP-LSTM和MV-LSTM的 $R10@1$ 性能提升了 $15％$。总体而言，MCAN（FM）和MCAN（NN）在性能方面具有可比性， MCAN（SM）略低于MCAN（FM）和MCAN（NN），如下图：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210204232737757.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
## Factoid Question Answering
Factoid Question Answering是回答基于事实的问题的任务，在此任务中，目标是为给定问题提供的答案的排序列表。TREC（文本检索会议）的QA数据集，TrecQA是QA最广泛评估和长期作为标准的数据集之一，评估指标使用MAP（mean average precision）和MRR（mean reciprocal rank）。

相关参数设置：MCAN中的LSTM encoder的 $d=300$ ，使用学习率 $3\times 10^{-4}$ 的Adam优化器对MCAN进行优化，L2正则化设置$10^{-6}$，处理词嵌入层以外的所有层都使用采样率为 $0.2$ 的dropout，序列长度取最大序列长度，同时使用预训练的 $300d$ 的GloVe 作为词嵌入模型，使用10 factors的FM模型。

所有MCAN变体都优于所有现有的最先进模型。其中MCAN（FM）是目前在这个广泛研究的数据集上表现最好的模型，如下图：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210204234016105.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
## Community Question Answering (cQA)
此任务涉及在社区论坛中对答案进行排名，与factoid QA不同，答案通常是主观的而不是事实的，而且，答案长度也要长得多。使用QatarLiving数据集，这是一个来自SemEval-2016 Task 3 Subtask的经过充分研究的基准数据集A（cQA），已被广泛用作cQA最新的神经网络模型的基准。包括36000个训练对，2400个开发对和3600个测试对。在这个数据集中，每个问题有十个答案，标记为“正向”和“负向”。评估指标使用Precision@1（P@1）和Mean Average Precision （MAP）

MCAN模型在此数据集上实现了最先进的性能。就P@1指标而言，MCAN（FM）相对于AI-CNN的改善在MAP方面为4.1％和1.1％。相对于CTRN模型，MCAN（FM）也取得了有竞争力的结果，如下图：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210204234440805.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
## Tweet Reply Prediction
使用来自Kaggle的顾客支持数据集，这个数据集中包含对知名品牌的Tweet-Response对，评估指标使用MRR (Mean reciprocal rank)和Precision@1 (accuracy)，实验结果如下图：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210204234717160.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
##  Ablation Analysis
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210204234810211.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
# 模型可解释性分析
通过观察 casted attention特征列出了一些观察结果，下面使用训练了带有FM压缩的MCAN模型，并提取了word-level  casted attention特征，这些特征称为 $f_i$，其中 $i\in[1，12]$。 $f_1,f_2,f_3$ 是从alignment pooling生成的， $f_4,f_5,f6$ 和 $f_7,f_8,f_9$ 分别从最大和平均co-attention中生成， $f_10,f_11,f_12$ 是从intra-attention生成的。

下图显示TrecQA测试集中的正和负QA对：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210204235513465.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
下图显示使用max-pooling attention和mean-pooling attention的casted attention特征：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210204235632415.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)


# 总结
本文主要贡献有以下几点：
+ 首次提出了一种新的思路，不是将attention作为pooling操作，而是作为一种特征增强方式使用，即casted attention。提出了一种用于通用序列对建模的整体架构，称为多播注意力网络Multi-Cast Attention Networks（MCAN）。这是一种新的注意力机制和通用模型架构，用于对话建模和问答系统领域中的排序任务，这种方法执行一系列soft-attention操作，每次返回一个压缩的标量特征，重新附加到原始的单词表示上。关键思想是为后续编码器层提供实值特征，旨在改进表示学习过程。这种设计有几个优点，例如，它允许投射任意数量的注意力机制，允许多种注意力类型（例如，co-attention, intra-attention）和注意力变体（例如，alignment-pooling, max-pooling, mean-pooling）同时执行。这不仅消除了调整co-attention层的昂贵需求，而且还为从业者提供了更大的可解释性。
+ 根据四个基准任务评估提出的模型，即对话回复预测（Ubuntu对话语料库），Factoid问答（TrecQA），社区问答（来自SemEval 2016的QatarLiving论坛）和推特回复预测（客户支持）。在Ubuntu对话语料库中，MCAN的表现优于现有的最先进模型9％。MCAN在经过充分研究的TrecQA数据集上也取得了0.838 (MAP) 和0.904 (MRR) 的最佳表现评分。
+ 对提出的MCAN模型的内部工作进行全面而深入的分析。实验表明，多播注意力特征是可解释的。