# 前言

> [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/pdf/2004.04906.pdf)
> nlp-paper：[NLP相关Paper笔记和代码复现](https://github.com/DengBoCong/nlp-paper)
> nlp-dialogue：[一个开源的全流程对话系统，更新中！](https://github.com/DengBoCong/nlp-dialogue)
> text-similarity：[文本相似度（匹配）计算模型包](https://github.com/DengBoCong/text-similarity)
> 说明：阅读原文时进行相关思想、结构、优缺点，内容进行提炼和记录，原文和相关引用会标明出处，引用之处如有侵权，烦请告知删除。
> 转载请注明：DengBoCong

在开放域的问答系统中，我们需要从大量的文本数据中搜索匹配我们想要的答案（或者学习文档的“信息知识”用于生成答案），而对每个问题都进行全文的数据“学习”是不现实的，因此往往依赖于高效的文本检索来选择候选段落，进而缩小目标范围。

做法往往是将模型分为两大块，首先有一个 Document Retriever，对给定的问题 question，从所有文档中检索，检索到文档后切分成段落，然后用一个称之为 Document Reader 的模块在段落中预测答案位置并给出分数，或者是进行模型的学习生成。而本文的重点就在于Retriever，如何在Retriever阶段提高检索的效率。

传统的检索模型用的较多的有TF-IDF 或 BM25算法，它们通过高效匹配关键词将问题和context表示成一个稀疏的高维度空间向量，这些算法很高效，但是不足之处在于仅仅是在词的匹配上进行检索，并未考虑语义的相关性，有很大的局限性。

所以文中从一个很直观的思路，即通过稠密且能够包含语义信息的空间向量进行表示，那么问题来了，我们怎么样在不进行额外的预训练的前提下，只是用question和passage（或answer）对来训练一个足够好的embedding model呢？论文提出了的方案非常简单，通过优化question和相关passage向量的最大化内积，目的是比较batch中所有的question和passage，这种看似简单的方法，在 top-20 文章检索准确率上却比 Lucene-BM25 系统高出 9%-19%。

# 模型细节
本文目标的本质在于，从大量的语料库来检索出和question最相关的一些passage。本文假设抽取式QA设置（extractive QA setting），将检索的答案限制出现在语料库中的一个或多个段落跨度内。这里我们确定一下表示符号，假设documents集合为 $D=\{d_1,d_2,..,d_D\}$，将每个document拆分为长度相同文本段落作为基本的检索单元，假设总共获得了 $M$ 个段落，则另 $C=\{p_1,p_2,...,p_m\}$，而每个passage $p_i$ 可以表示为序列token $\{w_1^{(i)},w_2^{(i)},...,w_{|p_i|}^{(i)}\}$。在给定一个question $q$ ，那么我们的任务就是如何从passage $p_i$ 中找到能够回答问题的跨度序列 $\{w_s^{(i)},w_{s+1}^{(i)},...,w_{e}^{(i)}\}$，那么我们就可以得到retriever映射 $R:(q,C)\rightarrow C_F$，即输入question $q$ 和 一个语料库 $C$，返回一个最小的匹配文本 $C_F\subset C$，其中 $|C_F|=k\ll |C|$。对于给定的 $k$ ，可以使用 top-k retrieval accuracy来进行evaluate。

实际模型结构很简单，使用两个独立的BERT分别对question和passage进行编码，将得到的两个表示向量进行dot-product或cosine similarity等，可以表示为如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/fb2d70de7aa443a7a8f37545f3402b32.png#pic_center)

论文中提到对数据构造需要注意一些地方，通常在QA数据集中，一般都会有答案所在的段落标记，这个就可以用来作为训练的正样本，但是负样本呢？似乎有很多中可选择的段落，最简单的就是把剩下的没有答案的段落作为负样本，但是有的数据集只有一个正样本段落，这个就比较麻烦。在论文中，针对这个问题，也做了一系列的实验，最后提出了一个比较好的负样本构造方法，同时也可以加速训练。
+ 负样本构造方法：
   + Random：从语料中随机抽取；
   + BM25：使用BM25检索的不包含答案的段落；
   + Gold：训练集中其他问题的答案段落
+ 正样本构造方法：因为数据集如TREC, WebQuestions 和 TriviaQA中只包含问题和答案，没有段落文本，因此，论文通过BM25算法，在Wikipedia中进行检索，取top-100的段落，如果答案没有在里面，则丢掉这个问题。对于 SQuAD 和 NQ数据集，同样使用问题在Wikipedia检索，如果检索到的正样本段落能够在数据集中匹配则留下，否则丢掉。

论文选择了1个正样本和n个负样本，共 $m$ 个样本，表示为 $D=\{\left \langle q_i,p_i^{+},p_i^{-},...,p_{i,n}^{-} \right \rangle\}_{i=1}^m$，然后使用softmax损失作为loss函数，如下：
$$L(q_i,p_i^{+},p_i^{-},...,p_{i,n}^{-} )=-log\frac{e^{sim(q_i,p_i^{+})}}{e^{sim(q_i,p_i^{+})+\sum_{j=1}^ne^{sim(q_i,p_{i,j}^{-})}}}$$
其中，$sim(q,p)=E_Q(q)^TE_P(p)$

论文中还提到一个小trick，就是in-batch negatives，怎么实施呢？假设我们一个mini-batch有 $B$ 个question以及其配套的一个相关的passage，这样我们就得到了 $Q$ 和 $P$ （假设维度大小是 $d$），则矩阵大小为 $B\times d$，那么我们使用dot-product计算相似度矩阵得到 $S=QP^T$，大小为 $B\times B$，我们怎么理解这个相似度矩阵呢？就是一个question和 $B$ 个passage做内积，除了与之相对应passage视为正样本外，其余视为负样本，这样我们就在一个mini-batch内完成了正负样本的学习。

# 实验结果
如下图所示，DPR算法模型使用仅仅1k的数据的时候就已经全面超越了BM25算法的准确性。当训练的数据量逐渐增加时，准确性逐渐提升。当使用全量数据时，DPR算法的准确性超越了BM25算法10个点以上，top-n的数量越少，DPR算法的优势越大。
![在这里插入图片描述](https://img-blog.csdnimg.cn/59de515baa404670a0ead3175e2519eb.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70)
下表显示了在DPR训练时使用的训练数据源和不同的算法组合的结果。Single表示仅使用对应的数据集进行训练，Multi表示将表中的4个数据集全部加入训练。可以看到，大部分的情况下，将其他类型的问题段落数据加入训练时有利于当前数据集的准确率的。其中BM25+DPR整合的方式为：$BM25(q,p)+\lambda\cdot sim(q,p)$，其中 $\lambda = 1.1$
![在这里插入图片描述](https://img-blog.csdnimg.cn/ae5e2efa5e7f4b0a932265e19f5624e9.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70)
IB策略即In-Batch，表示负样本是否在当前batch选取，#N标识负样本数量，得到结果如下图所示：
![在这里插入图片描述](https://img-blog.csdnimg.cn/d9c0df0a1e0940ef826abbb3215b0a29.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70)
使用dot-product还是cosine区别不大：
![在这里插入图片描述](https://img-blog.csdnimg.cn/38ebcdf79ae447f8b8c8ab83ad363a56.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70)
如最开始提到的，DPR相对于传统的BM25检索方法，有着更好的语义理解检索：
![在这里插入图片描述](https://img-blog.csdnimg.cn/9e1282b2730c467aa892a16e36fa7ae7.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70)
下图是使用各检索方法得到的数据集进行QA模型训练，对最终QA效果的影响，基本可以得到检索效果直接影响了最终的QA模型效果：
![在这里插入图片描述](https://img-blog.csdnimg.cn/b361608508f4407bbc816c61011fe372.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70)

