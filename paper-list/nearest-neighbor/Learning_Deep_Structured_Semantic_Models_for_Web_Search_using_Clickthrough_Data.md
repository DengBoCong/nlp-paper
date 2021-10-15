# 前言
> nlp-paper：[NLP相关Paper笔记和代码复现](https://github.com/DengBoCong/nlp-paper)
> nlp-dialogue：[一个开源的全流程对话系统，更新中！](https://github.com/DengBoCong/nlp-dialogue)
> text-similarity：[文本相似度（匹配）计算模型包](https://github.com/DengBoCong/text-similarity)
> Algorithm：[对ML、DL和数据结构算法进行整理讲解，同时含C++/Java/Python三个代码版本](https://github.com/DengBoCong/Algorithm)
> 说明：阅读原文时进行相关思想、结构、优缺点，内容进行提炼和记录，原文和相关引用会标明出处，引用之处如有侵权，烦请告知删除。
> 转载请注明：DengBoCong

本文是针对经典的DSSM(Deep Structured Semantic Models)的原理细究。模型2013提出的，时间虽然很早，但是很经典，所以才翻出来看看。我们通常说的基于深度网络的语义模型，其核心思想是将query和doc映射到到共同维度的语义空间中，通过最大化query和doc语义向量之间的余弦相似度，从而训练得到隐含语义模型，达到检索的目的。DSSM有很广泛的应用，比如：搜索引擎检索，广告相关性，问答系统，机器翻译等。论文见下：
[Learning Deep Structured Semantic Models for Web Search using Clickthrough Data](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf)

# 模型细节
DSSM是Representation-Based模型，其中Query端 Encoder 和 Doc端 Encoder都是使用 MLP实现，最后Score计算使用的是cosine similarity，后续模型的改进很多都是使用更好的Encoder结构，比如LSTM、GRU等结构，结构如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/1b1202f26e6e4998a7e101ef9c0d7750.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQm9Db25nLURlbmc=,size_20,color_FFFFFF,t_70,g_se,x_16)
### 输入处理
结构图中的Term Vector是基于单词词表的one hot表示，DSSM的输入使用了叫作Word Hashing的方式，该方法基于字母的n-gram。给定一个单词（good），我们首先增加词的开始和结束部分（#good#），然后将该词转换为字母n-gram的形式（假设为trigrams：#go，goo，ood，od#）。最后该词使用字母n-gram的向量来表示。在英文里面，3-gram 向量纬度变成了 最大是27*27*27。这样做有如下好处：
+ 英文单词维度是无限的，但是字母级别的n-gram是有限的，极大压缩维度
+ 字母n-gram 可以捕捉同一单词的不同语态时态语境变化
+ out-of-vocubulary 鲁棒性，(前后缀，语态时态词的相似变化)
+ collision：不同单词的letter n-gram表示一样认为是一次碰撞

论文使用的数据集是点击日志，里面通常包含了用户搜索的query和用户点击的doc，可以假定如果用户在当前query下对doc进行了点击，则该query与doc是相关的。

$D$ 表示被排序的候选文档集合，在实际中，对于正样本，每一个(query， 点击doc)对，使用 $(Q,D^+)$ 表示；对于负样本，随机选择4个曝光但未点击的doc，用$\{D_j^-;j=1,...,4\}$来表示。
> 这里对未点击的进行负采样，比例为1:4，论文中特意提到采用不同的采样比例对结果影响不大。

表示层倒没什么好讲的，三个全连接层，激活函数采用的是tanh，把维度降低到128。

### 损失函数
为弄清 DSSM 损失函数是 Pointwise / Pairwise / Listwise Loss 中的哪一种，首先要弄清楚这三种 Loss 的区别。Pointwise / Pairwise / Listwise Loss 的区别如下：
+ Pointwise Loss：评估单个样本真实标签的预测准确度，即计算单个 loss 时要么只用一个正样本，要么只用一个负样本。总误差既包括正样本带来的误差，也包括负样本带来的误差。
+ Pairwise Loss：评估一个样本对的标签与真实标签是否一致，即计算单个 loss 时会同时用到一个正样本和一个负样本；
+ Listwise Loss：当真实标签是所有文档关于给定查询的相关度时，Listwise Loss 往往是一些常用的 IR 评估指标；当真实标签是一个排好序的列表时，Listwise Loss 评估的是预测的排序列表和真实列表之间的差异。无论那每一种，计算单个 loss 时都会使用所有正负样本。
可见，Pointwise / Pairwise / Listwise Loss 可以简单地通过计算单个 loss 时使用的正负样本数量进行区分。


DSSM 损失函数不是 Pointwise loss，也不是像 triplet loss 这种传统的 Pairwise Loss。从不同的角度看，DSSM 损失函数即可以是 Pairwise Loss 的增强版，也可以是 Listwise loss 的近似版，介于二者之间。

DSSM 损失函数有如下优势：
+ 与 Pointwise Loss 相比，该损失函数具有 Pairwise Loss 的排序能力；
+ 与只用了一个负样本的 Pairwise Loss 相比，该损失函数采用更多的负样本，可以为正样本学习到更有区别性的表示，同时加速模型收敛；
+ 与真实分布相比，通过负采样技术可降低损失函数的计算复杂度，加速模型的训练和迭代。

但该损失函数也存在一些弊端：
+ 从 Pairwise Loss 的角度看，既不能评估同一查询对应的两个正样本之间的关系， 也没有考虑样本位置对排序结果的影响；
+ 从 Listwise Loss 的角度看，当样本量较少时，近似的“真实分布”无法较好地模拟真实分布，DSSM 的效果可能并不理想；
+ Sohn 指出 (N+1)-tuplet loss 一次更新需要的样本量太大，计算成本高，所以该大神又提出了 N-pair-mc losss。（既然 DSSM 损失函数不是 pointwise loss，那把 Pointwise loss 的缺点安在 DSSM 头上显然是不合理的）。

据说说在工业实践时（如推荐系统中的双塔模型）把 DSSM 的损失函数改为 triplet loss 取得了更好的收益。

# 总结
优点

+ 解决了LSA、LDA、Autoencoder等方法存在的一个最大的问题：字典爆炸（导致计算复杂度非常高），因为在英文单词中，词的数量可能是没有限制的，但是字母 n -gram的数量通常是有限的
+ 中文方面使用字作为最细切分粒度，可以复用每个字表达的语义，减少分词的依赖，从而提高模型的泛化能力；
+ 基于词的特征表示比较难处理新词，字母的 [公式] -gram可以有效表示，鲁棒性较强
+ 使用有监督方法，优化语义embedding的映射问题
省去了人工的特征工程
+ 采用有监督训练，精度较高。传统的输入层使用embedding的方式(比如Word2vec的词向量)或者主题模型的方式(比如LDA的主题向量)做词映射，再把各个词的向量拼接或者累加起来。由于Word2vec和LDA都是无监督训练，会给模型引入误差。


缺点：
+ Word Hashing可能造成词语冲突；
+ 采用词袋模型，损失了上下文语序信息。这也是后面会有CNN-DSSM、LSTM-DSSM等DSSM模型变种的原因；
+ 搜索引擎的排序由多种因素决定，用户点击时doc排名越靠前越容易被点击，仅用点击来判断正负样本，产生的噪声较大，模型难以收敛；
+ 效果不可控。因为是端到端模型，好处是省去了人工特征工程，但是也带来了端到端模型效果不可控的问题。