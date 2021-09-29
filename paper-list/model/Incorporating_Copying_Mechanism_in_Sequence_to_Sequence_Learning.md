
# 前言

> [Incorporating Copying Mechanism in Sequence-to-Sequence Learning](https://arxiv.org/pdf/1603.06393.pdf)
> nlp-paper：[NLP相关Paper笔记和代码复现](https://github.com/DengBoCong/nlp-paper)
> nlp-dialogue：[一个开源的全流程对话系统，更新中！](https://github.com/DengBoCong/nlp-dialogue)
> text-similarity：[文本相似度（匹配）计算模型包](https://github.com/DengBoCong/text-similarity)
> Algorithm：[对ML、DL和数据结构算法进行整理讲解，同时含C++/Java/Python三个代码版本](https://github.com/DengBoCong/Algorithm)s
> 说明：阅读原文时进行相关思想、结构、优缺点，内容进行提炼和记录，原文和相关引用会标明出处，引用之处如有侵权，烦请告知删除。
> 转载请注明：DengBoCong

最近看了一篇ACL2021的Dialog Policy Learning模型的文章，阅读笔记如下：
[阅读笔记](https://zhuanlan.zhihu.com/p/415170940)
模型结构里面使用到了一个Copying Mechanism的技巧，因此比较感兴趣的翻了这篇原文阅读。本篇论文提出了CopyNet模型，能够自动的，决定下一步的预测是生成模式还是复制模式。生成模式就是我们常说的注意力机制，复制模式就是这篇文章的一个创新点。复制模式其实不难理解，从我们人类的经验视角来看，在阅读文章或者做一些摘要的时候，除了自己会生成一些概括语句之外，还会从文章当中去摘抄一些核心句子。因此我们在生成句子时，可以选择性的复制某些关键词，比如如下这样：
![在这里插入图片描述](https://img-blog.csdnimg.cn/54e0ebaf95a94c7a8955615a558e4e6c.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQm9Db25nLURlbmc=,size_17,color_FFFFFF,t_70,g_se,x_16)
如上述的一些人名等OOV_token，如果单单使用传统的Attention是无法处理的，所以需要通过某种手段来解决。Copying Mechanism从一定程度上解决这个问题，用原Paper的说法，模型只需要更少的理解，就能够确保文字的保真度，对于摘要，对话系统等来说，能够提高文字的流畅度和准确率，并且也是端到端进行训练。

# 模型细节
模型依旧是Encoder-Decoder的结构，不过是在Decoder中，融入了Copying Mechanism，首先看一下模型的整体结构，如下图：
![在这里插入图片描述](https://img-blog.csdnimg.cn/ff619a83e87249c7968acc8ceeb5cd14.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQm9Db25nLURlbmc=,size_20,color_FFFFFF,t_70,g_se,x_16)
模型的Encoder部分就是使用了Bi-RNN，编码得到的hidden state 统一用 $M$ 表示（上图右侧的 $M$），下面我们重点来讲一下Decoder。
## Decoder
Decoder也是使用RNN来获取 $M$ 的信息从而预测目标序列，只不过并不是直接解码，而是做了如下三个改动：
#### 生成模式&复制模式
设有一个词表 $V=\{v_1,v_2,...,v_n\}$，使用UNK表示OOV，输入表示为 $X=\{x_1,x_2,...,x_{T_s}\}$。因为 $X$ 当中包含一些词汇表中没有的单词，所以使用copy mode，可以输出部分OOV单词，这样就使得我们整个词表为 $V\cup UNK\cup X$。在第 $t$ 步解码的state 表示为 $s_t$，$y_t$ 表示生成目标词概率，则两个模式的混合概率模型表示为：
$$p(y_t|s_t,y_{t-1},c_t,M)=p(y_t,g|s_t,y_{t-1},c_t,M)+p(y_t,c|s_t,y_{t-1},c_t,M)$$
其中，$g$ 表示生成模式，$c$ 表示复制模式，则两个模式的概率模型分别为：
$$p(y_t,g|\cdot)=\left\{\begin{matrix}\frac{1}{Z}e^{\psi_g(y_t)}, & y_t\in V \\ 0, & y_t\in X\cap\bar{V} \\ \frac{1}{Z}e^{\psi_g(UNK)}, &y_t\notin V\cup X \end{matrix}\right.$$
$$p(y_t,c|\cdot)=\left\{\begin{matrix} \frac{1}{Z}\sum_{j:x_j=y_t}e^{\psi_c(x_j)}, & y_t\in X \\ 0,& otherwise\end{matrix}\right.$$
其中，$\psi_g(\cdot),\psi_c(\cdot)$分别是generate mode和copy mode的分数计算方法，$Z=\sum_{v\in V\cup\{UNK\}}e^{\psi_g(x)}+\sum_{x\in X}e^{\psi_c(x)}$ 则是两种模式间共享的归一化项，两种模式分数的计算方法：
+ generate mode
$$\psi_g(y_t=v_i)=v_i^TW_os_t,v_i\in V\cup UNK$$
其中，$W_o\in\mathbb{R}^{(N+1)\times d_s}$，$v_i$是one-hot形式
+ copy mode
$$\psi_c(y_t=x_j)=\sigma (h_j^TW_c)s_t,x_j\in X$$
其中，$W_c\in\mathbb{R}^{d_h\times d_s}$，$\sigma$ 是一个非线性激活函数（tanh非线性激活函数比linear transformation的效果更好）

因此，总共需要考虑4个情况，目标词 $y_t$ 如果属于词汇表或者 $X$，就分别计算上述两个概率；如果既不属于词汇表，也不属于源端，就是 $UNK$；如果属于 $X$，但不属于词汇表，那么生成的概率为0；如果不属于 $X$，那么复制的概率为0。Z是两种模式共享的归一化项，下图可见：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2fe1c7ecfd6348cebf8479ae9b8db101.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQm9Db25nLURlbmc=,size_17,color_FFFFFF,t_70,g_se,x_16)
优化目标应该是在训练当中 $y_t$ 作为目标词，整个模型的目标是要让 $y_t$ 的混合概率最大，即-log似然概率最小：
$$L=-\frac{1}{N}\sum_{k=1}^N\sum_{t=1}^Tlog[p(y_t^{(k)}|y_{<t}^{(k)},X^{(k)})]$$
然后在预测的时候，因为没有目标词。按照作者给的模型框架图，应该是在最后一层前馈神经网，维度为词汇表的数量加源端句子的长度。然后把每一个词对应的概率加起来选择最大概率的词作为prediction（这个词如果没有出现在源端，则复制模式概率为零。这个词如果出现在了源端，但是不在目标词里，生成模式概率为0）。
#### State Update
注意力机制中，状态的更新为下公式：
$$s_t=f(y_{t-1},s_{t-1},c)$$
但是在CopyNet， $y_{t-1}$ 有一点小的变化，其被表达成以下形式：
$$[e(y_{t-1});\zeta(y_{t-1})]^T$$
前者是一个embedding，后者是一个加权和的计算，对于 $X$ 中的词，如果其等于 $y_{t-1}$ ，则以如下公式进行计算：
$$\zeta(y_{t-1})=\sum_{\tau=1}^{T_S}\rho_{t\tau h_\tau}$$
$$\rho_{t\tau}=\left\{\begin{matrix} \frac{1}{K}p(x_\tau,c|s_{t-1},M), & x_\tau=y_{t-1} \\ 0 & otherwise \end{matrix}\right.$$
否则其概率为0。也就是说，我们挑选出等于 $y_{t-1}$ 的词的隐层状态和词向量连接。其中 $K=\sum_{\tau^{'}:x_{\tau^{'}}=y_{t-1}}p(x_{\tau^{'}},c|s_{t-1},M)$，是归一化项。论文中把这里的操作称之为selective read，与attentive read相似。attentive read是用decoder的隐状态和encoder隐状态做attention，是soft操作，而selective read是用 $t-1$ 时刻的输出与encoder的输入做selection，不相等则为0，是hard操作。

# 实验结果
![在这里插入图片描述](https://img-blog.csdnimg.cn/d42dad7afb1b43bdb0aa3175067daca2.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQm9Db25nLURlbmc=,size_16,color_FFFFFF,t_70,g_se,x_16)

第一个实验是自动文摘，因为文摘任务当中，可以发现摘要中的大部分，都是可以从原文当中直接复制过来的。在源端可能会出现很多的out of vocabulary，所以基于注意力的encoder decoder模型，生成的摘要效果很差。
![在这里插入图片描述](https://img-blog.csdnimg.cn/261c78e67ff14be996520b995d80083e.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQm9Db25nLURlbmc=,size_20,color_FFFFFF,t_70,g_se,x_16)
第二个实验用在单轮对话系统，在单轮对话任务当中，虽然基于注意力的encoder decoder模型，能够去生成完整的有语义的句子，可是往往答非所问。
![在这里插入图片描述](https://img-blog.csdnimg.cn/75fa610b2e7b45d8acf18b71e352ae3e.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQm9Db25nLURlbmc=,size_20,color_FFFFFF,t_70,g_se,x_16)
# 总结
加入copy机制，不仅仅考虑生成模式，还考虑复制模式，对文摘或者单轮对话任务起到比较好的效果。CopyNet可以在一定程度上解决部分未登录词，但是并不能解决所有的未登录词问题，而且我觉得对于文本摘要这种需要从文章中提取关键词的任务好像更适合用CopyNet，因为文本摘要中的提取出来的命名实体识别的词更多一些。
