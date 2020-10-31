# 前言

> 标题：Sequential Matching Network: A New Architecture for Multi-turn Response Selection in Retrieval-Based Chatbots\
> 原文链接：[Link](https://arxiv.org/pdf/1612.01627v2.pdf)\
> Github：[NLP相关Paper笔记和实现](DengBoCong/nlp-paper)\
> 说明：阅读论文时进行相关思想、结构、优缺点，内容进行提炼和记录，论文和相关引用会标明出处，引用之处如有侵权，烦请告知删除。\
> 转载请注明：DengBoCong

# 介绍
本文的SMN模型结构可以说影响了很多后续相关的论文，所解决的是基于检索的聊天机器人中多回合对话的回复选择。在之前的工作，基于检索的聊天机器人的做法是将context里所有的utterances都连接在一起，将这个长长的context做处理然后和response作匹配，这样做可能会丢失语句间的关系或重要的上下文信息。Sequential Matching Network（SMN）模型就是为了解决这些问题而来的。

**SMN首先在多个粒度级别上将上下文中每个utterance都和response做匹配，然后通过卷积和池化操作从每对中提取重要的匹配信息作为向量，接着通过递归神经网络（RNN）按时间顺序累积矢量，该神经网络对utterance之间的关系进行建模，最后使用RNN的隐藏状态计算最终匹配分数。**

构建对话机器人的现有方法中，可以分为 generation-based（生成式）和retrieval-based（检索式），相对于生成式而言，检索式拥有的信息更加丰富，且运行流畅的特点。选择response的关键在于输入response匹配。与单回合对话不同，多回合对话需要在响应和对话上下文之间进行匹配，在该上下文中，不仅需要考虑response和输入信息之间的匹配，而且还需要考虑前一回合中response和utterances之间的匹配。总结而言该领域的任务存在如下的challenges：
+ 如何根据上下文识别重要信息（单词，短语和句子），这对于选择正确的response并利用相关信息进行匹配至关重要
+ 如何对上下文中的utterances之间的关系进行建模。

下图展示了challenges的例子：第二句的“hold a drum class”和第三句的“drum”相关性很强，所以回复是高度依赖于上下文以及语句之间的关系的。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201030203732642.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
在上面粗体中描述了SMN的结构流程，具体来说，对于每个utterance-response对而言，模型通过word embeddings和带有门控递归单元（GRU）的递归神经网络的隐藏状态来构造word-word相似度矩阵和sequence-sequence相似度矩阵。这两个矩阵分别在word级别和segment（单词子序列）级别上捕获成对的重要匹配信息，并且通过交替对矩阵进行卷积和池化操作，将信息提取和融合为匹配向量。

通过这种方式，可以在response的充分监督下识别上下文中多个粒度级别的重要信息，并以最小的损失进行匹配，然后将匹配向量喂给另一个GRU，以形成context和response的匹配分数，这个GRU会根据上下文中utterances的时间顺序在其隐藏状态下累积匹配。它以匹配的方式对utterances之间的关系和依存关系进行建模，并以utterances的顺序来监督配对匹配的累积。context和response的匹配程度是由具有GRU隐藏状态的logit模型计算的。

# 具体结构及实现
### 问题符号化
数据集表示为 $D=\{(y_i,s_i,r_i)\}_{i=1}^N$，其中$s_i=\{u_{i,1},...,u_{i,n_i}\}$表示对话上下文，$\{u_{i,k}\}_{k=1}^{n_i}$表示utterances，$r_i$表示response的候选，$y_i\in\{0,1\}$表示标签（当$y_i=1$ 时意味着 $r_i$ 是 $s_i$ 合适的回复，否则 $y_i=0$），目标是学习 $D$ 的匹配模型 $g(.,.)$，对于任何context-response pair $(s,r)$，$g(s,r)$ 衡量 $s$ 和 $r$ 之间的匹配程度。
### 模型结构
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201030215924927.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
按照上面的模型结构图，SMN首先将context-response分解为几个utterance-response pair匹配，然后通过递归神经网络将所有pairs匹配累积为基于上下文的匹配。模型分为三层：
+ 在word级别和segment级别将候选response与上下文中的每个utterance匹配，然后通过卷积提取两个层上的重要匹配信息，池化并编码为匹配向量。
+ 将上一层得到的匹配向量喂入第二层，在该层中，它们按照上下文中utterances的时间顺序在具有GRU的递归神经网络的隐藏状态下进行累积。
+ 将上一层的隐藏状态用于计算最终匹配分数

**这样的模型结构有以下的优势：**
+ 一个候选response可以在一开始就匹配上下文中的每个utterances，因此可以充分提取每个utterance-response pair中的匹配信息，并以最小的损失将其携带到最终匹配分数
+ 由于从每个utterance中提取信息是在不同的粒度级别上进行的，并且在response的充分监督下进行，因此可以很好地识别和提取对每个utterance中的response选择有用的语义结构。
+ 匹配和utterance关系是耦合而不是分开建模的，因此，作为一种knowledge，utterance关系（例如顺序）可以监督匹配分数的形成。

接下来针对三层的细节展开描述。
### Utterance-Response匹配
给定上下文 $s$ 中的语句 $u$ 和候选响应 $r$，然后对$u$ 和 $r$ 进行Embedding得到对应的表示 $U=[e_{u,1},...,e_{u,n_u}]$ 和 $R=[e_{r,1},...,e_{r,n_r}]$ ，其中 $e_{u,i},e_{r,i}\in \mathbb{R}^d$ 分别是$u$ 和 $r$ 的第 $i$ 个单词的嵌入。然后使用 $U\in \mathbb{R}^{d\times n_u}$ 和 $R\in \mathbb{R}^{d\times n_r}$ 构造word-word相似度矩阵 $M_1\in \mathbb{R}^{n_u\times n_r}$，以及sequence-sequence相似度矩阵 $M_2\in \mathbb{R}^{n_u\times n_r}$，这两个矩阵作为卷积神经网络（CNN）的两个input channels，CNN从矩阵中提取重要的匹配信息，并将该信息编码为匹配向量 $v$。具体来说，$\forall i,j$，$M_1$ 第 $(i,j)$ 个元素被定义为（公式1）：
$$e_{1,i,j}=e_{u,i}^T\cdot e_{r,j}$$
$M_1$ 在单词级别上模拟 $u$ 和 $r$ 之间的匹配。为了构造$M_2$ ，我们首先使用GRU将 $U$ 和 $R$ 转换为隐藏向量。假设 $H_u=[h_{u,1},...,h_{u,n_u}]$ 是 $U$ 的隐藏向量，则 $\forall i,h_{u,i}\in \mathbb{R}^m$ 定义为（公式2）：
$$z_i=\sigma (W_{ze_{u,i}}+U_zh_{u,i-1})$$  $$r_i=\sigma (W_{re_{u,i}}+U_rh_{u,i-1})$$  $$\tilde{h}_{u,i}=tanh(W_{he_{u,i}}+U_{h}(r_i\odot h_{u,i-1}))$$  $$h_{u,i}=z_i\odot\tilde{h}_{u,i}+(1-z_i)\odot h_{u,i-1}$$
其中 $h_{u,0}=0$，$z_i$ 和 $r_i$ 分别是update gate和reset gate，$\sigma(\cdot)$是一个sigmoid函数，$W_z,W_h,W_r,U_z,U_r,U_h$ 都是参数，同样，我们有 $H_r=[h_{r,1},...,h_{r,n_r}]$ 作为 $R$ 的隐藏向量，然后 $\forall i,j$，$M_2$ 第 $(i,j)$ 个元素被定义为（公式3）：
$$e_{2,i,j}=e_{u,i}^TAh_{r,j}$$
其中 $A\in \mathbb{R}^{m\times m}$ 是一个线性变换， $\forall i$，GRU对直到位置 $i$ 的单词之间的顺序关系和依赖关系进行建模，并对text segment进行编码，直到第 $i$ 个单词为隐藏矢量为止，$M_2$ 在segment级别上建模$u$ 和 $r$ 之间的匹配。

然后，CNN将 $M_1$ 和 $M_2$ 处理为  $v$。$\forall f=1,2$，CNN视 $M_f$ 为输入通道，交替进行卷积和最大池化操作。假设$\forall f=1,2$， $z^{(l,f)}=[z_{i,j}^{(l,f)}]_{I^{(l,f)}\times J^{(l,f)}}$ 表示 $l$ 层上类型为 $f$ 的特征图的输出，其中 $z^{(0,f)}=M_f$。在卷积层上，我们使用窗口大小为 $r_w^{(l,f)}\times r_h^{(l,f)}$ 的2D卷积运算，并将 $z_{i,j}^{(l,f)}$ 定义为（公式4）：
$$z_{i,j}^{(l,f)}=\sigma(\sum_{f^{'}=0}^{F_{l-1}}\sum_{s=0}^{r_{w}^{(l,f)}}\sum_{t=0}^{r_{h}^{(l,f)}}W_{s,t}^{(l,f)}\cdot z_{i+s,j+t}^{(l-1,f^{'})}+b^{l,k})$$
其中，$\sigma(\cdot)$是一个ReLU，$W^{(l,f)}\in \mathbb{R}^{r_w^{(l,f)}\times r_h^{(l,f)}}$ 和 $b^{l,k}$ 是参数，$F_{l-1}$ 是第 $(l-1)$ 层上的特征图的数量，最大池化操作基于卷积操作，可以表示为（公式5）：
$$z_{i,j}^{(l,f)}=\underset{p_w^{(l,f)}>s\geq0}{max} \underset{p_h^{(l,f)}>t\geq0}{max}z_{i+s,j+t}$$
其中 $p_w^{(l,f)}$ 和 $p_h^{(l,f)}$ 分别是2D池的宽度和高度，最终特征图的输出被串联并映射到低维空间，并通过线性变换作为匹配向量 $v\in \mathbb{R^q}$
### Accumulation匹配
假设 $[v_1,...,v_n]$ 是第一层的输出（对应 $n$ pairs），在第二层，GRU取 $[v_1,...,v_n]$ 作为输入，并将匹配序列编码为其隐藏状态 $H_m=[h_1^{'},...,h_n^{'}]\in  \mathbb{R^{q\times n}}$。 其详细参数设置与公式（2）相似，这个层有两个函数：
+ 它在上下文中建模utterances的依存关系和时间关系
+ 它利用时间关系来监督对accumulation的累积，作为基于上下文的匹配

### Prediction和Learning匹配
存在 $[h_1^{'},...,h_n^{'}]$，我们将 $g(s,r)$ 定义为（公式6）：
$$g(s,r)=softmax(W_2L[h_1^{'},...,h_n^{'}]+b_2)$$
其中，$W_2$ 和 $b_2$ 是参数，我们考虑  $L[h_1^{'},...,h_n^{'}]$ 的三个参数化：
+ 仅使用最后一个隐藏状态，  $L[h_1^{'},...,h_n^{'}]=h_n^{'}$。
+ 隐藏状态线性组合，  $L[h_1^{'},...,h_n^{'}]=\sum_{i=1}^{n}w_ih_i^{'}$，其中 $w_i\in  \mathbb{R}$
+ 我们遵循并运用注意力机制来组合隐藏状态，则 $L[h_1^{'},...,h_n^{'}]$ 被定义为（公式7）：
$$t_i=tanh(W_{1,1}h_{u_i,n_u}+W_{1,2}h_i^{'}+b_1)$$  $$a_i=\frac{exp(t_i^Tt_s)}{\sum_i(exp(t_i^Tt_s))}$$  $$L[h_1^{'},...,h_n^{'}]=\sum_{i=1}^{n}a_ih_i^{'}$$
其中， $W_{1,1} \in \mathbb{R^{q\times m}}$ ，$W_{1,2}\in  \mathbb{R^{q\times q}}$和 $b_1\in  \mathbb{R^{q}}$ 是参数， $h_i^{'}$ 和 $h_{u_i,n_u}$ 分别是第 $i$ 个匹配向量和第 $i$ 个utterance的最终隐藏状态， $t_s\in \mathbb{R^{q}}$ 是一个虚拟上下文向量，它是随机初始化的，并在训练中共同学习。我们用 $L[h_1^{'},...,h_n^{'}]$ 的三个参数化来表示我们的模型，分别是 $SMN_{last}$，$SMN_{static}$ 和 $SMN_{dynamic}$，并在实验中进行比较。

我们通过用 $D$ 最小化交叉熵来学习 $g(\cdot,\cdot)$。令 $\theta$ 表示 $SMN$ 的参数，则学习的目标函数 $L(D,\theta)$ 可表示为（公式8）：
$$-\sum_{i=1}^{N}[y_ilog(g(s_i,r_i))+(1-y_i)log(1-g(s_i,r_i))]$$

### 检索候选Response
作者利用启发式方法从索引中获取候选response，将前一轮的utterances $\{u_1,...,u_{n-1}\}$ 和 $u_n$ 进行计算，根据他们的**tf-idf**得分，从 $\{u_1,...,u_{n-1}\}$ 中提取前 $5$ 个关键字，然后，我们将扩展后的message用于索引，并使用索引的内联检索算法来检索候选response。最后，我们使用 $g(s,r)$ 对候选进行排名，并返回第一个作为对上下文的response。

# 实验结果
+ Ubuntu语料
+ 豆瓣多轮语料：下图给出三组的统计数据。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201031102216402.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
**下图显示两个数据集的评估结果：**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201031103318145.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
我们使用Ubuntu语料库中的示例对第二层中的相似性矩阵和GRU的gates进行可视化，以进一步阐明我们的模型如何在上下文中识别重要信息，以及如何使用前文所述的GRU的门机制选择重要的匹配向量，示例及可视化图如下：

```
{
	u1: how can unzip many rar (_number_ for example ) files at once; 
	u2: sure you can do that in bash; 
	u3: okay how? 
	u4: are the files all in the same directory? 
	u5: yes they all are; 
	r: then the command glebihan should extract them all from/to that directory
}
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020103110442392.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
**下面是消融（Ablation）实验的结果：**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201031104859530.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
**跨context长度的效果：**显示了豆瓣语料库上不同长度间隔的MAP的比较
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201031104816446.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
**最大context长度：**展示了在最大context长度方面，SMN在Ubuntu Corpus和Douban Corpus上的性能
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201031105902650.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
**错误分析：**
+ 逻辑一致性。 SMN在语义级别上对context和response进行建模，但很少关注逻辑一致性
+ 检索后没有正确的候选

# 总结
论文提出了一个新的基于上下文的模型，用于基于检索的聊天机器人中的多轮响应选择，论文还详尽的介绍了豆瓣对话语料库，并且做了实验去研究应该在context中取多少个utterance，即取多少轮对话，实验证明轮数取10的时候效果最好，很值得学习的