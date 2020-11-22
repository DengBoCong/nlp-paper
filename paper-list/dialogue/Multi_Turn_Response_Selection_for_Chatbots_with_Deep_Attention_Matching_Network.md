# 前言

> 标题：Multi-Turn Response Selection for Chatbots with Deep Attention Matching Network\
> 原文链接：[Link](https://www.aclweb.org/anthology/P18-1103.pdf)\
> Github：[NLP相关Paper笔记和代码复现](https://github.com/DengBoCong/nlp-paper)\
> 说明：阅读论文时进行相关思想、结构、优缺点，内容进行提炼和记录，论文和相关引用会标明出处，引用之处如有侵权，烦请告知删除。\
> 转载请注明：DengBoCong

# 介绍
聊天机器人中的一项重要任务是响应选择，其目的是在给定对话上下文的情况下，从一组候选响应中选择最匹配的响应。除了在基于检索的聊天机器人中发挥关键作用外，响应选择模型还可以用于对话生成的自动评估以及基于GAN（生成对抗网络）神经对话生成的判别器。之所以使用更丰富的上下文信息，是因为人为产生的响应在很大程度上取决于语义和场景上的不同粒度（单词，词组，句子等）的先前对话段。下图展示跨上下文和响应的段对之间的语义连接。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201122202203122.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
如图所示，在上下文和响应中，通常存在两种匹配的段对，它们的粒度不同：
+ 表面文本的相关性，例如单词“packages”-“package”和短语“debian package manager”-“debian package manager”的词法重叠。
+ 这些片段在语义/功能上有着彼此相关的潜在依赖关系。例如，响应中的“it”一词在上下文中指“dpkg”，以及响应中的短语“its just reassurance”，它潜在地指向“what packages are installed on my system”。

先前的研究表明，在上下文和响应中以不同的粒度捕获那些匹配的片段对是多回合响应选择的关键，如Multi-view和SMN。然而这些对话系统只考虑了表面的文本关联性（surface text relevance），且多采用RNN结构，这会极大的增加模型的推理代价，因此本文提出了基于注意力机制的结构。本文使用完全基于注意力的依赖项信息来研究将响应与多回合上下文匹配的过程，受到Transformer的影响，本文通过两种方式扩展注意力机制来获取表示和匹配信息：
+ **self-attention**：只使用堆叠的自注意力构造不同粒度的文本段表示，这样我们可以捕获其词级内的依存关系。self-attention其实就是Transformer的Encoder层（没有用Multi-Head），其中 $Q$ 、 $K$ 、$V$ 都是一样的，要么都是response、要么都是utterance。从word embedding开始堆叠self-attention层，每一层self-attention层抽取一种粒度的表示，不同层就抽取了不一样的表示，从而获得句子多粒度的表示。
+ **cross-attention**：尝试整个上下文和响应的注意力提取真正匹配的句段对。结构上还是Transformer的Encoder层，只不过输入不一样了。其中 $Q$ 是response（utterance），而 $K$ 、 $V$ 是utterance（response）。这样做的话，其实就像用response去表示utterance，用utterance去表示response，从而为response和utterance做匹配提供更多的信息。

本文在一个统一的神经网络中介绍了这两种注意力，网络命名为Deep Attention Matching Network（DAM），用于多回合响应选择。在实践中，DAM将上下文或响应中的语句的每个词作为抽象语义段的中心含义，并通过堆叠式的自注意力丰富其表示，从而逐渐围绕中心词生成越来越复杂的段表示 。考虑到文本相关性和依存性信息，上下文和响应中的每个语句都基于不同粒度的句段对进行匹配。这样，DAM通常会捕获上下文之间的匹配信息以及从单词级到句子级的响应，然后使用卷积和最大池化操作提取重要的匹配特征，最后通过单层感知器将其融合为一个匹配分数。更重要的是，由于大多数注意力计算可以完全并行化，因此DAM有望在实践中方便部署。

# 模型细节
## 问题符号化
给定一个对话数据集 $D=\{(c,r,y)z\}_{Z=1}^N$，其中 $c=\{u_0,...u_{n-1}\}$ 表示一个对话上下文，其中 $\{u_i\}_{i=0}^{n-1}$ 作为语句和 $r$ 作为候选响应。$y\in\{0,1\}$ 是一个二进制标签，指示 $r$ 是否是对 $c$ 的合适响应。我们的目标是学习与 $D$ 匹配的模型$g(c,r)$，该模型可以测量上下文 $c$ 和候选响应 $r$ 之间的相关性。
## 模型结构
下图展示了DAM的模型结构，该模型通过representation-matching-aggregation框架来将响应与多回合上下文进行匹配。对于上下文中的每个语句 $u_i=[w_{u_i,k}]_{k=0}^{n_{u_i}-1}$ 和它的候选响应 $r=[w_{r,t}]_{t=0}^{n_r-1}$（其中，$n_{u_i}$ 和 $n_r$ 代表单词数），DAM首先查找共享的单词嵌入表，并将 $u_i$ 和 $r$ 表示为单词嵌入序列，即 $U_i^0=[e_{u_i,0}^0,...,e_{u_i,n_{u_i}-1}^0]$ 和 $R^0=[e_{r,0}^0,...,e_{r,n_r-1}^0]$，其中 $e\in\mathbb{R}^d$ 表示 $d$ 维词嵌入。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201122220222537.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)

接下来，通过表示模块为 $u_i$ 和 $r$ 在不同的粒度上构造语义表示。实现方式是通过 $L$ 个相同的自注意层堆叠，每个第 $l$ 个自注意层都将第 $l-1$ 层的输出作为其输入，从而将输入语义向量合成为更复杂的表示形式。通过这种方式，逐渐构建了 $u_i$ 和 $r$ 的多粒度表示，分别表示为 $[U_i^l]_{l=0}^L$ 和 $[R^l]_{l=0}^L$。

给定 $[U_i^0,...,U_i^L]$ 和 $[R^0,...,R^L]$，语句 $u_i$ 和响应 $r$ 随后以segment-segment相似矩阵的方式相互匹配。实际上，对于每个粒度 $l\in[0...L]$，构造两种匹配矩阵，即self-attention匹配 $M_{self}^{u_i,r,l}$ 和cross-attention匹配 $M_{corss}^{u_i,r,l}$，这样便分别使用文字信息和依存关系信息测量话语和回应之间的相关性。这些匹配分数最终被合并为 $3D$ 的匹配 $Q^1$。$Q$ 的每个维度表示**each utterance in context, each word in utterance and each word in response**。然后，通过使用最大池操作进行卷积，提取跨多回合上下文的段对之间的重要匹配信息和候选响应，然后通过单层感知器将其进一步融合为一个匹配分数，代表候选响应与整体上下文之间的匹配程度。

注意力，我们使用一个共享组件，即“注意力模块”来实现表示中的self-attention和匹配中的cross-attention。在以下各节中，我们将详细讨论注意力模块的实现以及如何使用它来实现self-attention和cross-attention

##  注意力模块-Attentive Module
下图显示了Attentive模块的结构，与Transformer中使用的模块相似。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201122221757924.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
注意力模块具有三个输入语句：$Q$，$K$，$V$，即$Q=[e_{i=0}^{n_Q-1}]$，$K=[e_i]_{i=0}^{n_K-1}$，$V=[e_i]_{i=0}^{n_V-1}$，其中，其中 $n_Q$，$n_K$ 和 $n_V$ 表示每个句子中的单词数，而 $e_i$ 表示维数嵌入，$n_K$ 等于 $n_V$。注意力模块首先通过Scaled Dot-Product Attention将查询语句中的每个单词与关键语句中的单词关联，然后将结果应用于值语句，定义为：
$$Att(Q,K)=[softmax(\frac{Q[i]\cdot K^T}{\sqrt{d}})]_{i=0}^{n_Q-1}$$  $$V_{att}=Att(Q,K)\cdot V\in \mathbb{R}^{n_Q\times d}$$
其中 $Q[i]$是查询语句 $Q$ 中的第 $i$ 个嵌入，$V_{att}$ 的每一行（表示为 $V_{att}[i]$）存储值语句中可能与查询语句中的第 $i$ 个单词相关的单词的语义信息。对于每个 $i$，将 $V_{att}[i]$ 和 $Q[i]$ 加在一起，将它们组合成一个包含其联合含义的新表示形式。然后应用层归一化操作，可防止梯度消失或爆炸。接着将具有RELU的前馈网络FFN应用于标准化结果，以便进一步处理融合嵌入，定义为：
$$FFN(x)=max(0,xW_1+b_1)W_2+b_2$$
其中，$x$ 是与查询语句 $Q$ 形状相同的 $2D$ 张量，$W_1$，$b_1$，$W_2$，$b_2$ 是学习参数。$FFN(x)$ 的结果是一个具有与 $x$ 相同形状的 $2D$ 张量，然后将 $FFN(x)$ 残留连接添加到 $x$，最后将融合结果标准化为最终输出。将整个注意力模块表示为：
$$AttentiveModule(Q,K,V)$$

如上所述，注意力模块可以捕获查询语句和关键语句之间的依存关系，并进一步使用依存关系信息将查询语句和值语句中的元素合成为组成表示形式。我们利用Attentive Module的此属性来构造多粒度语义表示以及与依赖项信息的匹配。
## 整理表示
给定 $U_i^0$ 或 $R^0$ （语句 $u_i$ 或响应 $r$ 的单词级嵌入表示），DAM将 $U_i^0$ 或 $R^0$ 用作输入，并分层堆叠Attentive模块以构造 $u_i$ 和 $r$ 的多粒度表示，即表示为：
$$U_i^{l+1}=AttentiveModule(U_i^l,U_i^l,U_i^l)$$  $$R^{l+1}=AttentiveModule(R^l,R^l,R^l)$$
其中 $l$ 的范围是 $0$ 到 $L − 1$，表示不同的粒度级别。通过这种方式，每个话语或响应中的单词会反复发挥作用，以合成越来越多的整体表示形式，我们将这些多粒度表示形式表示为 $U_i^0,...,U_i^L$ 和 $R^0,...,R^L$ 之后。
## Utterance-Response 匹配
给定 $[U_i^l]_{l=0}^L$ 和 $[R^l]_{l=0}^L$ ，在每个粒度级别 $l$ 上构造了两种segment-segment匹配矩阵，即self-attention匹配 $M_{self}^{u_i,r,l}$ 和cross-attention匹配 $M_{corss}^{u_i,r,l}$。其中， $M_{self}^{u_i,r,l}$ 被定义为：
$$M_{self}^{u_i,r,l}=\{U_i^l[k]^T\cdot R^l[t]\}_{n_{u_i}\times n_r}$$
其中，矩阵中的每个元素都是 $U_i^l[k]$ 和 $R^l[t]$ 的点积，第 $k$ 个嵌入在 $U_i^l$ 中，第 $t$ 个嵌入在 $R^l$ 中，反映了 $u_i$中 第 $k$ 个分段和 $r$ 中第 $t$ 个分段之间，在 第$l$ 层粒度的相关性。 cross-attention匹配矩阵基于cross-attention，其定义为：
$$\tilde{U}_i^l=AttentiveModule(U_i^l,R^l,R^l)$$  $$\tilde{R}^l=AttentiveModule(R^l,U_i^l,U_i^l)$$  $$M_{corss}^{u_i,r,l}=\{\tilde{U}_i^l[k]^T\cdot \tilde{R}^l[t]\}_{n_{u_i}\times n_r}$$
其中，我们使用注意力模块将 $U_i^l$ 和 $R^l$ 相互交叉，分别为它们构造两个新的表示形式，分别写为 $\tilde{U}_i^l$ 和 $\tilde{R}^l$。 $\tilde{U}_i^l$ 和 $\tilde{R}^l$ 都隐式捕获了跨语句和响应的语义结构信息。这样，那些相互依存的片段对在表示上彼此接近，并且那些潜在相互依存的对之间的点积可以增加，从而提供了依赖于感知的匹配信息。
## Aggregation
在得到 $M_{self}$ 和 $M_{cross}$ 后，就需要将它们聚合起来，做法是将所有的 $2D$ 匹配矩阵聚合成一个大的 $3D$ 匹配图像 $Q$，具体的聚合方法就是将 $M_{self}$ 和 $M_{cross}$ 所有的矩阵排列起来，所以就增加了一个维度，新的维度（可以称之为深度）大小是 $2(L+1)$ ，具体公式如下：
$$Q=\{Q_{i,k,t}\}_{n\times n_{u_i}\times n_r}$$
其中 $Q_{i,k,t}$ 可以表示为：
$$Q_{i,k,t}=[M_{self}^{u_i,r,l}[k,t]]_{l=0}^L\oplus [M_{corss}^{u_i,r,l}[k,t]]_{l=0}^L$$

聚合成 $3D$ 匹配图像后，采用了两次的 $3D$ 卷积和最大池化去提取特征，在实际试验中，第一次 $3D$ 卷积的输入通道数为 $2(L+1)$ ，输出通道数为 $32$，卷积核的大小是 $[3,3,3]$，步幅为 $[1,1,1]$，最大池化层的核大小是 $[3,3,3]$，步幅为 $[3,3,3]$。第二次 $3D$ 卷积的输入通道数为 $32$，输出通道数为 $16$，卷积核的大小是 $[3,3,3]$，步幅为 $[1,1,1]$，最大池化层的核大小是 $[3,3,3]$，步幅为 $[3,3,3]$。通过卷积和池化提取到特征后（用 $f_{match}(c,r)$ 表示提取后的特征），后面接一层线性层将维度转化成 $1$，用来表示匹配的分数，具体公式如下：
$$g(c,r)=\sigma(W_3f_{match}(c,r)+b_3)$$

其中 $W_3$ 和 $b_3$ 是学习参数，$\sigma$ 是sigmoid函数，如果 $r$ 是 $c$ 的合适候选相应，则给出概率。DAM的损失函数为负对数似然，定义为：
$$p(y|c,r)=g(c,r)y+(1-g(c,r))(1-y)$$  $$L(\cdot)=-\sum_{(c,r,y)\in D}log(p(y|c,r))$$

# 实验结果
实验要求每个比较模型从给定对话上下文 $c$ 的 $n$ 个可用候选中选择 $k$ 个最匹配的响应，然后我们将在 $k$ 个选定的真实响应中的正确回复的召回率作为主要评估指标来计算，记为 $R_n@k=\frac{\sum_{i=1}^ky_i}{\sum_{i=1}^ny_i}$，其中 $y_i$ 是每个候选项的二进制标签。除此之外，我们还使用MAP（Mean Average Precision），MRR (Mean Reciprocal Rank)，Precision-at-one P@1。

## 参数配置
使用的词汇表和词嵌入大小和SMN模型一致。上下文中语句最大设置为 $9$，每个语句最多 $50$ 个单词，并使用word2vec进行词嵌入。我们使用零填充来处理可变大小的输入，并且将FFN中的参数设置为200，与词嵌入大小相同。 我们测试了$1-7$层自注意力层，其中 $5$ 层自注意力层在验证集上获得了最佳分数。试验中两次的 $3D$ 卷积和最大池化，第一次 $3D$ 卷积的卷积核的大小是 $[3,3,3]$，步幅为 $[1,1,1]$，最大池化层的核大小是 $[3,3,3]$，步幅为 $[3,3,3]$。第二次 $3D$ 卷积的卷积核的大小是 $[3,3,3]$，步幅为 $[1,1,1]$，最大池化层的核大小是 $[3,3,3]$，步幅为 $[3,3,3]$。我们使用adam优化器，学习率初始化为 $1e-3$，并在训练过程中逐渐降低，并且批大小为 $256$。

下表显示DAM的评估结果以及所有比较模型：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201122233518572.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
论文还使用Ubuntu语料通过数量分析和可视化分析DAM中的self-attention和cross-attention的工作方式。下图的左侧部分显示了在具有不同语句数量的上下文中，Ubuntu 语料上 $R_{10} @1$ 的变化，右侧提供了在具有不同平均语句长度和self-attention数量的性能比较：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201122233939735.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
下图给出了第 $0$，第 $2$ 和第 $4$ self-attention匹配矩阵，第 $4$ cross-attention匹配矩阵的可视化结果，以及第4层中的self-attention和cross-attention的分布。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201122234258665.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
# 总结
模型存在的不足：
+ 模糊候选响应：候选响应虽然基本上适合于会话上下文，但是还有一些不适当的细节。
+ 逻辑错误：在给定的对话上下文，由于逻辑不匹配，候选响应不合适。

作者认为在训练过程中生成对抗性示例而非随机抽样可能是解决模糊候选和逻辑错误的一个好主意，并且捕获隐藏在对话文本后的逻辑信息也值得在将来进行研究 。