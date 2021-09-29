
# 前言

> [Retrieve & Memorize: Dialog Policy Learning with Multi-Action Memory](https://arxiv.org/pdf/2106.02317.pdf)
> nlp-paper：[NLP相关Paper笔记和代码复现](https://github.com/DengBoCong/nlp-paper)
> nlp-dialogue：[一个开源的全流程对话系统，更新中！](https://github.com/DengBoCong/nlp-dialogue)
> text-similarity：[文本相似度（匹配）计算模型包](https://github.com/DengBoCong/text-similarity)
> Algorithm：[对ML、DL和数据结构算法进行整理讲解，同时含C++/Java/Python三个代码版本](https://github.com/DengBoCong/Algorithm)
> 说明：阅读原文时进行相关思想、结构、优缺点，内容进行提炼和记录，原文和相关引用会标明出处，引用之处如有侵权，烦请告知删除。
> 转载请注明：DengBoCong

本文的定位是在对话系统的子任务Dialogue policy learning，其目标是为了提高在task-oriented对话系统中的context-to-response的质量。作者针对现有解决对话中的one-to-many问题的模型方法进行优化，提高候选actions的多样性和对话历史表示的隐藏信息量，从而提高response生成的质量。提出的模型可以分为两部分，context-aware neural retrieval module (CARM) 和 memory-augmented multi-decoder network(MAMD)。

CARM是使用预训练语言模型将对话历史和 state 转换为每个样本的context表示，然后使用 context vector 和其他样本的 latent space 表示之间的距离度量来检索多个候选 system actions，这样是的获得的 system actions更加多样化且包含context信息。这些system actions并不是直接用于模型的解码，而是现将其编码成一个memory bank，然后使用MAMD网络在训练过程中动态地配合memory bank生成system actions。除此之外，为了更好的提升模型的稳定性，作者在训练过程中加入了 random sampling机制，即以一定概率随机替换候选的 system actions。

# 模型细节
## Context-Aware Retrieval Module
首先先定义一下符号，$X_t=\{U_1,...,U_{t-1},R_{t-1},U_t\}$ 表示第 $t$ 轮的对话context， 其中 $U_i=u_1u_2,...,u_{m_i}$ 和 $R_i=r_1r_2,...,r_{n_i}$ 分别表示第 $i$ 句用户和系统的response，$B_t=b_1b_2...b_p$ 和 $A_t=a_1a_2...a_q$ 分别表示 belief state 和system actions。模型的目标就是基于context $X_t$ 和belief state $B_t$，生成第 $t$ 轮的 system actions $A_t$ 和 system response $R_t$。模型结构如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/d3d8a1b3cc0449c0b571325b650a6330.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQm9Db25nLURlbmc=,size_20,color_FFFFFF,t_70,g_se,x_16)
首先是将context $X_t$ 和 belief state $B_t$ 级联起来喂入BERT中，如下：
$$H=BERT([CLS]\oplus B_t\oplus[SEP]\oplus X_t)$$
然后用 $[CLS]$ 的隐向量作为获取的上下文特征表示，从而用于计算相似的对话上下文，相似计算使用 $L_2$ 距离，如下：
$$L_2(h_i^{CLS},h_j^{CLS})=||h_i^{CLS}-h_j^{CLS}||_2$$
选出 $k$ 个上下文关联的actions表示为 $\{\bar{A}_1,\bar{A}_2,...,\bar{A}_k\}$。为了让BERT更加适应于当前任务，作者使用actions预测任务对BERT进行预训练：
$$p(y|B_t,X_t)=classifier(h^{CLS})$$
其中， $y\in\mathbb{R}^D$ 是system actions的one-hot形式，$classifier$ 是简单的线性分类器。

##  Memory-Augmented Multi-Decoder Network
#### Encoding Module
Encoder是Bi-GRU，首先将用户当前的语句、前一个system response、前一个 belief state和候选system actions 分别进行编码：
$$H_u=Encoder(U_t)$$
$$H_{pre\_r}=Encoder(R_{t-1})$$
$$H_{pre\_b}=Encoder(B_{t-1})$$
$$M_t=Encoder_M(\bar{A}_1\oplus\bar{A}_2\oplus...\oplus\bar{A}_k)=\{m_1,...,m_k\}$$
#### Belief State Generation
然后使用上面得到的Encoder 隐状态向量来生成第 $t$ 轮的belief state $B_t$：
$$s_{\tau}=Attn(h_{\tau-1},H_u,H_{pre\_r},H_{pre\_b})$$
$$c_{\tau}=[s_{\tau}\oplus e(b_{\tau-1})]$$
$$p(b_{\tau}|b_{1:\tau-1}),h_{\tau}=Dec_b(c_{\tau},h_{\tau-1},H_{pre\_b})$$
其中，$e(b_{\tau-1})$ 是前一个token的embedding，$h_{\tau-1}$ 是最后一步解码的隐状态向量，$h_0=0$，$p(b_{\tau}|b_{1:\tau-1})$ 是整个词汇的分布，Attn的详细计算如下：
$$a_i=tanh(W[h\oplus H_i])$$
$$a_i=Softmax(a)$$
$$CatAttn(h,H)=\sum_{i=1}^na_iH_i$$
$$h_a=CatAtten(h,H_a)$$
$$h_b=CatAtten(h,H_b)$$
$$h_c=CatAtten(h,H_c)$$
$$Attn(h,H_a,H_b,H_c)=[h_a\oplus h_b\oplus h_c]$$
$Dec_b$ 是一个使用了复制数据增强机制的belief state 解码器，如下：
$$h_t=GRU(c_t,h_{t-1})$$
$$p_{vocab}=Softmax(W_vh_t)$$
$$s_i=h_t^Ttanh(W_cH_i)$$
$$p_{copy}=Softmax(s)$$
$$p_{final}(w)=p_{vocab}(w)+\sum_{i:X(i)=w}p_{copy}^i$$
$$Dec(c_t,h_{t-1},H)=p_{final},h_t$$
该阶段使用交叉熵进行损失计算，并获得hidden state为$H_b=\{h_0,h_1,...,h_p\}$。
#### Memory-Augmented Action Generation
![在这里插入图片描述](https://img-blog.csdnimg.cn/3e169aace4bb40729424cc3e42c8448a.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQm9Db25nLURlbmc=,size_20,color_FFFFFF,t_70,g_se,x_16)
如上图所示，生成system actions如下，首先计算 $s_\tau$：
$$s_{\tau}=Attn(h_{\tau-1},H_u,H_{pre\_r},H_b)$$
然后使用 $h_{\tau-1}$ 和 memory bank $M_t$ 进行注意力计算：
$$a_\tau^i=tanh(W[h_{\tau-1}\oplus m_i])$$
$$a_\tau =Softmax(a_\tau)$$
$$v_\tau=\sum_{i=1}^ka_\tau^im_i$$
$$c_\tau=[s_\tau\oplus e(a_{\tau-1})\oplus e(DB_t)\oplus v_\tau]$$
$$p(a_\tau|a_{1:\tau-1}),h_\tau=Dec_a(c_\tau,h_{\tau-1},H_b)$$

其中，$e(a_{\tau-1})$ 是前一个token的embedding，$e(DB_t)$ 是检索出的实体embedding
#### Response Generation
最后使用各hidden state生成response：
$$s_{\tau}=Attn(h_{\tau-1},H_u,H_b,H_a)$$
$$c_\tau=[s_\tau\oplus e(r_{\tau-1})]$$
$$p(r_\tau|r_{1:\tau-1}),h_\tau=Dec_\tau(c_\tau,h_{\tau-1},H_b)$$

最后损失函数，上述三个generation的损失函数相加进行优化。
# 实验结果
![在这里插入图片描述](https://img-blog.csdnimg.cn/ae81a466bd9f4a24a83cbdc446269040.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQm9Db25nLURlbmc=,size_20,color_FFFFFF,t_70,g_se,x_16)
![在这里插入图片描述](https://img-blog.csdnimg.cn/164c22ed9b014ad6877ecb0364c1bbab.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQm9Db25nLURlbmc=,size_11,color_FFFFFF,t_70,g_se,x_16)


