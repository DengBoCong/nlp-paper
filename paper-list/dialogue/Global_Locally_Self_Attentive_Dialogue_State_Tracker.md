# 前言

> 标题：Global-Locally Self-Attentive Dialogue State Tracker\
> 原文链接：[Link](https://arxiv.org/pdf/1805.09655.pdf)\
> Github：[NLP相关Paper笔记和实现](https://github.com/DengBoCong/nlp-paper)\
> 说明：阅读论文时进行相关思想、结构、优缺点，内容进行提炼和记录，论文和相关引用会标明出处，引用之处如有侵权，烦请告知删除。\
> 转载请注明：DengBoCong

# 介绍
对话状态跟踪（在对话上下文中估计用户目标和请求）是面向任务的对话系统的重要组成部分。在本文中，提出了“全局-局部自注意力对话状态追踪”（GLAD），该学习器使用全局本地模块来学习用户话语的表示和以前的系统动作。模型使用全局模块在不同类型（称为插槽）的对话状态的估计量之间共享参数，并使用本地模块学习特定于插槽的特征。DST中的状态（state）通常由一个请求（request）和联合目标（joint goals）的集合组成。请求即请求系统返回所需信息（例如：request(address)），目标即用户想要完成的动作（例如：inform(food=french)），请求和目标可以用槽位-值对（slot-value pair）来表示（(food,french)， (request, address)）。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201019151853610.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
作者认为传统的DST极其依赖Spoken Language Understanding（SLU），而依赖SLU极易导致错误的积累，且通常的SLU对比较少见的slot-value对容易分错，为了解决这些问题，本文提出了一种全局-局部自注意力对话状态追踪方法（Global-Locally Self-Attentive Dialogue State Tracker， GLAD），使用全局模块在预测每个slot的预测器之间共享参数，而使用局部模块来学习每个slot的特征表示。通过这种设计，能够使GLAD在更小的训练数据上泛化到更少见的slot-value对。

# 模型结构
模型整体框架如下，主要包含两个模块：encoder模块和scoring模块。在encoder模块中，可以看到分别针对system action，user utterance和候选slot-value对这三者进行encoder。注意，后面讨论的都是针对某个具体的候选slot-value对。scoring模块包含两个打分器，分别给当前话语和之前轮次系统行为对当前状态预测的贡献打分。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201019232904710.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
## 全局-局部自注意力编码器
在针对三部分输入进行encoding的过程中，使用的都是Global-Locally Self-Attentive Encoder，这个encoder的框架如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201019233304219.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
基本可以分为如下几个步骤：
+ 将输入分别通过global和local的BiLSTM得到各自的表征，全局与局部双向LSTM类似，输入embedding后的序列，得到编码输出，区别在于全局双向LSTM的参数在不同slot间共享，而每个slot都有各自的局部双向LSTM参数。
$$H^g=biLSTM^g(X)\in \mathbb{R}^{n \times d_{rnn}}$$  $$H^s=biLSTM^s(X)\in \mathbb{R}^{n \times d_{rnn}}$$
+ 将global和local的进行加权融合，两个LSTM的输出通过一个混合层相结合，形成全局-局部编码的最终输出，其中 $\beta^s$ 是学习到的一个0-1之间的权重值（每个slot的 $\beta^s$ 不同）。
$$H = \beta^sH^s+(1-\beta^s)H^g\in \mathbb{R}^{n \times d_{rnn}}$$
+ 类似的，针对global和local的分别使用attention机制，并最后做融合。其中，对 $H$ 的每个元素 $H_i$ 计算一个标量attention分值，通过softmax标准化，再对 $H_i$ 做加权求和，得到最终的输出。
$$a_{i}^{g}=W^gH_i+b^g\in \mathbb{R}$$  $$p^g=softmax(a^g)\in \mathbb{R}^n$$  $$c^g=\sum_ip_{i}^gH_i\in  \mathbb{R}^{d_{rnn}}$$

$$a_{i}^{s}=W^sH_i+b^s\in \mathbb{R}$$  $$p^s=softmax(a^s)\in \mathbb{R}^n$$  $$c^s=\sum_ip_{i}^sH_i\in  \mathbb{R}^{d_{rnn}}$$
最终的全局-局部自注意力表示为：
$$c=\beta^sc^s+(1-\beta^s)c^g\in\mathbb{R}^{n \times d_{rnn}}$$
+ 注意，encoder会输出两个东西，$H$ 和 $c$ 都会在下面的scoreing module中使用到：

## Encoding module
分别针对三个输入利用上面的encoder进行encoding，其中，用 $U$ 表示当前话语的embedding序列， $A_j$ 表示之前的第 $j$ 个系统行为， $V$ 表示当前考虑的slot-value对，以上述全局-局部自注意力编码器为基础的编码模块的输出为：
$$H^{utt},c^{utt}=encode(U)$$  $$H_j^{act},c_j^{act}=encode(A_j)$$  $$H^{val},c^{val}=encode(V)$$

## Scoring Module
分为三个步骤：
+ 通过类似attention机制进行utterance与slot-value的匹配和打分，具体而言，当前用户话语对于当前slot-value对是否在当前轮次中的贡献是用户直接表述出来的（比如：how about a French restaurant in the centre of town?）。针对这种情况，使用当前slot-value对表示 $c^{val}$ 对 $H^{utt}$ 进行attention并加权求和，用所得结果为该slot-value对打分。
$$a_i^{utt}=(H_i^{utt})^Tc^{val}\in\mathbb{R}$$  $$p^{utt}=softmax(a^{utt})\in\mathbb{R}^m$$  $$q^{utt}=\sum_ip_i^{utt}H_i^{utt}\in\mathbb{R}^{d_{rnn}}$$  $$y^{utt}=Wq^{utt}+b\in\mathbb{R}$$
+ 同样地，先进行system action与utterance之间的融合，尔后再与slot-value进行匹配。具体而言，而当当前用户话语没有呈现足够信息时，对当前轮次状态的推断则需要考虑之前轮次的系统行为（例如用户只回答了“yes”）。针对这种情况，采取与上述attention过程类似的思路对过往轮次系统行为对当前状态推断的贡献打分。
$$a_j^{act}=(C_j^{act})^Tc^{utt}\in\mathbb{R}$$  $$p^{act}=softmax(a^{act})\in\mathbb{R}^{l+1}$$  $$q^{act}=\sum_ip_j^{act}C_j^{act}\in\mathbb{R}^{d_{rnn}}$$  $$y^{act}=(q^{act})^Tc^{val}\in\mathbb{R}$$
+ 最后打分由两部分分数加权求和，并经过sigmoid标准化。
$$y=\sigma(y^{utt}+wy^{act})\in\mathbb{R}$$

# 实验
+ 展示了GLAD与以前的最新模型相比的性能
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020102011100272.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
+ 我们在开发套件上进行了分解实验，以分析GLAD不同组件的有效性。
   + 时间顺序对于状态跟踪很重要
   + Self-attention可实现特定于插槽的强大功能学习
   + Global-local共享可改善目标跟踪
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201020111233472.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
这是一个示例，其中self-attention模块专注于话语的相关部分。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201020111654205.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
下图展示了，在训练数据中显示了GLAD的性能以及这两种共享变体在不同出现次数上的表现。对于具有大量训练数据的槽值对，模型之间没有明显的性能差异，因为有足够的数据可以概括
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201020112030255.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
下图展示GLAD的预测示例
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201020112258567.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
# 总结
GLAD的核心是Global-Locally自注意编码器，其全局模块允许在插槽之间共享参数，而本地模块则允许学习特定于插槽的特征。这使GLAD可以在很少的训练数据的情况下对稀有的插槽值对进行概括。global和local的思想值得一些借鉴。