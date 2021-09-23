
# 前言

> [Slot-Gated Modeling for Joint Slot Filling and Intent Prediction](https://aclanthology.org/N18-2118.pdf)
> nlp-paper：[NLP相关Paper笔记和代码复现](https://github.com/DengBoCong/nlp-paper)
> nlp-dialogue：[一个开源的全流程对话系统，更新中！](https://github.com/DengBoCong/nlp-dialogue)
> text-similarity：[文本相似度（匹配）计算模型包](https://github.com/DengBoCong/text-similarity)
> Algorithm：[对ML、DL和数据结构算法进行整理讲解，同时含C++/Java/Python三个代码版本](https://github.com/DengBoCong/Algorithm)
> 说明：阅读原文时进行相关思想、结构、优缺点，内容进行提炼和记录，原文和相关引用会标明出处，引用之处如有侵权，烦请告知删除。
> 转载请注明：DengBoCong

本篇论文是2018年的一篇顶会论文，其提出的结构在当时的联合意图识别(ID)和槽位填充(SF)上实现最好性能，到现在也很值得学习一下。论文提出的Slot-Gated结构，其关注于学习Intent和Slot attention向量之间的关系（其ID和SF的attention权重独立），通过全局优化获得更好的semantic frame。门控机制让我想起16年的一项将线性门控机制应用于卷积结构的工作，同样也是使用了门控机制，提出了GLU结构，门控机制真是很奇妙的一种结构。阅读笔记如下：
[论文阅读笔记：将线性门控机制应用于卷积结构](https://zhuanlan.zhihu.com/p/395977833)

论文主要贡献在于：
+ 提出Slot-Gated方法实现了最好的性能表现。
+ 通过数据集实验表明Slot-Gated的有效性。
+ Slot-Gated有助于分析slot filling和intent的关系。

通过在ATIS和Snips数据集实验，相比于attention模型semantic frame准确率提升了4.2%。在此之前的最佳模型，是用Attention+Rnn对ID和SF联合建模，但是这种方法只是通过一个共同的loss函数 $loss_{total} = loss_{ID}+loss_{SF}$ 隐式地将二者建立关联，而本文提出的Slot-Gated机制则是显式建立联系。下面是ID和SF的示例：
![在这里插入图片描述](https://img-blog.csdnimg.cn/3d30abcf4d724c9ba62e18b0334cf854.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQm9Db25nLURlbmc=,size_20,color_FFFFFF,t_70,g_se,x_16)
# 模型细节
模型使用BiLSTM结构，输入为 $\mathbf{x}=(x_{1},…,x_{T})$，生成前向隐层状态 $\underset{h_{i}}{\rightarrow}$ 和反向隐层状态 $\underset{h_{i}}{\leftarrow}$ ，最终将二者拼接得到 $h_{i}=[\underset{h_{i}}{\rightarrow};\underset{h_{i}}{\leftarrow}]$。模型结构如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/0569045d212945b29e8b12ce61ba44f3.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQm9Db25nLURlbmc=,size_20,color_FFFFFF,t_70,g_se,x_16)
### Slot Filling
如上图a中所示，SF任务是将输入 $\mathbf{x}=(x_{1},…,x_{T})$ 映射成输出 $\mathbf{y}=(y_{1}^{S},…,y_{T}^{S})$ 。对于每个步长的输入word对应的隐层 $h_{i}$ ，首先计算slot context vector $c_{i}^{S}$ （实际上是self-attention，对应上图中的slot attention）：
$$c_{i}^{S}=\sum_{j=1}^Ta_{i,j}^Sh_j$$
其中，$\alpha_{i,j}^{S}$ 是attention score：
$$\alpha_{i,j}^{S}=\frac{exp(e_{i,j})}{\sum_{k=1}^Texp(e_{i,k})}\\e_{i,k}=\sigma(W_{he}^Sh_k+W_{i,e}h_i)$$
其中，$\sigma$是激活函数，$W_{he}^S$是权重矩阵，接着使用 $h_{i}$ 和 $c_{i}^{S}$ 做softmax得到第 $i$ 个word对应的slot label $y_{i}^{S}$
$$y_i^S=softmax(W_{hy}^S(h_i+c_i^S))$$

> $c_i^S\in\mathbb{R}^{bs*T}$，且和 $h_j$  shape一致。
> $e_{i,k}\in\mathbb{R}^1$，$e_{i,k}$ 计算的是 $h_k$ 和输入向量 $h_i$ 之间的关系。
> 作者TensorFlow源码 $W_{ke}^Sh_k$ 用的卷积实现，而 $W_{ie}^Sh_i$ 用的线性映射_linear()。
### Intent Prediction
Intent context vector $c^{I}$ 的计算方式类似于 $c_{i}^{S}$ ，区别在于预测意图时只使用BILSTM最后一个隐层状态 $h_T$：
$$y^I=softmax(W_{hy}^I(h_T+c^I))$$

Attention具体细节见：Attention-Based Recurrent Neural Network Models for Joint Intent Detection and Slot Filling

### Slot-Gated Mechanism
Slot-Gated的主要目的是使用intent context vector来改善slot-filling的表现，结构如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/7a10c75901fc48efae2dedf402542b08.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQm9Db25nLURlbmc=,size_20,color_FFFFFF,t_70,g_se,x_16)
通过引入了一个额外的gate，利用intent上下文向量来建模slot-intent关系，以提高槽填充性能。 首先，组合slot上下文向量 $c_i^S$ 和intent上下文向量 $c^I$ 以通过Figure 3中所示的slot gate：
$$g=\sum v\cdot tanh(c_i^S+W\cdot c^I)$$
> $c^I,c_i^S,v\in\mathbb{R}^d，$d$ 是输入向量 $h$ 的维度。$
> $g_i\in\mathbb{R}^1$，获得 $c^I$ 的权重
> 论文源码使用的是：$g=\sum v\cdot tanh(c_i^S+W\cdot (c^I+h_T))$

其中 $v$ 和 $W$ 分别是可训练的向量和矩阵。 在一个时间步骤中对元素进行求和。$g$ 可以看作联合上下文向量（$c_i^S$ 和 $c^I$）的加权特征。加入 $g$：
$$y_i^S=softmax(W_{hy}^S(h_i+c_i^S\cdot g))$$

为了比较slot gate的效果，本文还提出了一个去掉slot attention的结构，见Figure 2 右图，公式如下：
$$g=\sum v\cdot tanh(h_i+W\cdot c^I)\\y_i^S=softmax(W_{hy}^S(h_i+h_i\cdot g))$$

### Joint Optimization
模型的联合目标函数为：
$$p(y^{S},y^{I}|\mathbf{x})\\=p(y^{I}|\mathbf{x})\prod_{t=1}^{T}p(y^{S}_{t}|\mathbf{x})\\=p(y^{I}|x_{1},…,x_{T})\prod_{t=1}^{T}p(y^{S}_{t}|x_{1},…,x_{T})$$
其中，$p(y^{S},y^{I}|\mathbf{x})$ 是 SF和ID的联合条件概率。

# 实验结果
![在这里插入图片描述](https://img-blog.csdnimg.cn/ff7e720e02e04394a05b9844553d259f.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQm9Db25nLURlbmc=,size_20,color_FFFFFF,t_70,g_se,x_16)
根据Table 3，两种Slot-Gated模型的性能均优于baselines，但是在ATIS数据集上intent attention最优，在Snips上full attention最优，原文是这么说明的：
> Considering different complexity of these datasets, the probable reason is that a simpler SLU task, such as ATIS, does not require additional slot attention to achieve good results, and the slot gate is capable of providing enough cues for slot filling. On the other hand, Snips is more complex, so that the slot attention is needed in order to model slot filling better (as well as the semantic frame results).

作者特意强调slot-gate模型在frame acc上的改善，因为frame acc是同时衡量两个任务的指标。
> It may credit to the proposed slot gate that learns the slot-intent relations to provide helpful information for global optimization of the joint model.

# 结论
本文提出的一种将intent detect和slot filling显示关联学习的架构，并实验证明有效，说明可以深挖如何在ID和SF显示关联上设计更好的架构，例如本文是单向的门结构，ID结果输入到SF，是否能够将二者相互关联，SF结果也可以输入到ID，或者提出更优雅表征显式关系的结构。