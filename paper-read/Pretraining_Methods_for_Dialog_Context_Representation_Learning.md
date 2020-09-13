
<font color=#999AAA >提示：阅读论文时进行相关思想、结构、优缺点，内容进行提炼和记录，论文和相关引用会标明出处。</font>

# 前言

> 标题：Pretraining Methods for Dialog Context Representation Learning
> 原文链接：[Link](https://arxiv.org/pdf/1906.00414.pdf)
> 转载请注明：DengBoCong

# Abstract
本文考察了各种用于学习对话上下文表示的无监督预训练目标， 提出了两种新颖的对话上下文编码器预训练方法，并研究了四种方法。使用MultiWoz数据集对每个预训练目标进行了微调，并在一组下游对话任务上进行了评估，并观察到了出色的性能改进。 进一步的评估表明，我们的预训练目标不仅可以带来更好的性能，而且可以带来更好的收敛性，并且模型需要的数据更少，并且具有更好的领域通用性。

# Introduction
目前预训练方法仍处在起步阶段，我们仍然不能完全了解他们的性质。大多数方法都是基于语言模型的，给一个句子，预测当前词，下一个词或者被mask的词。如Word2Vec，Glove，ELMO等。这些方法将自然语言看作是word token的流，需要复杂的模型利用大规模的语料库和庞杂的计算来发现更高级别的依赖关系。BERT模型也是基于语言模型，但是加入了句子对级别的信息，预测两句话是否是连续的。这种方法在预训练时利用了语句之间的关系。但是，在对话上下文建模这种存在多轮的依赖关系的任务上还并没有行之有效的预训练方法，于是本文在这个方面做了一些尝试。本文目的就是研究几个预训练话语级语言表示的方法，本文迈出了建立对话系统预训练方法系统分析框架的第一步。

评估预训练方法的四个假设：
+ 预训练能够在整个可用数据集上进行微调，且提升下游任务
+ 预训练结果需要更好的收敛
+ 预训练将在有限的数据下表现出色
+ 预训练有助于领域通用化

对话与其他文本的区别：
+ 对话必须是语句之间连贯的，并在多轮上达到一个交际的目的。
+ 对话在本质上是互动的，说话者之间有反馈，而且说话者轮流进行发言。

本文的主要贡献：
+ 针对对话上下文表示研究四个不同的预训练方法，包括两个新的方法
+ 在四个下游任务上，综合分析预训练对对话上下文表示的影响

# Related Work
这项工作与NLP系统的辅助多任务学习和带预训练的迁移学习的研究紧密相关。
## Training with Auxiliary Tasks
结合有用的辅助损失函数来补充主要目标已被证明可以改善深度神经网络模型的性能。一些辅助损失函数专门设计来提高特殊任务的性能。在一些案例中，辅助函数被用来提升模型的泛化能力。经过适当的辅助任务预训练后，模型可以捕获更长的依赖关系。
## Transfer Learning with Pretraining
基本过程通常是首先在无监督目标的海量文本数据上预训练功能强大的神经编码器。 第二步是使用更小的域内数据集对特定的下游任务微调此预训练模型。ELMo使用BiLSTM网络来训练双向语言模型来同时预测前一个词和后一个词。OpenAI的GPT使用Transformer网络和BERT进行了两个目标的同时训练：掩蔽语言模型和下一句预测。每个模型均已在GLUE基准上展示了最新的结果。这些利用大规模预训练的模型优于仅使用域内数据的系统。用于学习从输入文本中提取话语级别信息的预训练方法的工作很少。BERT中的下一句话预测损失是朝着这个方向迈出的一步。尽管这些预训练方法擅长于对顺序文本进行建模，但它们并未明确考虑对话的独特话语级功能。因此，我们在研究预训练目标时采取了第一步，以提取对话上下文的更好的话语级表示形式。
# Pretraining Objectives
本文定义了一种强有力的表示形式，它可以捕获整个对话历史中的话语级信息以及构成该历史的话语中的话语级信息，在本文的定义下，当表示允许模型在各种下游任务上表现更好时，表示就足够通用了。
+ 一个任意T轮对话（对话历史）的表示符号：$c = [u_1,...,u_t]$，$u_i$是一个话语。
+ 对话回复$R = {r_1,...,r_M}$

## Next-Utterance Retrieval（NUR-检索下一句话）
NUR的目的就是在$k$个候选回复中选择正确的下一句话。对于此任务，本文使用分层编码器来生成对话上下文的表示，方法是首先通过双向长期短期记忆网络（biLSTM）独立运行每个话语，然后使用所得的话语表示来生成整个对话上下文的表示。给定$[u_1，... u_{T-1}]$，NUR的任务是从R中选择正确的下一个话语$u_T$。损失运算公式如下：

$$
\hat{u_i}=f_u(u_i), i\in [1,T-1]\\
[h_1,...h_{T-1}]=f_c(u_1,...\hat{u}_{T-1})\\
r_{gt} = f_r(u_T)\\
r_{j} = f_r(r_j),r_j\sim p_n(r)\\
a_{gt} = (h_{T-1})^{T}r_{gt}\\
a_{j} = (h_{T-1})^{T}r_{j}
$$

总损失如下：

$$
L = -log_p(u_T|u_1,...u_{T-1})\\
=-log(\frac{exp(a_{gt})}{exp(a_{gt}+\sum_{j=1}^{K}exp(a_j)})
$$

## Next-Utterance Generation（NUG-生成下一句话）
给定对话历史，根据对话历史生成下一句话。预训练时使用分层Encoder-Decoder结构，在进行下游任务时，仅使用Encoder。对话上下文和下一个话语被编码为式8，最小化损失为式9：

$$
L = -log_p(u_T|u_1,...u_{T-1})\\
=-\sum_{k}^{N}log_p(w_k|w<k,h_{T-1})
$$

##  Masked-Utterance Retrieval（MUR-根据mask的对话历史检索下一句话）
与NUR相同，给定对话历史，从$K$个候选回复中选择正确的下一句话，区别
+ 对话历史中的一句话被随机选择的另一句话替换。
+ 用替换掉的句子的表示作为对话历史的表示。

替换的语句索引为$t$，且是在对话部分中随机采样的

$$
t \sim Uniform[1,T]
$$

其中

$$
\hat{u_i}=f_u(u_i), i\in [1,T]\\
[h_1,...h_T]=f_c(u_1,...\hat{u}_T)\\
r_{gt} = f_r(u_T)\\
r_{j} = f_r(r_j),r_j\sim p_n(r)\\
a_{gt} = (h_T)^{T}r_{gt}\\
a_{j} = (h_T)^{T}r_{j}
$$

总损失：

$$
L = -log_p(u_T|u_1,...,q,...u_{T-1})\\
=-log(\frac{exp(a_{gt})}{exp(a_{gt}+\sum_{j=1}^{K}exp(a_j)})
$$

## Inconsistency Identification（InI-识别不一致语句）
识别一段对话历史中不一致的句子。输入是一段对话历史，其中的一句被随机替换掉，模型需要找到被替换的是哪一句。

$$
L = -log_p(t|u_1,...,q,...u_T)\\
=-log(\frac{exp(a_t)}{\sum_{j=1}^{T}exp(a_i)})
$$

其中

$$
\hat{u_i}=f_u(u_i)),i\in [1,T]\\
[h_1...h_T]=f_c(\hat{u_1},...\hat{u_T})\\
a_i=(h_T)^{T}h_i,i\in[1,T]
$$

这个任务的目标是建模单个语句的表示和对话上下文的全局表示的一致性。
# Downstream Tasks
本文选择了以下四个下游任务来测试预训练表示的通用性和有效性。实验数据用的是MultiWoz，其中8422个对话用于训练，1000个用于验证，另外1000个用于测试。
+ 预测对话状态：这是一个多分类任务，给定对话历史，预测当前的对话状态。对话状态由27种实体的1784个值的one-hot向量拼接而成。这个任务度量了系统维护完整且准确的对话上下文状态表示的能力。由于输出有1784维，这就要求预训练的对话历史表示模型必须有足够强的概括性，才能对对话状态进行准确的预测。
+ 预测对话行为：与上个任务类似，是一种多分类任务。给定对话历史，预测系统下一步可能采取的动作，输出是一个32维的对话行为向量。
+ 生成下一句话
+ 检索下一句话

# Experiments
每个模型都训练了15个epoch，选择在验证集上表现最好的模型用于测试。实验中所用参数如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200913193940733.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
为了更直接的评估预训练过程中目标设置不同的差异，这里的预训练和fine-tune都是在同一数据集上进行的。在完整数据集上的表现：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200913194136216.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
该实验是为了测试预训练是否对下游任务有用。表中的第一行对每个任务的模型进行随机初始化。如表一所示，预训练表示展示出了它的有效性和通用性。通过非监督的预训练，模型产生的对话表示提升了很多下游任务的性能。通用性体现在这些下游任务都受益于相同的预训练模型。

在对话行为预测（DAP）和下一句话生成（NUG）任务上，以识别不一致语句（InI）为目标的预训练模型效果最好。这可能是因为在序列生成模型中全局上下文表示和局部话语表示同样重要。

在对话行为预测（DAP）任务上，以识别不一致语句（InI）和根据mask的对话历史检索下一句话（MUR）的得分都远远高于基线和其他方法，这可能是因为这两种方法都是训练来学习每个话语的表示，而不仅仅是一个整体的上下文表示。

在检索下一句话（NUR）任务上，以生成下一句话（NUG）为目标进行预训练时效果最好，这可能是因为生成下一个话语必须捕获的信息与检索下一个话语所需的信息类似。

+ 本文设置了实验观察预训练表示对下游任务在收敛性上的影响。实验证明，预训练过的模型能更快地收敛到更好的效果。
+ 一个好的预训练模型应该在下游任务中仅有少量数据的微调的情况下，也能达到很好地效果。本文做了实验验证在微调数据仅有(1%, 2%, 5%, 10% and 50%)时，在下游任务上的表现。
+ 该实验模拟了在下游任务中添加新域时的场景，假设在所有领域都存在大量的无监督的未标记数据，而对于下游任务仅有50个（0.1%）域内的标记数据和1000个（2%）新域的标注数据。在域内数据上做测试，实验证明预训练模型产生了更通用的表示，并促进了域的泛化。

# Conclusions
在这篇文章中，提到了4种无监督的预训练目标来学习对话的上下文的表示，并在有限的微调数据和域外数据的条件下，证明了预训练模型对于提高下游任务性能方面的有效性。其中根据mask的对话历史检索下一句话和不一致语句识别是本文提出的两种新的目标。

在本文中，无监督预训练被证明能够有效地学习对话上下文的表示。也就是说在有大量未标记的对话数据时，可以采取本文中的几种方法进行预训练。尤其是在标注数据量比较少的情况下。