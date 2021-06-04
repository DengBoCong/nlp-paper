# 前言

> [TOD-BERT: Pre-trained Natural Language Understanding for Task-Oriented Dialogue](https://arxiv.org/pdf/2004.06871.pdf)\
> nlp-paper：[NLP相关Paper笔记和代码复现](https://github.com/DengBoCong/nlp-paper)\
> nlp-dialogue：[一个开源的全流程对话系统，更新中！](https://github.com/DengBoCong/nlp-dialogue)\
> text-similarity：[文本相似度（匹配）计算模型包](https://github.com/DengBoCong/text-similarity)\
> 说明：阅读原文时进行相关思想、结构、优缺点，内容进行提炼和记录，原文和相关引用会标明出处，引用之处如有侵权，烦请告知删除。\
> 转载请注明：DengBoCong

众所周知，预训练BERT语言模型在许多NLP任务重大放异彩，用来文本内容和语义的表征学习很有效果，而且大大降低了下游任务的训练时间。但是由于普通文本和任务型对话之间的语言模式的潜在差异，使得现在的预训练语言模型在实际使用中作用不大。

至于对话领域的预训练语言模型，大多数都用的是开放域的数据集来训练，例如Reddit或Twitter，用于对话响应的生成。而对于任务导向型对话预训练语言模型来说，还没有效果很好或者说是开源了的好模型，所以本篇论文针对面向任务的对话系统，结合现有 BERT 基础架构提出了一个新的预训练目标。

TOD-BERT带来了如下的成果：
+ 在任务导向型对话的IR(意图识别)，DST(对话状态追踪)，DAP（对话行为预测），RS（响应选择）这四个下游任务中，达到了SOTA的效果；
+ 实验证明了，TOD-BERT具有较强的few-shot能力，可以缓解面向任务对话的数据稀缺的问题；
+ TOD-BERT在训练时，把BERT中NSP（预测下一个句子是否是下一句）替换为RCL（Response contrastive loss响应对比损失）；

论文提供的源码以及我应用在对话系统中的实践仓库如下：
[TOD-BERT](https://github.com/jasonwu0731/ToD-BERT)
[nlp-dialogue](https://github.com/DengBoCong/nlp-dialogue)

# 模型细节
## 数据集
首先我们需要来看一下，模型使用了九种数据集，都是多轮对话，在数据集上也取得了很好的效果，使用多领域多轮对话也是因为坐着极力想证明，做任务导向型的对话所用的预训练模型，一定要用任务导向型的语料库来训练效果才好。
> This paper aims to prove this hypothesis: self-supervised language model pre-training using task-oriented corpora can learn better representations than existing pre-trained models for task-oriented downstream tasks.

使用语料如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210603205457757.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70)
## 模型结构
预训练语言模型方面，选用的是Base BERT（不过文中提到使用其他语言模型也是可以的，选用BERT只是因为它比较火，事实上论文的源码也是使用了很多其他的语言模型），损失函数方面使用的是MLM和RCL（后面详细说明）。

还有一个比较特别的就是TOD-BERT为用户和系统引入了两个特殊的token来模拟相应的对话行为，即在用户的utterance前加入[USR]，在系统的utterance前加入[SYS]。从而能够将一个多轮对话的所有的utterance连接起来，变成一个序列，结构如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210603210649847.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70)
## 损失函数
首先是MLM，熟悉BERT的应该都知道，我在这里就提一下：
$$ L_{mlm}=-\sum_{m=1}^MlogP(x_m) $$

其中，$M$ 为被mask的token的总数，$P(x_m )$ 是token $x_m$ 在词汇表上预测的概率。

接下来重点看一下RCL，它有如下几个优点：
+ 能够学到[CLS]这个token的更好的表示，这对于所有下游任务都是必不可少的；
+ 刺激模型去捕获潜在的对话顺序、结构信息和响应相似性；

具体做法是，采用双编码器方法，模拟了多个负样本。有一批对话 $\{D_1,…,D_b \}$，随机选择的回合 $t$ 分割每个对话，例如，$D_1$ 将被分割成两部分，一部分是上下文 $S_1^1,U_1^1...S_t^1,U_t^1$，另一部分是是响应 $S_{(t+1)}^1$ ，其中 $S$ 为system的话语，$U$ 是user的话语。用TOD-BERT分别编码所有上下文和它们对应的响应，通过从 $b$ 对话中获得输出[CLS]的表示得到了一个上下文矩阵 $C$ 和一个响应矩阵 $R$，将同一批次中的其他响应视为随机选择的负样本，就得到了损失函数：
$$L_{rcl}=-\sum_{i=1}^blogM_{i,i}$$  $$M=Softmax(CR^T)\in \mathbb{R}^{b\times b}$$
将batch size增加到一定程度，可以在下游任务上获得更好的性能，特别是在响应选择方面。Softmax函数规范化每行的向量。在文章的设置里，增加batch size也意味着改变对比学习中的正负例比。batch size是一个能够被硬件限制的超参数。作者在预训练阶段也尝试了不同的负抽样策略，如局部抽样，但与随机抽样相比没有显著变化。

模型最后的损失函数，是MLM和RCL的加权和，但文章中直接进行了相加，使用AdamW优化器训练TOD-BERT，在所有层和注意权重上的drop率为0.1，使用GELU激活函数。模型用一个延迟开发集的困惑分数实现早停策略(early-stopped)，batch-size 大小为32，每个序列最大长度为512个token。

# 下游任务
论文中为了保证和原始BERT比较的公平性，尽可能的没有添加其他组件，目的是为了证明用任务导向型语料训练出来的模型要比一般语料库训练的模型在任务导向型对话领域更强，下面提一下在意图识别、对话状态追踪、对话行为预测、响应选择这四个下游任务的应用方式：
+ **意图识别**：意图识别任务是一个多分类问题，输入一个句子 $U$，然后模型在超过 $i$ 个可能的意图中预测一个单一的意图，公式如下。其中，$F$ 是预训练语言模型，我们用它的[CLS]嵌入作为输出的表示，$W_1\in  \mathbb{R}^{(I\times d_B)}$ 是一个可训练的线性映射。模型在预测的分布 $P_{int}$和真实意图标签之间使用交叉熵损失来训练。
$$P_{int}=Softmax(W_1(F(U)))\in \mathbb{R}^I$$
+ **对话状态追踪**：对话状态追踪可以被看作一个使用预定义主体的多分类问题。与意图不同，我们使用对话的历史 $X$（一系列的话语）作为输入，然后模型预测每一个对话回合中每一个(域，槽)对的槽值。每一个对应的值 $V_i^j$（表示第 $j$ 个(域，槽)对的第 $i$ 个值）被传递到预训练的模型中，并在训练期间被固定表示，公式如下。其中 $Sim$ 是余弦相似函数，$S^j\in  \mathbb{R}^{|v^j|}$是第 $j$ 个(域，槽)对在其可能值上的概率分布。$G_j$ 是第 $j$ 个槽的槽投影层，$|G|$ 投影层的数量和(域，槽)对的数量相同，该模型采用将所有对的交叉熵损失加和的方法进行训练。
$$S_i^j=Sim(G_j(F(X)),F(v_i^j))\in \mathbb{R}^1$$
+ **对话行为预测**：对话行为预测是一个多标签分类问题，因为系统的响应可能会包含多种对话行为，例如同时包含请求request和通知inform。模型将对话历史作为输入并且对每种可能的对话行为预测一个二分类的结果，公式如下。其中 $W_2\in \mathbb{R}^{(d_B\times N)}$ 是一个可训练的线性映射，$N$ 是可能的对话行为的数量，在经过一个sigmoid层之后，$A$ 中的每个值都在0到1之间。模型用二分类交叉熵损失来进行训练，当$A_i>0.5$ 时，第 $i$ 个对话行为被认为是触发的对话行为。
$$A=Sigmoid(W_2(F(X)))\in \mathbb{R}^N$$
+ **响应选择**：响应选择是一个排序问题，它的目的是从候选池中检索最相关的系统响应。我们使用双编码器策略并计算源 $X$ 和目标 $Y$ 之间的相似性分数，公式如下。其中，$Y_i$ 是第 $i$ 个候选响应，$r_i$ 是它的余弦相似度分数。源 $X$ 可以被截断，在论文的实验中，将上下文长度限制为最多256个token，并从语料库中随机的抽取几个系统响应当作负例。虽然它可能不是一个真的负例，但是这样去训练和评估它的结果是很常见的。
$$r_i=Sim(F(X),F(Y_i))\in \mathbb{R}^1$$

# 实验结论
对每一个下游任务，论文首先用全部的数据集进行实验，然后模拟few-shot设置去展示TOD-BERT的能力。为了减少样本的方差，实验至少对每一个few-shot实验用三次不同的随机种子去运行，并报告这些有限数据场景的平均值和标准差。论文研究了两个版本的TOD-BERT，一个是TOD-BERT-mlm，在预训练时它只用到了MLM损失；另一个是TOD-BERT-jnt，它用MLM损失和RCL损失进行联合训练。我们将TOD-BERT和BERT以及其余的基线（包括另外两个强大的预训练模型，GPT2和DialoGPT）进行比较。对于基于GPT的模型，论文使用隐藏状态的平均池作为它的输出表示，因为这样比只用最后一个token要更好。
## Linear Probe
在对每个预先训练好的模型进行微调之前，首先通过探测它们的输出表示来研究它们的特征提取能力。提出了探测方法来确定所学习的嵌入在本质上携带了什么信息。在一个固定的预先训练的语言模型上使用一个单层感知器来探测输出表示，并且只针对具有相同超参数的下游任务微调该层，下图为MWOZ上的域分类、OOS上的意图识别和MWOZ上的对话行为预测的探测结果。TOD-BERT-jnt在这个设置中实现了最高的性能，这表明它的表示包含了最有用的信息。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210603235539282.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70)
## 意图识别
在一个最大的意图识别数据集OOS中，TOD-BERT优于BERT和其他强基线，如下图所示。评估所有数据的准确性，分别评估了域内意图和域外意图。有两种方法可以预测域外的意图，一种是将其视为附加类，另一种是设置预测置信度的阈值，这里展示第一次设置的结果。TOD-BERT-jnt达到了最高的域内和域外的准确率。另外，在训练集中每个意图的类别中都随机抽取一个和十个话语，进行1-shot和10-shot实验。与1-shot的BERT相比，TOD-BERT-jnt在所有意图的正确率增加了13.2%，在domain的准确率增加了16.3%。
![在这里插入图片描述](https://img-blog.csdnimg.cn/202106032359167.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70)
## 对话状态追踪
在对话状态跟踪任务中，常用两种评价指标：联合目标准确度和槽准确度。联合目标准确度将预测的对话状态与每个对话回合的真实状态进行比较。真实状态包括所有可能(域,槽)对的槽值。当且仅当所有预测值与其基本真值完全匹配时，输出才被认为是正确的预测。另一方面，槽准确度将每个(域,槽,值)三元组分别与其基本真值标签进行比较。在下图中，比较了MWOZ2.1数据集上的BERT和TOD-BERT-jnt，发现后者的联合目标精度提高了2.4%。实验数据还展示了使用1%、5%、10%和25%数据的少量few-shot实验，注意，1%的数据有大约84个对话。TOD-BERT在所有情境下的表现都优于BERT，这进一步显示了任务型对话预训练中的优势。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210604110947994.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70)
## 对话行为预测
在三个不同的数据集上进行了实验，并展示了对话行为预测任务（一个多标签分类问题）的micro-F1和macro-F1分数。对于MWOZ数据集，从原始系统标签中删除域信息，例如，“出租车通知”将简化为“通知”。这一进程将可能的对话行为从31个减少到13个（对行为进行合并）。对于DSTC2和GSIM语料库，遵循Paul等人的方法，应用通用对话行为映射，将原始对话行为标签映射为通用对话行为格式，分别在DSTC2和GSIM中产生19个和13个系统对话行为。运行另外两个基线，MLP和RNN，以进一步显示基于BERT的模型的优势。MLP模型简单地利用包中的单词进行对话行为预测，RNN模型是一个双向GRU网络。在下图中，我们可以观察到，在全数据场景中，无论是哪种数据集或哪种评估指标，TOD-BERT始终比BERT和其他baseline效果更好。在few-shot实验中，TOD-BERT-mlm在1%数据的情况下，在MWOZ语料库上的micro-F1和macro-F1分别比BERT好3.5%和6.6%。还发现，10%的训练数据可以获得接近全数据训练的良好性能。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210604111726722.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70)
## 响应选择
为了评估任务型对话中的响应选择，使用k-to-100准确度。k-of-100是使用100个示例的随机批计算的，因此来自同一批中的其他示例的响应可以用作随机负例候选。这使我们能够高效地计算跨多个示例的度量。虽然不能保证随机负例确实是“真”负例，但是1/100的度量仍然提供了一个有用的评估信号。在推断过程中，运行五个不同的随机种子来抽样批次并展示平均结果。在下图中，对三个数据集（MWOZ、DSTC2和GSIM）进行了响应选择实验。TOD-BERT-jnt在MWOZ上的1-to-100准确率为65.8%，3-to-100准确率为87.0%，分别超过BERT18.3%和11.5%。在DSTC2和GSIM数据集中也一致地观察到类似的结果，并且TOD-BERT-jnt的优势在few-shot场景中更为明显。我们不报告TOD-BERT-jnt在MWOZ数据集上的few-shot设置，是因为在预训练的响应对比学习阶段使用了完整的MWOZ数据集，这和其它的模型去比较是不公平的。响应选择结果对训练批量很敏感，因为批量越大，预测越困难。在实验中，我们将所有模型的批大小设置为25。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210604111914617.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70)
# 总结
+ 使用和任务相同的domain数据集进行预训练
+ 使用更有意义的损失函数

