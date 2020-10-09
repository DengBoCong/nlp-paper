
> 标题：Latent Intention Dialogue Models\
> 原文链接：[Link](https://arxiv.org/pdf/1705.10229.pdf)\
> Github：[NLP相关Paper笔记和实现](DengBoCong/nlp-paper)\
> 说明：阅读论文时进行相关思想、结构、优缺点，内容进行提炼和记录，论文和相关引用会标明出处，引用之处如有侵权，烦请告知删除。\
> 转载请注明：DengBoCong

# Abstract
开发能够做出自主决策并通过自然语言进行交流的对话代理是机器学习研究的长期目标之一。传统方法要么依靠手工制作一个小的状态动作集来应用不可扩展的强化学习，要么构建确定性模型来学习无法捕获自然对话可变性的对话语句。论文提出了一种隐意图对话模型（Latent Intention Dialogue Model, LIDM），通过离散的隐变量来学习对话意图，这些隐变量可以看作引导对话生成的动作决策，进而运用强化学习可以提升性能。实际上在任务型对话中，这个隐含的意图可以理解为是action。

# Introduction
本文的贡献有两个方面：首先，我们表明神经变分推理框架能够从数据中发现离散的，可解释的意图，从而形成对话主体的决策基础。其次，代理能够基于相同框架内的外部奖励来修改其对话策略，这很重要，因为它为构建自主对话代理程序提供了垫脚石，该对话代理程序可以通过与用户交互来不断提高自身水平。实验结果证明了我们的潜在意图模型的有效性，该模型在基于语料库的自动评估和人工评估方面均达到了最新水平。

# Latent Intention Dialogue Model for Goal-oriented Dialogue
> Knowledge graph（KG）和Knowledge base（KB）几乎可以看做同义词，只不过Knowledge base是知识库，而Knowledge graph则是基于知识库的图结构。

LIDM基于(Wen et al,2017，这篇文章也有[笔记](论文阅读笔记：A Network-based End-to-End Trainable Task-oriented Dialogue System))中描述的端到端系统架构。目标导向型对话通过人机对话交互，帮助用户完成特定的任务。给定一个用户输入 $u_t$ 和知识库(KB)，模型需要将输入解析为可执行的命令Q在知识库搜索。基于返回的搜索结果，模型需要以自然语言的形式给出回复 ，整体模型结构如下
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020100809581517.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)

将分为三个模块：Representation Construction，Policy Network和Generator。
## Representation Construction
表征构建部分主要是为了捕捉用户的意图和获取知识库查询结果。$S_t$ 是对话状态向量，$u_t$ 是用户输入经过BiLSTM的最后隐层状态；belief vector 是特定槽-值队的概率分布拼接，通过预训练的RNN-CNN belief tracker抽取。$m_{t-1}$ 是上一轮的机器回复，$b_{t-1}$ 是上一轮的belief vector。基于belief vector，查询Q与知识库交互，并返回表示匹配程度的one-hot vector $x_t$。
+ 将utterance通过BiLSTM取两个方向最后一个时刻的隐层状态concat后得到表征 $u$
$$u_t=biLSTM_\Theta(u_t)$$
+ 分别对utterance句子 $u_t$ 和系统上一时刻的回复 $m_{t-1}$ 先用CNN，尔后通过一个RNN做轮与轮之间的belief tracker 
$$b_t=RNN-CNN(u_t,m_{t-1},b_{t-1})$$
+ 通过当前轮得到的 $b$ 生成一个查询语句Q(**基于belief向量，通过取每个槽位的最大值的并集来形成查询Q**)，查询KB后能够得到一个结果表征 $x$（**表示KB中的匹配程度**）
+ 将$u,b,x$进行concatenation后得到state表征$s$
$$s_t=u_t⊕b_t⊕x_t$$

## Policy Network and Generator
基于之前构建的对话状态，策略网络通过一个MLP输出latent intention（实际上就是一个action） $z_t$。
$$\pi_\Theta (z_t|s_t)=softmax(W_2^T\cdot tanh(W_1^Ts_t+b_1)+b_2)$$  $$z_t^{(n)}\sim \pi_\Theta(z_t|s_t)$$
和以往的隐变量是一个多维值连续的向量不同，这里的 $z_t$ 是离散的多个类别，类别数是人工设定的，$\pi_t$ 输出的是 $z_t$ 的softmax概率。从强化学习的角度来看，一个隐意图 $z_t^{(n)}$ 可以看作是一个动作。从策略网络采样得到一个隐意图 $z_t^{(n)}$ 加上之前的状态向量$S_t$，计算出控制向量 $d_t$，作为条件控制LSTM 语言模型生成回复。
$$d_t=W_4^Tz_t⊕[sigmoid(W_3^Tz_t+b_3)\cdot W_5^Ts_t]$$  $$p_\Theta(m_t|s_t,z_t)=\coprod_{j}p(w_{j+1}^t|w_j^t,h_{j-1}^t,d_t)$$
上面两式子中的 $z_t$ 是 $z_t^{(n)}$的one-hot形式，$w_j^t$是生成的第 $t$ 轮对话的最后一个词，$h_{j-1}^t$ 论文说是解码器的最后一个隐层状态（下标是 $j-1$有点奇怪）。从而，LIDM模型的公式化表示就是：
$$p_\Theta(m_t|s_t)=\sum_{z_t}p_\Theta(m_z|z_t,s_t)\pi_\Theta(z_t|s_t)$$
## Inference
LIDM模型的公式化式很难直接求解，因为有隐变量的存在，套路是将其转化为优化下界。变分推断部分，LIDM通过识别网络 $q_{\phi}(z_t|s_t,m_t)$ 来逼近隐变量的后验分布 $p(z_t|s_t,m_t)$, 训练时优化变分下界。
$$L =\mathbb{E}_{q\phi (z_t)}[logp_\Theta (m_t|z_t,s_t)]-\lambda D_{KL}(q_\phi (z_t)||\pi_\Theta (z_t|s_t))$$  $$\leq log\sum_{z_t}p_\Theta (m_t|z_t,s_t)\pi_\Theta (z_t|s_t)$$  $$=logp_\Theta (m_t|s_t)$$
其中识别网络 $q_{\phi}(z_t|s_t,m_t)$
$$q_{\phi}(z_t|s_t,m_t)=Multi(o_t)=softmax(W_6o_t)$$   $$o_t=MLP_\phi(b_t,x_t,u_t,m_t)$$   $$u_t=biLSTM_\phi(u_t),m_t=biLSTM_\phi(m_t)$$
而值得注意的是，目标函数中包含三个不同的模型，因此各自的参数也各不相同，需要分开进行优化
+ 对于生成回复模型（即p(m|z,s)部分）的参数优化
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201008121203569.png#pic_center)
+ 对于policy network的参数优化
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201008121241452.png#pic_center)
+ 对于inference network的参数优化（这里我觉得文中定义的r(m,z,s)似乎少了一项）
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201008121319185.png#pic_center)
## Optimization
变分推断的隐变量 $z$ 是从一个分布采样得到，存在high variance的缺点，并且容易陷入LSTM语言模型的disconnection 现象中。论文通过半监督学习提升训练的稳定性和阻止disconnction。针对这些问题，论文提出用Semi-Supervision和Reinforcement learning缓解这一问题。
### Semi-Supervision
论文将隐意图的推断看作是无监督的聚类任务，因此运用标准的聚类算法处理一部分语料，并生成标签 $\hat{z}_t$，并把它当作一个观测变量。训练的数据分成两部分：未标注的样本集 $(m_t,s_t)\in\mathbb{U}$ 和聚类生成标注样本集 $(m_t,s_t,\hat{z})\in\mathbb{L}$ ，分别用不同的目标函数：
+ 将未能够聚类的样例U，继续利用上面的方法进行训练；
$$L_1 =\sum_{(m_t,s_t)\in U}\mathbb{E}_{q\phi (z_t|s_t,m_t)}[logp_\Theta (m_t|z_t,s_t)]-\lambda D_{KL}(q_\phi (z_t|s_t,m_t)||\pi_\Theta (z_t|s_t))$$
+ 将能够聚类得到了标签的样例则利用下式进行优化
$$L_2 =\sum_{(m_t,\hat{z}_t,s_t)\in L}log[p_\Theta(m_t|\hat{z}_t,s_t)\pi_\Theta(\hat{z}_t|s_t)q_\phi (\hat{z}_t|s_t,m_t)]$$
+ 二者形成一个joint objective，$\alpha$是监督样本和无监督样本之间的权衡因子。
$$L=\alpha L_1+L_2$$

### Reinforcement Learning
策略网络 $\pi_\Theta(z_t|s_t)$ 是从潜在的数据分布学习到的，但不一定是最优的决策。论文提出用强化学习对策略网络进行微调参数，对未标注的数据集 $\mathbb{U}$ 训练过程如下： 从策略网络采样得到动作 $z_t^{(n)}$，得到反馈 $r_t^{(n)}$，进行梯度更新：
$$\frac{\partial \jmath}{\partial \theta }\approx \frac{1}{N}\sum_{n}r_t^{(n)}\frac{\partial log\pi_\theta(z_t^{(n)}|s_t)}{\partial \theta'}$$
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201008155355324.png#pic_center)

# Experiments
实验用的数据集是**CamRest676 corpus**，是一个预订餐馆的对话语料。用户可以用三个信息槽（food, pricerange, area）来限制搜索的范围，一旦完成预订，系统可以返回另外三个信息（address, phone, postcode）。语料共有676个对话，将近2750轮。对话隐意图的数量I设置为50，70和100。实验结果用task success rate和BLEU作为指标。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201008155700274.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
+ Ground Truth: human-authored response
+ NDM：the vanilla neural dialogue model
+ NDM+Att: NDM plus an attention mechanism on the belief tracker
+ NDM+Att+SS: the attentive NDM with self-supervised sub-task neuron

可以看到，LIDM模型在BLEU指标上有良好的结果，但在任务成功率上表现不佳。将LIDM训练的策略网络作为initial policy，经过Reinforcement Learning微调参数后，任务成功率得到大幅度的提升，并且BLEU指标没有下降太多。

论文还在Succss、Compression、Naturalness三个指标上进行人工打分，得到的结果如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/202010081559391.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
Comprehension和Naturalness的最高分是5分。可以看到，其实三个系统的任务成功率其实相差不是很大，不过在Comprehension和Naturalness上LIDM均超过了NDM。

从下面的示例对话可以感受下隐意图变量控制生成的回复。不同的隐意图会生成不同的回复，但也可以看到一些隐意图（如intention 0）可以生成风格非常不同的回复，作者认为这是变分推断的variance性质造成的。

另外一个有趣的现象是，经过RL训练后的LIDM，会采取更“贪婪”的追求任务成功的行为，具体表现在Table 5，一旦在数据库搜索到相关结果，会抢先用户提问之前将该餐馆的所有信息都返回给用户，这也造成对话的轮数更简短。这也说明增强学习训练效果的一个直观体现。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201008160057779.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201008160114977.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
# Conclusion
在本文中，我们提出了一个通过离散潜在变量模型学习对话意图的框架，并介绍了用于目标导向对话模型的潜在意图对话模型（LIDM）。我们已经表明，LIDM可以从基础数据分布中发现有效的初始策略，并且能够使用强化学习基于外部奖励来修改其策略。我们认为，这是构建自主对话代理的有希望的一步，因为学习到的离散潜在变量接口使代理能够使用几种不同的范例进行学习。实验表明，所提出的LIDM能够与人类受试者进行交流，并且优于以前发表的结果。