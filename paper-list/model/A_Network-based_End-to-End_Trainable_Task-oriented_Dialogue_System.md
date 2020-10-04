
> 标题：A Network-based End-to-End Trainable Task-oriented Dialogue System\
> 原文链接：[Link](https://arxiv.org/pdf/1604.04562.pdf)\
> Github：[NLP相关Paper笔记和实现](https://github.com/DengBoCong/nlp-paper)\
> 说明：阅读论文时进行相关思想、结构、优缺点，内容进行提炼和记录，论文和相关引用会标明出处，引用之处如有侵权，烦请告知删除。\
> 转载请注明：DengBoCong

论文作者将对话建模成一个seq2seq的映射问题，该seq2seq框架以对话历史数据（通过belief tracker建模）和数据库查询结果（通过Database Operator得到结果）作为支撑。

# Abstract
教会机器完成与人自然交流的任务是充满挑战性的，当前，开发面向任务的对话系统需要创建多个组件，通常这涉及大量的手工制作或获取昂贵的标记数据集以解决每个组件的统计学习问题。在这项工作中，我们介绍了基于神经网络的文本输入，文本输出的端到端可训练的面向目标的对话系统，以及一种基于pipeline的Wizard-of-Oz框架的收集对话数据的新方法。这种方法使我们能够轻松开发对话系统，而无需对手头的任务做太多假设。结果表明，该模型可以自然地与人类交谈，同时帮助他们完成餐馆搜索领域的任务。
# Introduction
建立面向任务的对话系统（例如酒店预订或技术支持服务）很困难，因为它是针对特定应用的，并且训练数据的可用性通常有限。为了缓解这个问题，最近的面向任务的对话系统设计的机器学习方法已经将该问题作为部分可观察到的马尔可夫决策过程（POMDP）进行了研究，目的是使用强化学习（RL）进行训练，通过与真实用户的互动在线对话策略。然而，语言理解和语言生成模块仍然依赖于监督学习，因此需要语料库来训练。此外，为了使RL易于处理，必须仔细设计状态和动作空间，这可能会限制模型的表达能力和可学习性，而且训练此类模型所需的奖励方法难以设计且难以在运行时进行衡量。

另一方面，从序列到序列学习激发了一些努力来构建端到端可训练的，非任务导向的对话系统。该系列方法将对话视为目标序列转导问题的来源，应用编码器网络将用户查询编码为代表其语义的分布矢量，然后使用解码器网络以生成每个系统响应，这些模型通常需要大量数据来训练。它们允许创建有效的聊天机器人类型的系统，但是它们缺乏支持领域特定任务的任何功能，例如，能够与数据库进行交互，并将有用的信息汇总到他们的系统中回应。

在这项工作中，我们通过平衡两个研究方向的优势和劣势，为面向任务的对话系统提出了一种基于神经网络的模型。
+ 该模型是端到端可训练的，但仍模块化连接
+ 它并不能直接为用户目标建模，但是尽管如此，它仍然可以通过在每一步提供**relevant**且**appropriate**的响应来学习完成所需的任务
+ 它具有数据库（DB）属性（槽-值对）的显式表示形式，用于实现较高的任务成功率，但具有用户意图的分布表示（对话行为）允许模棱两可的输入
+ 并且它使用了词法分解和权重绑定策略来减少训练模型所需的数据，但是如果有更多数据可用，它仍然保持较高的自由度。

# Model
模型结构图如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020100322492021.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
在每个回合中，系统都会从用户那里获得token序列作为输入，并将其转换为两个内部表示形式：
+ 由 intent network生成的分布表示
+ 由一组belief trackers生成的称为belief state的槽值对上的概率分布

然后数据库operator挑选belief state中最可能的值以形成对数据库的查询，策略网络将搜索结果、意图表示和信念状态进行转换和组合，以形成代表下一个系统动作的单个向量。然后，该系统动作向量用于调节响应生成网络，该网络以骨架(skeletal)形式中的token生成所需的系统token输出。然后，通过将数据库实体的实际值代入骨架句结构来形成最终的系统响应。

>具体而言，在每一轮对话中，通过Intent Network得到一个用户输入的向量表征，通过Belief Tracker得到一个slot-value的概率分布，随后database operator针对概率最大的slot-value在数据库中进行查询，得到的结果将会和Intent Network输出的向量表征以及Belief Tracker输出的slot-value分布概率共同输入给policy network，随后获得一个向量，该向量表征了该系统的下一个action，然后该action被输入到Generation Network中产生回复。

每个组件的详细说明如下。

### Intent Network
Intent Network可以看作是序列到序列学习框架中的编码器，其工作是在 $t$回合， 将输入tokens为 $w_0^t，w_1^t,...,w_N^t$ 的序列编码为分布向量表示 $z_t$。通常，使用长短期记忆（LSTM）网络，其中最后一个时间步中隐藏层 $z_t^N$ 被表示为：
$$z_t=z_t^N=LSTM(w_0^1,w_1^t,...w_N^t)$$
或者，可以使用卷积神经网络（CNN）代替LSTM作为编码器
$$z_t=CNN(w_0^1,w_1^t,...w_N^t)$$
本文中都进行探讨。
### Belief Trackers
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201004101353442.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
Belief Tracker也叫做Dialogue State Tracker，它的详细结构如上图所示，包含两个主要结构：
+ Delexicalised CNN
+ Jordan-type RNN

在Delexicalised CNN中（**delexicalised主要指句子中同一类型的实体词都被替换成该类型的统一符号，以在slot类型中共享参数**），在当前对话轮次 $t$ 的用户输入 $u$，以及上一轮次系统的回复 $m$，分别通过该CNN结构后的输出进行concatenation，只不过需要注意的是，在各自的CNN结构中，除了使用CNN的最终输出外，也会利用各自输入句子中被delexicalised处的浅层CNN特征（为了保证各层卷积的输出能够与输入句子长度对应一致，在每一层卷积输入的首尾两端进行padding），如果当前句子没有被delexicalised的词则进行padding。
$$f_{v,cnn}^t=CNN_{s,v}^{(u)}(u_t)⊕CNN_{s,v}^{(m)}(m_{t-1})$$
在Jordan-type RNN中可以看到，和普通RNN结构不同，Jordan-type RNN更为简单，没有输入层，直接将上一个时刻的输出，以及来自于Delexicalised CNN的输出进行concatenation后当做RNN的隐藏层状态，并通过softmax后得到当前时刻的输出，具体计算过程公式如下所示：
$$f_v^t=f_{v,cnn}^t⊕p_v^{t-1}⊕p_∅^{t-1}$$ $$g_v^t=w_s \cdot sigmoid(W_sf_v^t+b_s)+b_s^{'}$$ $$p_v^t=\frac{exp(g_v^t)}{exp(g_{∅,s})+\sum_{v'∈V_s}exp(g_{v'}^t)}$$

其中concat到Jordan-type RNN隐藏层的除了CNN的输出外，还有两个概率，一个是上一轮的该槽位取值某个 $v$ 的概率大小，另一个是直到当前轮次 $t$ 时，用户还未提及该槽位，也可以用一个概率大小来表征，直接在第三个公式中利用分母中多余的一个参数代替普通的 $g(v)$ 计算即可（这样的话，该槽对应所有可能取值的概率之和，以及用户未提及该槽的概率，才能够使得所有概率之和为1）。

特别需要注意的是，论文中采用的方法是，先针对每个具体的task构建一个本体库$G$，它是一个知识图谱，在这个知识图谱中，定义了该task下存在的各种可能槽位以及槽位对应的可能取值。而槽位分为两种类型：informable slot和requestable slot，前者是用户用来限定查询范围的一些信息（比如订餐task中的食物类型，价格区间等等），后者是用户想要咨询的信息（比如询问地址和电话等，那么地址和电话此时便是一个requestable slot）。此后针对该本体知识图谱$G$中的每一个槽位$s$，有两种处理办法：
+ 对于informable slot，每一个槽位$s$都有一个专门的Jordan type RNN进行跟踪。例如针对食物类型有一个专门的RNN进行跟踪，在跟踪过程中，每一个轮次$t$都会计算一次RNN在当前时刻$t$的输出，用以更新食物类型这个槽位上所有可能取值的概率分布
+ 对于requestable slot，因为不需要跟踪，并未使用RNN结构，然而原文未做详细解读，个人猜测就是每个时刻做一个简单的二分类，输出一个binary distribution，大概意思就是用户当前是否向系统请求了该槽位的信息

### Database Operator
通过Belief Tracker后，可以针对所有informable slot的所有可能取值，通过下式形成一个查询语句 $q$（不过个人在这里有些疑问，按照下式的意思，大概是针对每一个槽位都取一个概率最大的值，并将所有informable slot全部合并形成一个 $query$，这样的话，岂不就会在每一轮的查询语句中，都包含了所有的informable slot，只不过每一轮的查询语句 $q$ 中各个槽位的具体取值不一样而已。如果是这样个人感觉不太合理，如果不是这样那是否公式的argmax应该放到取合集符号的外边来呢？），使用查询语句在数据库中进行查询后，会得到一个针对数据库中实体的一个 $01$ 向量（类似于bag-of-words中，该向量的每一位表示数据库中的某个实体，如果命中了数据库中的某个实体，则该位置1）。
$$\bigcup_{s'\in S_1 }{\underset{v}{argmax}p_{s'}^t}$$
此外，如果查询结果只要有命中（即向量x不全为0），则这里还会维护一个DB pointer，它将随机指向一个命中的实体。并且根据每轮查询的结果进行动态更新。这个pointer主要是在后面生成网络中用来显示一个有意义的句子（因为生成的句子是类似于模板一样的，并没有显示具体的实体信息）。

### Policy network
在该模块中，实际上和强化学习中的policy action还有点不一样，这里的policy实际上就是一个融合另外三个模块，并输出的一个向量。公式如下：
$$o_t=tanh(W_{zo}z_t+W_{po}\hat{p}_t+W_{xo}\hat{x}_t)$$
+ $z$ 便是intent network输出的向量。
+ 其中对于belief tracker模块的处理如下（也就是将各个informable slot的概率进行concatenation）$\hat{p}_t=\oplus_{s\in G}\hat{p}_s^t$，而针对每一个具体的slot，它也是一个向量，其由三部分组成：（该slot下各个可能取值的概率之和，用户表示对当前槽位无所谓的概率，用户到当前轮次并未提及该槽位的概率）。这里为什么要针对每一个slot下的各个可能取值的概率大小进行求和，就是因为在generation network中，对于槽位信息很重要，但是对于每个槽位下的可能取值则不重要（因为无论是输入的句子还是生成的句子都是delexicalised）
+ 对于database operator的输出而言，同样的，对于查询语句得到的结果能够查询到的实体个数很重要，但是具体查询到的是什么实体并不重要。因此最后 $x$ 便转化为了一个6位的one-hot向量，每一位分别表示没有查询到任意实体，查询结果命中一个实体，查询结果命中两个实体，…，查询结果命中等于或超过五个实体。

### Generation Network
生成网络就是一个普通的decoder，只不过在输入中加入Policy Network输出的action vector，也就是向量 $o$，公式如下：
$$P(w_{j+1}^t|w_j^t,h_{j-1}^t,o_t)=LSTM_j(w_j^t,h_{j-1}^t,o_t)$$
在每一个时刻输出一个token，该token有三种可能：
+ 一个正常的词；
+ 一个delexicalised slot name，在最终输出的时候会将其替换为实际的一个有意义的词，比如<s.food>会替换为”food”或”type of food”；
+ 一个delexicalised slot value，在最终输出的时候会将其替换为在Database Operator中DB pointer维护的一个实体。

#  Wizard-of-Oz Data Collection
Wizard-of-Oz数据集搜集范式这个就不做描述和介绍了，应该已经很熟悉了。

#  Empirical Experiment
+ 使用交叉熵预先训练belief tracker的参数
$$L_1(\Theta_b)=-{\sum}_t{\sum}_s(y_s^t)^Tlogp_s^t$$
其中，y代表真实label。对于完整的模型，我们有三个informable追踪器（食品，价格范围，面积）和七个requestable追踪器（地址，电话，邮编，姓名，以及三个slot）。
+ 在固定了tracker的参数之后，使用来自生成网络语言模型的交叉熵损失函数对模型的其余部分进行训练
$$L_1(\Theta_{/b})=-{\sum}_t{\sum}_j(y_j^t)^Tlogp_j^t$$
其中$y_j^t$和$p_j^t$分别是在第 $t$ 轮decoder第 $j$ 步的时候输出的target token和预测 token。我们将每个对话视为一个批次，并使用随机梯度下降和小的l2正则化项来训练模型。


收集的语料库按3：1：1的比例划分为训练，验证和测试集。early stopping是基于正则化的验证集来实现的，并且梯度裁剪被设置为1.所有隐藏层大小被设置为50，并且所有权重在-0.3和0.3之间被随机初始化，包括字嵌入。输入和输出的词汇大小大约为500，其中可以灵活化的去掉罕见单词和可以被delexicalisation的单词。我们对实验中的所有CNN使用了三个卷积层，并且所有滤波器大小都设置为3.池化操作仅在最后的卷积层之后应用。
> 梯度裁剪是一种在非常深度的网络（通常是循环神经网络）中用于防止梯度爆炸（exploding gradient）的技术。执行梯度裁剪的方法有很多，但常见的一种是当参数矢量的 L2 范数（L2 norm）超过一个特定阈值时对参数矢量的梯度进行标准化，这个特定阈值根据函数：新梯度=梯度* 阈值/L2范数（梯度）
{new_gradients = gradients * threshold / l2_norm(gradients)}确定。


在该评估中，模型使用了三个评估指标：

+ BLEU评分（on top-1 and top-5 candidates）：我们使用实体值替换进行lexicalising之前，在模板输出句子上计算BLEU分数。
+ 实体匹配率：通过确定每个对话结束时实际选择的实体是否与用户指定的任务相匹配来计算实体匹配率。我们通过确定每个对话结束时实际选择的实体是否与指定给用户的任务相匹配来计算实体匹配率。 如果（1）所提供的实体匹配，并且（2）系统回答来自用户的所有相关信息请求（例如，地址是什么），则对话被标记为成功。
+ 客观任务成功率。

**下表是对tracker的评估结果**
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020100412021267.png#pic_center)
**下表是基于语料的评估结果**
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020100412030771.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
我们使用t-SNE生成一个降维视图，该视图嵌入了前三个生成的输出词（完整模型，不注意）嵌入，绘制和标记，该图如下图所示。
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020100416220381.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
下表是认为评估的结果
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201004162227849.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
我们还对NN模型和由handcrafted 语义分析器，基于规则的策略和信念跟踪器以及基于模板的生成器组成的人工模块化基准系统（HDC）进行比较。 结果如下表：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201004162258400.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
# Conclusions and Future Work
目前的模型是一个基于文本的对话系统，它不能直接处理噪声语音识别输入，也不能在用户不确定时要求用户确认。 事实上，这种类型的模型在多大程度上可以扩展到更大更广的领域，这仍然是希望在今后的工作中追求的一个悬而未决的问题。