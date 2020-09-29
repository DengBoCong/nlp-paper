> 标题：An End-to-End Trainable Neural Network Model with Belief Tracking for Task-Oriented Dialog\
> 原文链接：[Link](https://arxiv.org/pdf/1708.05956.pdf)\
> Github：[NLP相关Paper笔记和实现](https://github.com/DengBoCong/nlp-paper)\
> 说明：阅读论文时进行相关思想、结构、优缺点，内容进行提炼和记录，论文和相关引用会标明出处，引用之处如有侵权，烦请告知删除。\
> 转载请注明：DengBoCong

# Abstract
我们提出了面向任务的对话系统的新型端到端可训练神经网络模型，该模型能够跟踪对话状态，基于知识（KB）的API调用，并将结构化的KB查询结果合并到系统响应中，从而成功完成面向任务的对话。通过在对话历史上的进行belief tracking和KB结果处理，进而模型产生结构良好的系统响应。我们使用从第二个Dialog State Tracking Challenge（DSTC2）语料库转换而来的数据集在饭店搜索域中评估模型。实验结果表明，在给定对话历史记录的情况下，该模型可以很好地跟踪对话状态。此外，我们的模型在产生适当的系统响应方面表现出令人鼓舞的结果，其使用基于每个响应的准确性评估指标优于以前的端到端可训练神经网络模型。
# Introduction
端到端可训练神经网络模型可以直接针对最终系统目标函数（例如任务成功率）进行优化，从而缓解了可信分配和在线适应的挑战。在这项工作中，我们提出了面向任务的对话的端到端可训练神经网络模型，该模型将统一网络应用于belief tracking，基于知识（KB）操作和响应创建。该模型能够跟踪对话状态，与KB交互以及将结构化KB查询结果合并到系统响应中，从而成功完成面向任务的对话框。我们表明，在给出对话历史记录的情况下，我们提出的模型可以有效地跟踪状态。与先前的端到端可训练神经网络模型相比，我们的模型还证明了在提供适当的系统响应和进行面向任务的对话方面的有更好的性能。
# Related Work
+ Dialog State Tracking
   + 在口语对话系统中，对话状态跟踪或belief tracking是指在可能的对话状态上保持分布的任务，这些状态直接确定系统的动作。
   + 使用诸如CRF或RNN之类的序列模型进行判别的方法可以灵活地探索任意特征并实现最新的DST性能，从而解决了生成模型的局限性。
+  End-to-End Task-Oriented Dialog Models
   + 我们的模型使用统一的网络进行信念跟踪，知识操作和响应生成，以充分探索可以在不同任务之间共享的知识

# Proposed Method
我们将面向任务的对话建模为一个多任务序列学习问题，其组件用于编码用户输入，跟踪信念状态，发出API调用，处理KB结果以及生成系统响应。模型架构如下图所示，对话框中的多轮序列使用LSTM递归神经网络进行编码， 根据对话历史记录，会话状态保持在LSTM状态。LSTM状态向量用于生成：（1）通过从非词法化系统响应候选列表中选择句结构；（2）信念跟踪器中每个插槽值的概率分布；（3）指向检索到的KB结果中与用户查询匹配的实体。通过用预测的插槽值和实体属性值替换去词化的token来生成最终的系统响应。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200929090815736.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
+ Utterance Encoding：这里的话语编码是指将单词序列编码为连续的密集向量。我们使用双向LSTM将用户输入编码为句向量，其中用户输入第 $k$ 轮对话共 $T_k$ 个 单词表示为$U_k=(w_1,w_2,...,w_{T_k})$。用户的句向量 $U_k$ 表示为 $U_k=[\overrightarrow{h_{T_k}^{U_k}},\overleftarrow{h_{1}^{U_k}}]$，$\overrightarrow{h_{T_k}^{U_k}}$和 $\overleftarrow{h_{1}^{U_k}}$ 是第 $k$ 轮最后的前向和反向的句级LSTM状态。
+  Belief Tracking：信念跟踪（或对话状态跟踪）通过沿对话顺序积累信息来维持和调整对话状态（例如用户的目标）。在第 $k$ 轮从用户输入中收集新信息后，神经对话模型会更新每种插槽类型 $m∈M$ 的候选值的概率分布 $P(S_{k}^{m})$。在回合 $k$，对话级LSTM（LSTMD）更新其隐藏状态 $s_k$，并在接收到用户输入编码 $U_k$ 和 K_B$ 指示器 $I_k$（将在下面的部分中进行说明）之后，使用它来推断用户目标的任何更新。
$$s_k=LSTM_D(s_{k-1}, [U_k, I_k])$$$$P(S_{k}^{m}|U_{\leq k}, I_{\leq k}) = SlotDist_m(s_k)$$
其中，$SlotDist_m(s_k)$是在插槽类型 $m∈M$ 上具有softmax激活功能的多层感知器（MLP）
+ Issuing API Calls：基于对话状态，模型可以基于信念跟踪输出发出API调用以查询KB。该模型首先生成一个简单的API调用命令模板。 通过使用信念追踪器针对每个目标插槽的最佳假设替换命令模板中的插槽类型token，产生最终的API调用命令。
+ KB Results Processing：一旦神经对话模型接收到KB查询结果，它将通过从返回的列表中选择实体来向用户建议选项。KB搜索或数据库查询的输出通常具有定义明确的结构，并且实体属性与实体索引相关联。在对话的第 $k$ 轮，将二进制KB指示器 $I_k$ 传递给神经对话模型。该指标由上一次API调用中检索到的实体数和当前实体指针决定。当系统处于向用户建议实体的状态时，如果接收到零值 $I_k$，则该模型很可能会通知用户与当前查询匹配的实体不可用，否则，如果 $I_k$ 有值，该模型可能会根据实体指针 $P(E_k)$ 的更新概率分布从检索结果中选择一个实体：
$$P(E_k|U_{\leq k}, I_{\leq k}) = EntityPointerDist(s_k)$$
其中，其中 $EntityPointerDist$ 是具有softmax激活的MLP。
+ System Response Generation：在对话的第 $k$ 轮，从非词化响应候选列表中选择句结构 $R_k$。最终的系统响应是通过用预测的插槽值和实体属性值替换非词性化token来产生的。
+ Model Training：我们通过找到参数集 $θ$ 来训练神经对话模型，该参数集 $θ$ 最小化了目标标签，实体指针和去词化系统响应的预测分布和真实分布的交叉熵：
$$\underset{\Theta}{min}\sum_{k=1}^{K}-[\sum_{m=1}^{M}\lambda_{S^m}logP(S_{k}^{m*}|U_{\leq k},I_{\leq k};\Theta )$$$$+\lambda_ElogP(E_{k}^{*}|U_{\leq k},I_{\leq k};\Theta )$$$$+\lambda_RlogP(R_{k}^{*}|U_{\leq k}, I_{\leq k};\Theta )]$$
其中，其中 $λ_s$ 是每个系统输出成本的线性插值权重。$S_{k}^{m∗}$，$E_K^*$和$R_k^*$ 是第 $k$ 轮每个任务的真实标签。
+  Alternative Model Designs：直观地，如果模型被明确告知目标值估计并且知道其先前对用户做出的响应，则该模型可能会提供更好的响应。因此，我们设计并评估了一些替代模型架构，以验证这种假设：
   + 具有先前发出的非词化系统响应的模型连接回对话级LSTM状态：$s_k=LSTM_D(s_{k-1},[U_k,I_k,R_{k-1}])$
   + 具有先前发出的插槽标签的模型连接回对话级LSTM状态：$s_k=LSTM_D(s_{k-1},[U_k,I_k,S_{k-1}^{1},...,R_{k-1}^{M}])$
   + 具有先前发出的响应和插槽标签的模型都已连接回对话框LSTM状态：$s_k=LSTM_D(s_{k-1},[U_k,I_k,R_{k-1},S_{k-1}^{1},...,R_{k-1}^{M}])$

# Experiments
我们使用DSTC2中的数据进行模型评估。 在这项研究中，我们通过保留对话状态注释并添加系统命令（API调用）来结合原始的DSTC2语料库和此转换版本。下表汇总了该数据集的统计信息。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200929093437880.png#pic_center)
我们使用Adam优化方法进行批量为32的小批量模型训练，在模型训练期间，具有dropout的正则化应用于非循环连接[26]，dropout率为0.5。我们将梯度的最大范数设置为5，以防止梯度爆炸。对话层LSTM和话语层LSTM的隐藏层大小分别设置为200和150，大小为300的单词嵌入是随机初始化的，我们还尝试使用在Google新闻数据集上受过训练的预训练词向量来初始化词嵌入。

下表显示了使用不同用户话语编码方法和不同单词嵌入初始化的模型的评估结果：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200929104233676.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
下表显示了不同的递归模型架构的评估结果：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200929104439793.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
下表中的结果所示，我们的系统可实现与最新系统相当的信念跟踪性能：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200929104545966.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
即使使用相同的评估度量，我们的模型在设计上的设置也与下表中的其他已发布模型相比略有不同
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200929104800976.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
# Conclusions
在这项工作中，我们为面向任务的对话系统提出了一种新颖的端到端可训练神经网络模型。该模型能够跟踪对话的状态，通过发出API调用与知识交互，并将结构化的查询结果合并到系统响应中，从而成功完成面向任务的对话框。在餐厅搜索域的评估中，使用来自第二个“Dialog State Tracking Challenge”语料库的转换数据集，我们提出的模型显示了在对话轮次序列上跟踪对话状态的鲁棒性能。该模型还展示了在生成适当的系统响应方面的有不错的性能，优于以前的端到端可训练神经网络模型。