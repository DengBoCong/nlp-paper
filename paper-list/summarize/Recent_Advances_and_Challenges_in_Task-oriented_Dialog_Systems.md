> 标题：Recent Advances and Challenges in Task-oriented Dialog Systems\
> 原文链接：[Link](https://arxiv.org/pdf/2003.07490.pdf)\
> Github：[NLP相关Paper笔记和实现](https://github.com/DengBoCong/nlp-paper)\
> 说明：阅读论文时进行相关思想、结构、优缺点，内容进行提炼和记录，论文和相关引用会标明出处，引用之处如有侵权，烦请告知删除。\
> 转载请注明：DengBoCong

# Abstract
由于在人机交互和自然语言处理中的重要性和价值，面向任务的对话系统在学术界和工业界都受到越来越多的关注。在本文中，我们调查了面向任务的对话系统的最新进展和挑战。我们还讨论了面向任务的对话系统的三个关键主题：（1）提高数据效率以促进在资源匮乏的环境中进行对话建模；（2）为对话策略学习建模多回合模型以实现更好的任务完成性能；（3）将领域本体知识整合到对话模型中。此外，我们回顾了对话评估和一些常用语料库的最新进展。我们认为，尽管这项调查不完整，但可以为面向任务的对话系统的未来研究提供启发。
# Introduction
通常，面向任务的对话系统是建立在结构化本体之上的，该本体定义了任务的领域知识。有关面向任务的对话系统的现有研究可以大致分为两类：pipeline和end-to-end。建立pipeline系统通常需要大规模的标记对话数据来训练每个组件，模块化的结构使系统比端到端的系统更具解释性和稳定性，因此，大多数现实世界的商业系统都是以这种方式构建的。而端到端的结构像是黑匣子，这更加不可控。如下图所示，对于pipeline和end-to-end方法中的每个单独组件，我们列出了一些关键问题，在这些问题中提出了典型的作品。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200928160614477.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
在pipeline方法中，最近的研究更多地集中在对话框状态跟踪和对话框策略组件上，这也称为“对话框管理”。基于域本体，通过预测每个槽的值，DST任务可以视为分类任务（受限制与训练数据，OOV问题），对话策略学习任务通常被认为是强化学习任务。然而，与其他众所周知的RL任务不同，对话策略的训练需要真实的人作为环境，这是非常昂贵的。面向任务的对话系统中的三个关键问题：
+ 数据效率：资源匮乏的问题是主要的挑战之一。
+ 多回合策略：提出了许多解决方案以解决多轮交互式训练中的这些问题，以更好地进行策略学习，包括基于模型的计划，奖励估计和端到端策略学习。
+ 本体整合：面向任务的对话系统必须查询知识库（KB）以检索一些实体以生成响应，由于没有显式的状态表示形式，因此这种简化使构造查询变得困难。

# Modules and Approaches
有关面向任务的对话系统的现有研究可以大致分为两类：pipeline和end-to-end。在pipeline方法中，该模型通常由几个组件组成，包括自然语言理解（NLU），对话状态跟踪（DST），对话策略和自然语言生成（NLG），如下图所示：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020092820241314.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
值得注意的是，尽管NLU-DST-Policy-NLG框架是pipeline系统的典型配置，但还有其他一些配置。有一些研究合并了一些典型的组件，例如单词级DST和单词级策略。在端到端方法中，对话系统在端到端方式，无需指定每个单独的组件。
+ NLU：主要是识别对话动作，其由意图和插槽值组成，即由意图识别和槽值提取组成，示例如下。

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020092820284111.png#pic_center)
+ DST：对话状态跟踪器通过将整个对话上下文作为输入来估算每个时间步的用户目标。在时间 $t$ 的对话状态可以看作是直到 $t$ 之前的对话回合的抽象表示。
+ 对话策略：以对话状态为条件，对话策略会产生下一个系统动作。如下图所示，在特定的时间步 $t$ 处，用户在 $a_t$ 处执行操作，收到奖励 $R_t$，状态更新为 $S_t$。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200928204116814.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
+ NLG：该任务将对话用作输入并生成自然语言响应。为了改善用户体验，所产生的话语应该（1）充分传达对话行为的语义以完成任务，并且（2）与人类语言类似，是自然的，特定的，信息丰富的。
+ End-to-end方法：面向任务的对话系统的端到端方法受到开放域对话系统研究的启发，如下图。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200928205223851.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
# Evaluation
大多数评估研究都遵循PARADISE框架，一种是对话成本，它衡量对话中产生的成本，例如对话回合数。另一个是任务成功，评估系统是否成功解决了用户问题。评估面向任务的对话系统的方法可以大致分为以下三种：
+ Automatic Evaluation
+ Simulated Evaluation
+ Human Evaluation

# Corpora
收集了具有不同域和注释粒度的大量语料库，以促进对面向任务的对话系统的研究。如下图所示：
+ informable slot 一般是由用户告知系统的，用来约束对话的一些条件，系统为了完成任务必须满足这些约束。
+ requestable slot 一般是用户向系统咨询的，可以来做选择的一些slot。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200928210217414.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
# Challenges
+ 数据效率：资源匮乏的问题是主要的挑战之一。回顾了为缓解此问题而提出的一些最新方法。我们首先回顾一下迁移学习方法，这些方法可以从大规模数据中获取先验知识，或者从其他任务中采用经过训练的模型。然后，我们介绍了一些无监督的方法，这些方法可以通过启发式规则在资源很少的情况下直接学习而几乎没有注释。此外，我们还回顾了最近在构建数据驱动的用户模拟器方面的工作。
+ 多回合策略：提出了许多解决方案以解决多轮交互式训练中的这些问题，以更好地进行策略学习，包括基于模型的计划，奖励估计和端到端策略学习。面向任务的对话系统的对话管理的最新研究主要集中在以下主题上：（1）带有带有用于自由槽位的值解码器的DST；（2）进行对话计划以提高策略学习中的样本效率（3）用户目标估计，以预测任务成功和用户满意度。
+ 本体整合：面向任务的对话系统必须查询知识库（KB）以检索一些实体以生成响应，由于没有显式的状态表示形式，因此这种简化使构造查询变得困难。我们介绍有关（1）对话任务模式集成（2）面向任务的对话模型中的知识库集成的一些最新进展。

# Discussion and Future Trends
在本文中，我们回顾了面向任务的对话系统的最新进展，并讨论了三个关键主题：数据效率、多回合策略、本体知识整合。最后，我们讨论面向任务的对话系统的一些未来趋势：
+ 对话系统的预训练方法
+ 领域适应，跨领域应用
+ 鲁棒性
+ End-to-end模型