NLP-Paper | Still work
========================
本人在学习的过程中觉得值得阅读的论文就更新上来，对于自己精读的论文，会写一些阅读笔记上传，有兴趣的也可以一同更新完善。(内容同步更新到[CSDN](https://dengbocong.blog.csdn.net/)、[知乎](https://www.zhihu.com/people/dengbocong))

**注：对部分复现论文代码放在这 ☞ [paper-code](https://github.com/DengBoCong/paper/tree/master/paper-code)**

# Contents | 内容
+ [综述](#综述)
+ [预训练](#预训练)
+ [模型](#模型)
+ [对话系统](#对话系统)
+ [数据集](#数据集)
+ [评估](#评估)
+ [其他](#其他)

# Summarize | 综述
1. [A Survey on Dialogue Systems:Recent Advances and New Frontiers](https://arxiv.org/pdf/1711.01731.pdf)：对话系统的最新研究和方向 | Chen et al,2017
2. [Recent Advances and Challenges in Task-oriented Dialog Systems](https://arxiv.org/pdf/2003.07490.pdf) | [阅读笔记](https://github.com/DengBoCong/paper/blob/master/paper-list/summarize/Recent_Advances_and_Challenges_in_Task-oriented_Dialog_Systems.md)：面向任务型对话系统的最新研究和方向 | Zhang et al,2020

# Pretraining | 预训练
1. [Pretraining Methods for Dialog Context Representation Learning](https://arxiv.org/pdf/1906.00414.pdf) | [阅读笔记](https://github.com/DengBoCong/paper/blob/master/paper-list/pretraining/Pretraining_Methods_for_Dialog_Context_Representation_Learning.md)：作者列举了四种针对对话上下文表示的预训练方法，其中两种是作者新提出的 | Shikib et al,2019

# Model | 模型
1. [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) | [阅读笔记](https://github.com/DengBoCong/paper/blob/master/paper-list/model/Attention_Is_All_You_Need.md)：Transformer的开山之作，值得精读 | Ashish et al,2017
2. [A Neural Conversational Model](https://arxiv.org/pdf/1506.05869.pdf)：Seq2Seq结构的对话模型 | Oriol et al,2015
3. [Context Sensitive Spoken Language Understanding Using Role Dependent LSTM Layers](https://www.merl.com/publications/docs/TR2015-134.pdf)：使用LSTM在SLU方面做的工作，通过agent和client角色划分，能够解决多轮对话中的歧义问题 | Hori et al,2015
4. [An End-to-End Trainable Neural Network Model with Belief Tracking for Task-Oriented Dialog](https://arxiv.org/pdf/1708.05956.pdf) | [阅读笔记](https://github.com/DengBoCong/paper/blob/master/paper-list/model/An_End-to-End_Trainable_Neural_Network_Model_with_Belief_Tracking_for_Task-Oriented_Dialog.md)：面向任务的对话系统的新型端到端可训练神经网络模型 | Liu et al,2017
5. [A Network-based End-to-End Trainable Task-oriented Dialogue System](https://arxiv.org/pdf/1604.04562.pdf) | [阅读笔记](https://github.com/DengBoCong/paper/blob/master/paper-list/model/A_Network-based_End-to-End_Trainable_Task-oriented_Dialogue_System.md)：非常值得一读的任务型对话模型架构 | Wen et al,2016
6. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf) | [阅读笔记](https://github.com/DengBoCong/paper/blob/master/paper-list/model/BERT_Pre-training_of_Deep_Bidirectional_Transformers_for_Language_Understanding.md)：BERT的顶顶大名，使用Transformer的Encoder双向架构 | Devlin et al,2018

# Dialogue | 对话系统
1. [The Hidden Information State model: A practical framework for POMDP-based spoken dialogue management](https://www.sciencedirect.com/science/article/abs/pii/S0885230809000230)：关于对话状态管理的文章，可以用来补充相关背景知识 | Young et al,2010
2. [Neural Belief Tracker: Data-Driven Dialogue State Tracking](https://arxiv.org/pdf/1606.03777.pdf) | [阅读笔记](https://github.com/DengBoCong/paper/blob/master/paper-list/dialogue/Neural_Belief_Tracker_Data-Driven_Dialogue_State_Tracking.md)：NBT框架，理解Belief state和tracking的好文 | Young et al,2017
3. [Latent Intention Dialogue Models](https://arxiv.org/pdf/1705.10229.pdf) | [阅读笔记](https://github.com/DengBoCong/paper/blob/master/paper-list/dialogue/Latent_Intention_Dialogue_Models.md)：离散潜在变量模型学习对话意图的框架 | Wen et al,2017
4. [Global-Locally Self-Attentive Dialogue State Tracker](https://arxiv.org/pdf/1805.09655.pdf) | [阅读笔记](https://github.com/DengBoCong/paper/blob/master/paper-list/dialogue/Global_Locally_Self_Attentive_Dialogue_State_Tracker.md)：全局-局部自注意力状态跟踪 | Zhong et al,2018

# Dataset | 数据集
1. [CrossWOZ: A Large-Scale Chinese Cross-Domain Task-Oriented Dialogue Dataset](https://arxiv.org/pdf/2002.11893.pdf) | [阅读笔记](https://github.com/DengBoCong/paper/blob/master/paper-list/dataset/CrossWOZ.md)：第一个大规模的中文跨域任务导向对话数据集 | Qi Zhu et al,2020
2. [MultiWOZ 2.2: A Dialogue Dataset with Additional Annotation Corrections and State Tracking Baselines](https://arxiv.org/pdf/2007.12720.pdf) | [阅读笔记](https://github.com/DengBoCong/paper/blob/master/paper-list/dataset/MultiWOZ_2_2.md)：MultiWOZ是一个著名的面向任务的对话数据集，被广泛用作对话状态跟踪的基准，MultiWOZ 2.2是目前最新版本 | Zang et al,2020
3. [The Second Dialog State Tracking Challenge](https://www.aclweb.org/anthology/W14-4337.pdf)DSTC系列语料是专门用于对话状态跟踪的，非常经典，不过它的官网貌似无用了 |  Henderson et al,2014

# Evaluate | 评估
新增
# Other | 其他
1. [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf) | [阅读笔记](https://github.com/DengBoCong/paper/blob/master/paper-list/other/Layer_Normalization.md)：层归一化方法，针对Batch Normalization的改进 | Jimmy et al,2016
2. [Scheduled Sampling for Transformers](https://arxiv.org/pdf/1906.07651.pdf) | [阅读笔记](https://github.com/DengBoCong/paper/blob/master/paper-list/other/Scheduled_Sampling_for_Transformers.md)：在Transformer应用Scheduled Sampling | Mihaylova et al,2019
3. [Self-Attention with Relative Position Representations](https://arxiv.org/pdf/1803.02155.pdf) | [阅读笔记](https://github.com/DengBoCong/paper/blob/master/paper-list/other/Self_Attention_with_Relative_Position_Representations.md)：对Transformer里面用到的位置编码进行讨论，对自注意力进行改造，从而使用相对位置编码代替硬位置编码 | Mihaylova et al,2019