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

# Dialogue | 对话系统

# Dataset | 数据集
1. [CrossWOZ: A Large-Scale Chinese Cross-Domain Task-Oriented Dialogue Dataset](https://arxiv.org/pdf/2002.11893.pdf) | [阅读笔记](https://github.com/DengBoCong/paper/blob/master/paper-list/dataset/CrossWOZ.md)：第一个大规模的中文跨域任务导向对话数据集 | Qi Zhu et al,2020
2. [MultiWOZ 2.2: A Dialogue Dataset with Additional Annotation Corrections and State Tracking Baselines](https://arxiv.org/pdf/2007.12720.pdf) | [阅读笔记](https://github.com/DengBoCong/paper/blob/master/paper-list/dataset/MultiWOZ_2_2.md)：MultiWOZ是一个著名的面向任务的对话数据集，被广泛用作对话状态跟踪的基准，MultiWOZ 2.2是目前最新版本 | Zang et al,2020
3. [The Second Dialog State Tracking Challenge](https://www.aclweb.org/anthology/W14-4337.pdf)DSTC系列语料是专门用于对话状态跟踪的，非常经典，不过它的官网貌似无用了 |  Henderson et al,2014

# Evaluate | 评估

# Other | 其他
1. [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf) | [阅读笔记](https://github.com/DengBoCong/paper/blob/master/paper-list/other/Layer_Normalization.md)：层归一化方法，针对Batch Normalization的改进 | Jimmy et al,2016
