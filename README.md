NLP-Paper | Still work
========================
本人在学习的过程中觉得值得阅读的论文就更新上来，对于自己精读的论文，会写一些阅读笔记上传，有兴趣的也可以一同更新完善。(内容同步更新到[CSDN](https://dengbocong.blog.csdn.net/)、[知乎](https://www.zhihu.com/people/dengbocong))

**注：对部分复现论文代码以及NLP其他工具代码放在这 ☞ [paper-code](https://github.com/DengBoCong/paper/tree/master/paper-code)**

# Contents | 内容
+ [综述](#综述)
+ [预训练](#预训练)
+ [模型](#模型)
+ [对话系统](#对话系统)
+ [语音系统](#语音系统)
+ [数据集](#数据集)
+ [评估](#评估)
+ [其他](#其他)

# Summarize | 综述
1. [A Survey on Dialogue Systems:Recent Advances and New Frontiers](https://arxiv.org/pdf/1711.01731.pdf)：对话系统的最新研究和方向 | Chen et al,2017
2. [Recent Advances and Challenges in Task-oriented Dialog Systems](https://arxiv.org/pdf/2003.07490.pdf) | [阅读笔记](https://github.com/DengBoCong/paper/blob/master/paper-list/summarize/Recent_Advances_and_Challenges_in_Task-oriented_Dialog_Systems.md)：面向任务型对话系统的最新研究和方向 | Zhang et al,2020

# Pretraining | 预训练
1. [Pretraining Methods for Dialog Context Representation Learning](https://arxiv.org/pdf/1906.00414.pdf) | [阅读笔记](https://github.com/DengBoCong/paper/blob/master/paper-list/pretraining/Pretraining_Methods_for_Dialog_Context_Representation_Learning.md)：作者列举了四种针对对话上下文表示的预训练方法，其中两种是作者新提出的 | Shikib et al,2019
2. [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/pdf/1508.07909.pdf)：就是我们所熟知的Byte Pair Encoding，是一种使用一些出现频率高的byte pair来组成新的byte的方法 | Sennrich et al,2015
3. [Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/pdf/1609.08144.pdf)：wordpiece作为BERT使用的分词方式，其生成词表的方式和BPE非常相近，区别在于BPE选择频率最高的相邻字符对进行合并，而wordpiece是基于概率生成的。 | Yonghui et al,2016
4. [Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates](https://arxiv.org/pdf/1804.10959.pdf)：unigram在给定词表及对应概率值下，直接以最大化句子的likelihood为目标来直接构建整个词表 | Kudo et al,2018
5. [Fully Character-Level Neural Machine Translation without Explicit Segmentation](https://arxiv.org/pdf/1610.03017.pdf)：比较经典的Character-Level的Subword算法模型 | Jason et al,2016
6. [Learning Character-level Representations for Part-of-Speech Tagging](http://proceedings.mlr.press/v32/santos14.pdf)：Character-level去构建word-level，该网络结构主要是对字符进行卷积以生成单词嵌入，同时使用固定窗口对PoS标记的字嵌入进行操作。 | Jason et al,2016
7. [Character-Aware Neural Language Models](https://arxiv.org/pdf/1508.06615.pdf)：提供一种功能强大，功能强大的语言模型，其可编码子词相关性，同时解决先前模型的罕见字问题，使用更少的参数获得可比较的表现力。 | Yoon et al,2015
8. [Achieving Open Vocabulary Neural Machine Translation with Hybrid Word-Character Models](https://arxiv.org/pdf/1604.00788.pdf)：一个非常出色的框架，主要是在word-level进行翻译，但是在有需要的时候可以很方便的使用Character-level的输入。 | Luong et al,2016
9. [A Joint Model for Word Embedding and Word Morphology](https://arxiv.org/pdf/1606.02601.pdf)：该模型的目标与word2vec相同，但是使用的是Character-level的输入，它使用了双向的LSTM结构尝试捕获形态并且能够推断出词根。 | Kris et al,2016
10. [Enriching Word Vectors with Subword Information](https://arxiv.org/pdf/1607.04606.pdf)：word2vec的升级版，对于具有大量形态学的稀有词和语言有更好的表征，它也可以说是带有字符n-gram的w2v skip-gram模型的扩展。 | Piotr et al,2016


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
5. [Sequential Matching Network: A New Architecture for Multi-turn Response Selection in Retrieval-Based Chatbots](https://arxiv.org/pdf/1612.01627v2.pdf) | [阅读笔记](https://github.com/DengBoCong/paper/blob/master/paper-list/dialogue/Sequential_Matching_Network_A_New_Architecture_for_Multi_turn_Response_Selection_in_Retrieva.md)：SMN检索式对话模型，多层多粒度提取信息 | Devlin et al,2018
6. [Multi-Turn Response Selection for Chatbots with Deep Attention Matching Network](https://www.aclweb.org/anthology/P18-1103.pdf) | [阅读笔记](https://github.com/DengBoCong/paper/blob/master/paper-list/dialogue/Multi_Turn_Response_Selection_for_Chatbots_with_Deep_Attention_Matching_Network.md)：DAM检索式对话模型，完全基于注意力机制的多层多粒度提取信息 | Xiangyang et al,2018

# Speech | 语音系统
1. [A Comparative Study on Transformer vs RNN in Speech Applications](https://arxiv.org/pdf/1909.06317.pdf) | [阅读笔记](https://github.com/DengBoCong/paper/blob/master/paper-list/speech/A_Comparative_Study_on_Transformer_vs_RNN_in_Speech_Applications.md)：Transformer应用在语音领域上与RNN对比的论文，并在ESPnet上面开源了模型代码 | Nanxin et al,2019
2. [Neural Speech Synthesis with Transformer Network](https://arxiv.org/pdf/1809.08895.pdf) | [阅读笔记](https://github.com/DengBoCong/paper/blob/master/paper-list/speech/Neural_Speech_Synthesis_with_Transformer_Network.md)：本文受Transformer启发，使用多头自注意力机制取代Tacotron2中的RNN结构和原始注意力机制。 | Naihan et al,2018

# Dataset | 数据集
1. [CrossWOZ: A Large-Scale Chinese Cross-Domain Task-Oriented Dialogue Dataset](https://arxiv.org/pdf/2002.11893.pdf) | [阅读笔记](https://github.com/DengBoCong/paper/blob/master/paper-list/dataset/CrossWOZ.md)：第一个大规模的中文跨域任务导向对话数据集 | Qi Zhu et al,2020
2. [MultiWOZ 2.2: A Dialogue Dataset with Additional Annotation Corrections and State Tracking Baselines](https://arxiv.org/pdf/2007.12720.pdf) | [阅读笔记](https://github.com/DengBoCong/paper/blob/master/paper-list/dataset/MultiWOZ_2_2.md)：MultiWOZ是一个著名的面向任务的对话数据集，被广泛用作对话状态跟踪的基准，MultiWOZ 2.2是目前最新版本 | Zang et al,2020
3. [The Second Dialog State Tracking Challenge](https://www.aclweb.org/anthology/W14-4337.pdf)：DSTC系列语料是专门用于对话状态跟踪的，非常经典，不过它的官网貌似无用了 |  Henderson et al,2014
4. [MuTual: A Dataset for Multi-Turn Dialogue Reasoning](https://arxiv.org/pdf/2004.04494.pdf) | [阅读笔记](https://github.com/DengBoCong/paper/blob/master/paper-list/dataset/MuTual_A_Dataset_for_Multi_Turn_Dialogue_Reasoning.md)：MuTual 数据集，用于针对性地评测模型在多轮对话中的推理能力 |  L Cui et al,2020


# Evaluate | 评估
新增
# Other | 其他
1. [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf) | [阅读笔记](https://github.com/DengBoCong/paper/blob/master/paper-list/other/Layer_Normalization.md)：层归一化方法，针对Batch Normalization的改进 | Jimmy et al,2016
2. [Scheduled Sampling for Transformers](https://arxiv.org/pdf/1906.07651.pdf) | [阅读笔记](https://github.com/DengBoCong/paper/blob/master/paper-list/other/Scheduled_Sampling_for_Transformers.md)：在Transformer应用Scheduled Sampling | Mihaylova et al,2019
3. [Self-Attention with Relative Position Representations](https://arxiv.org/pdf/1803.02155.pdf) | [阅读笔记](https://github.com/DengBoCong/paper/blob/master/paper-list/other/Self_Attention_with_Relative_Position_Representations.md)：对Transformer里面用到的位置编码进行讨论，对自注意力进行改造，从而使用相对位置编码代替硬位置编码 | Mihaylova et al,2019
4. [ProjectionNet: Learning Efficient On-Device Deep Networks Using Neural Projections](https://arxiv.org/pdf/1708.00630.pdf) | [阅读笔记](https://github.com/DengBoCong/paper/blob/master/paper-list/other/ProjectionNet_Learning_Efficient_On_Device_Deep_Networks_Using_Neural_Projections.md)：一种叫ProjectionNet的联合框架，可以为不同机器学习模型架构训练轻量的设备端模型。 | Google et al,2017

5. [Massive Exploration of Neural Machine Translation Architectures](https://arxiv.org/pdf/1703.03906.pdf) | [阅读笔记](https://github.com/DengBoCong/paper/blob/master/paper-list/other/Massive_Exploration_of_Neural_Machine_Translation_Architectures.md)：展示了以NMT架构超参数为例的首次大规模分析，实验为构建和扩展NMT体系结构带来了新颖的见解和实用建议。 | Denny et al,2017