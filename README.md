<h1 align="center">NLP-Paper</h1>
<div align="center">

[![Blog](https://img.shields.io/badge/blog-@DengBoCong-blue.svg?style=social)](https://www.zhihu.com/people/dengbocong)
[![Paper Support](https://img.shields.io/badge/paper-repo-blue.svg?style=social)](https://github.com/DengBoCong/nlp-paper)
![Stars Thanks](https://img.shields.io/badge/Stars-thanks-brightgreen.svg?style=social&logo=trustpilot)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=social&logo=appveyor)

</div>

更新一些在我学习过程中阅读过的且感觉不错的论文，对于一些经典或思路很新奇的论文，会进行精读，并写一些阅读笔记同步上传更新。(内容同步更新到[知乎](https://www.zhihu.com/people/dengbocong)、[CSDN](https://dengbocong.blog.csdn.net/))，**论文按照时间顺序排放**。

**注：对部分复现论文代码以及NLP其他工具代码放在这 ☞ [paper-code](https://github.com/DengBoCong/paper/tree/master/paper-code)**

# Contents | 内容
+ [综述](#综述)
+ [预训练](预训练)
+ [模型](#模型)
+ [对话系统](#对话系统)
+ [语音系统](#语音系统)
+ [数据集](#数据集)
+ [评估](#评估)
+ [文本相似度(匹配)](文本相似度(匹配))
+ [深度学习](#深度学习)
+ [机器学习](#机器学习)
+ [向量检索](#向量检索)

# Summarize | 综述
+ [A Survey on Dialogue Systems:Recent Advances and New Frontiers](https://arxiv.org/pdf/1711.01731.pdf)：对话系统的最新研究和方向 | Chen et al,2017

+ [Recent Advances and Challenges in Task-oriented Dialog Systems](https://arxiv.org/pdf/2003.07490.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/260194067)：面向任务型对话系统的最新研究和方向 | Zhang et al,2020

+ [Pre-trained Models for Natural Language Processing: A Survey](https://arxiv.org/pdf/2003.08271.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/352152573)：超详细的NLP预训练语言模型总结清单 | Xipeng Qiu et al,2020

+ [Recent Advances in Deep Learning Based Dialogue Systems: A Systematic Survey](https://arxiv.org/pdf/2105.04387.pdf): 对话系统综述：新进展新前沿 | JinJie Ni et al,2021

+ [Data Augmentation Approaches in Natural Language Processing: A Survey](https://arxiv.org/pdf/2110.01852.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/420295576)：哈工大的工作，对15中NLP数据增强方法进行了总结和对比，有详细的优缺点说明，还有一些使用技巧，实用性非常强，需要的时候可以详细的参考原文以及相关的文献的应用细节。几个开源工具：Easy DA、UNsupervised DA、nlpaug、eda_nlp_for_Chinese | Bohan Li et al,2021

# Pretraining | 预训练
+ [Character-Aware Neural Language Models](https://arxiv.org/pdf/1508.06615.pdf)：提供一种功能强大，功能强大的语言模型，其可编码子词相关性，同时解决先前模型的罕见字问题，使用更少的参数获得可比较的表现力。 | Yoon et al,2015

+ [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/pdf/1508.07909.pdf)：就是我们所熟知的Byte Pair Encoding，是一种使用一些出现频率高的byte pair来组成新的byte的方法 | Sennrich et al,2015

+ [Achieving Open Vocabulary Neural Machine Translation with Hybrid Word-Character Models](https://arxiv.org/pdf/1604.00788.pdf)：一个非常出色的框架，主要是在word-level进行翻译，但是在有需要的时候可以很方便的使用Character-level的输入。 | Luong et al,2016

+ [Learning Character-level Representations for Part-of-Speech Tagging](http://proceedings.mlr.press/v32/santos14.pdf)：Character-level去构建word-level，该网络结构主要是对字符进行卷积以生成单词嵌入，同时使用固定窗口对PoS标记的字嵌入进行操作。 | Jason et al,2016

+ [A Joint Model for Word Embedding and Word Morphology](https://arxiv.org/pdf/1606.02601.pdf)：该模型的目标与word2vec相同，但是使用的是Character-level的输入，它使用了双向的LSTM结构尝试捕获形态并且能够推断出词根。 | Kris et al,2016

+ [Enriching Word Vectors with Subword Information](https://arxiv.org/pdf/1607.04606.pdf)：word2vec的升级版，对于具有大量形态学的稀有词和语言有更好的表征，它也可以说是带有字符n-gram的w2v skip-gram模型的扩展。 | Piotr et al,2016

+ [Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/pdf/1609.08144.pdf)：wordpiece作为BERT使用的分词方式，其生成词表的方式和BPE非常相近，区别在于BPE选择频率最高的相邻字符对进行合并，而wordpiece是基于概率生成的。 | Yonghui et al,2016

+ [Fully Character-Level Neural Machine Translation without Explicit Segmentation](https://arxiv.org/pdf/1610.03017.pdf)：比较经典的Character-Level的Subword算法模型 | Jason et al,2016

+ [Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates](https://arxiv.org/pdf/1804.10959.pdf)：unigram在给定词表及对应概率值下，直接以最大化句子的likelihood为目标来直接构建整个词表 | Kudo et al,2018

+ [How to Fine-Tune BERT for Text Classification?](https://arxiv.org/pdf/1905.05583.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/358516009)：BERT在Text Classification上的一些微调实验 | Xipeng Qiu et al,2019

+ [Pretraining Methods for Dialog Context Representation Learning](https://arxiv.org/pdf/1906.00414.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/240742891)：作者列举了四种针对对话上下文表示的预训练方法，其中两种是作者新提出的 | Shikib et al,2019

+ [Pre-trained Models for Natural Language Processing: A Survey](https://arxiv.org/pdf/2003.08271.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/352152573)：超详细的NLP预训练语言模型总结清单 | Xipeng Qiu et al,2020

+ [TOD-BERT: Pre-trained Natural Language Understanding for Task-Oriented Dialogue](https://arxiv.org/pdf/2004.06871.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/377845426)：任务导向型对话的预训练自然语言理解模型 | Chien-Sheng Wu et al,2020

+ [LogME: Practical Assessment of Pre-trained Models for Transfer Learning](https://arxiv.org/pdf/2102.11005.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/358844524)：一种通用且快速的评估选择适合下游任务的预训练模型的打分方法，logME | Kaichao You et al,2021

+ [Are Pre-trained Convolutions Better than Pre-trained Transformers?](https://arxiv.org/pdf/2105.03322.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/380195756)：将Transformer的Attention换成了卷积，尝试预训练模型新方式 | Yi Tay et al,2021

# Model | 模型
+ [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf)：经典的TextCNN，static/non-static几种特征向量学习方式 | Yoon Kim et al,2014

+ [Pointer Networks](https://arxiv.org/pdf/1506.03134.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/48959800)：原文是围绕解决凸包而设计的的网络结构，直接使用Attention的权重用于预测，能够适应输入的规模，后面许多网络结构应用发展成了Copying Mechanism来解决OOV问题 | Oriol Vinyals et al,2015

+ [Incorporating Copying Mechanism in Sequence-to-Sequence Learning](https://arxiv.org/pdf/1603.06393.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/415502906)：CopyNet模型，使用Copying Mechanism来缓解未登录词问题的模型，在文本摘要等生成词多含输入词的任务中，效果不错 | Jiatao Gu et al,2016

+ [Language Modeling with Gated Convolutional Networks](https://arxiv.org/pdf/1612.08083.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/395977833)：受LSTM门控机制的启发，将线性门控机制应用于卷积结构，文中对比GLU、GTU等结构性能 | Yann N. Dauphin et al,2016

+ [A SIMPLE BUT TOUGH-TO-BEAT BASELINE FOR SENTENCE EMBEDDINGS](https://openreview.net/pdf?id=SyK00v5xx)：Smooth Inverse Frequency，一种简单但是效果好的Sentence Embedding方法 | Sanjeev Arora et al,2017

+ [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/pdf/1704.04368.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/419659043)：结合Copying Mechanism和Coverage mechanism两种技巧的LSTM-Base模型，一定程度上解决OOV和重复词问题，经典值得一读 | Abigail See et al,2017

+ [Supervised Learning of Universal Sentence Representations from Natural Language Inference Data](https://arxiv.org/pdf/1705.02364.pdf)：InferSent，通过不同的encoder得到Sentence Embedding，并计算两者差值、点乘得到交互向量，从而得到相似度。 | Alexis Conneau et al,2017

+ [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/250946855)：Transformer的开山之作，值得精读 | Ashish et al,2017

+ [Unsupervised Random Walk Sentence Embeddings: A Strong but Simple Baseline](https://www.aclweb.org/anthology/W18-3012.pdf)：Unsupervised Smooth Inverse Frequency，USIF改进SIF对句向量长度敏感，在相似度任务上提升很大 | Kawin Ethayarajh Arora et al,2018

+ [Multi-Cast Attention Networks for Retrieval-based Question Answering and Response Prediction](https://arxiv.org/pdf/1806.00778.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/349369847)：一种用于通用序列对建模的整体架构，结合多种注意力机制进行特征增强 | Yi Tay et al,2018

+ [Sliced Recurrent Neural Networks](https://arxiv.org/ftp/arxiv/papers/1807/1807.02291.pdf)：切片RNN网络，尝试突破RNN时序限制的模型 | Zeping Yu et al,2018

+ [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/269997771)：BERT的顶顶大名，使用Transformer的Encoder双向架构 | Devlin et al,2018

+ [Pay Less Attention With Lightweight And Dynamic Convolutions](https://arxiv.org/pdf/1901.10430.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/396143249)：论文研究Lightweight、Dynamic Convolutions，卷积结构同样能够达到和Self-Attention媲美的效果 | Felix Wu et al,2019

+ [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/pdf/1906.08237.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/361737484)：XLNet--自回归语言模型的复兴，30多项任务超越BERT | Zhilin Yang et al,2019

+ [CTRL: A Conditional Transformer Language Model For Controllable Generation](https://arxiv.org/pdf/1909.05858.pdf)
| [阅读笔记](https://zhuanlan.zhihu.com/p/405493225)：CTRL语言模型，提供Control Code进行定向文本生成，相较于GPT可对文本风格进行控制 | Keskar et al,2019

+ [Reformer: The Efficient Transformer](https://arxiv.org/pdf/2001.04451.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/411882151)：使用LSH Attention、Reversible layers、Chunking FFN layers，降低Transformer计算复杂度和内存空间消耗 | Nikita Kitaev et al,2020

+ [Synthesizer: Rethinking Self-Attention for Transformer Models](https://arxiv.org/pdf/2005.00743.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/380602965)：在Transformer架构下，对Self-Attention计算的探索研究，看完会对Self-Attention有个新认识 | Yi Tay et al,2020

+ [Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://arxiv.org/pdf/2012.07436.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/351321328)：一种效果远超Transformer的长序列预测模型，针对LSTF问题上的研究改进 | Haoyi Zhou et al,2020

# Dialogue | 对话系统
+ [The Hidden Information State model: A practical framework for POMDP-based spoken dialogue management](https://www.sciencedirect.com/science/article/abs/pii/S0885230809000230)：关于对话状态管理的文章，可以用来补充相关背景知识 | Young et al,2010

+ [Context Sensitive Spoken Language Understanding Using Role Dependent LSTM Layers](https://www.merl.com/publications/docs/TR2015-134.pdf)：使用LSTM在SLU方面做的工作，通过agent和client角色划分，能够解决多轮对话中的歧义问题 | Hori et al,2015

+ [A Neural Conversational Model](https://arxiv.org/pdf/1506.05869.pdf)：Seq2Seq结构的对话模型 | Oriol et al,2015

+ [A Network-based End-to-End Trainable Task-oriented Dialogue System](https://arxiv.org/pdf/1604.04562.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/261701071)：非常值得一读的任务型对话模型架构 | Wen et al,2016

+ [Neural Belief Tracker: Data-Driven Dialogue State Tracking](https://arxiv.org/pdf/1606.03777.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/262289823)：NBT框架，理解Belief state和tracking的好文 | Young et al,2016

+ [Attention-Based Recurrent Neural Network Models for Joint Intent Detection and Slot Filling](https://arxiv.org/pdf/1609.01454.pdf)：使用Attention-Based的RNN模型进行联合意图识别和槽位填充，达到不错的效果 | Bing Liu et al,2016

+ [Sequential Matching Network: A New Architecture for Multi-turn Response Selection in Retrieval-Based Chatbots](https://arxiv.org/pdf/1612.01627v2.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/270554147)：SMN检索式对话模型，多层多粒度提取信息 | Devlin et al,2016

+ [Latent Intention Dialogue Models](https://arxiv.org/pdf/1705.10229.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/263034049)：离散潜在变量模型学习对话意图的框架 | Wen et al,2017

+ [An End-to-End Trainable Neural Network Model with Belief Tracking for Task-Oriented Dialog](https://arxiv.org/pdf/1708.05956.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/260345363)：面向任务的对话系统的新型端到端可训练神经网络模型 | Liu et al,2017

+ [Multi-Turn Response Selection for Chatbots with Deep Attention Matching Network](https://www.aclweb.org/anthology/P18-1103.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/306846122)：DAM检索式对话模型，完全基于注意力机制的多层多粒度提取信息 | Xiangyang et al,2018

+ [Slot-Gated Modeling for Joint Slot Filling and Intent Prediction](https://aclanthology.org/N18-2118.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/413261222)：提出了Slot-Gated机制，联合意图识别和槽位填充效果提升 | Chih-Wen Goo et al,2018

+ [Global-Locally Self-Attentive Dialogue State Tracker](https://arxiv.org/pdf/1805.09655.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/266982344)：全局-局部自注意力状态跟踪 | Zhong et al,2018

+ [Sequence-to-Sequence Data Augmentation for Dialogue Language Understanding](https://arxiv.org/pdf/1807.01554.pdf)：使用seq2seq生成模型对语义文本进行数据增强，核心步骤为Delexicalisation->Diversity rank->generation->surface realisation | Yutai Hou et al,2018

+ [Data Augmentation with Atomic Templates for Spoken Language Understanding](https://arxiv.org/pdf/1908.10770.pdf)：使用Atomic Templates（act-slot-value）进行对话数据增强，使用seq2seq生成模型进行语句生成 | Zijian Zhao et al,2019

+ [A Closer Look At Feature Space Data Augmentation For Few-Shot Intent Classification](https://arxiv.org/pdf/1910.04176.pdf)：针对SLU的Intent分类任务，对其文本数据进行数据增强并比较效果，其中Linear+Transfer learning效果最佳 | Varun Kumar et al,2019

+ [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/pdf/2004.04906.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/403589222)：DPR一种高效的开放域问答检索技术，应用了BERT进行编码 | Karpukhin et al,2020

+ [TOD-BERT: Pre-trained Natural Language Understanding for Task-Oriented Dialogue](https://arxiv.org/pdf/2004.06871.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/377845426)：任务导向型对话的预训练自然语言理解模型 | Chien-Sheng Wu et al,2020

+ [Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering](https://arxiv.org/pdf/2007.01282.pdf)：Fusion-in-Decoder生成式阅读理解模型 | Izacard et al,2020

+ [DISTILLING KNOWLEDGE FROM READER TO RETRIEVER FOR QUESTION ANSWERING](https://openreview.net/pdf?id=NTEz-6wysdb) | [阅读笔记](https://zhuanlan.zhihu.com/p/372694270)：一种模型训练模型的开放域问答方法 | Izacard et al,2021

+ [Retrieve & Memorize: Dialog Policy Learning with Multi-Action Memory](https://arxiv.org/pdf/2106.02317.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/415170940)：联合检索和记忆块的多action的Dialog Policy Learning模型，在action生成和response生成上效果SOTA | Yunhao Li et al,2021

+ [Increasing Faithfulness in Knowledge-Grounded Dialogue with Controllable Features](https://arxiv.org/pdf/2107.06963.pdf)：通过可控特征来增加知识对话系统的学习 | Rashkin et al,2021

+ [Constraint based Knowledge Base Distillation in End-to-End Task Oriented Dialogs](https://arxiv.org/pdf/2109.07396.pdf)：基于KB的End2End的Task-Oriented的对话系统，使用pairwise相似度过滤相关信息来获得KB中的n元结构，就这一点上倒没有什么新奇，只不过相对于之前的方式修改的entity格式。不过在避免检索到部分entity相似但并不是目标的record的情况，作者加入了辅助的损失函数用于embedding constraint，这种做法确实减少了相同entity之间的相似性，从而提高record的可靠性，值得借鉴。基于现有的F1指标的缺点，提出multiset entity F1 | Dinesh Raghu et al,2021

# Speech | 语音系统
+ [Attention-Based Models for Speech Recognition](https://proceedings.neurips.cc/paper/2015/file/1068c6e4c8051cfd4e9ea8072e3189e2-Paper.pdf)：Tacotron2使用的Location Sensitive Attention  |  Chorowski et al,2015

+ [Tacotron: A Fully End-To-End Text-To-Speech Synthesis Model](http://bengio.abracadoudou.com/cv/publications/pdf/wang_2017_arxiv.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/337042442)：Tacotron，端到端的语音合成系统 | Yuxuan et al,2017

+ [Natural TTS Synthesis By Conditioning Wavenet On Mel Spectrogram Predictions](https://arxiv.org/pdf/1712.05884.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/337042442)：Tacotron2，相较于Tacotron有着更好的性能，使用WaveNet作为Vocoder | Jonathan et al,2017

+ [Syllable-Based Sequence-to-Sequence Speech Recognition with the Transformer in Mandarin Chinese](https://arxiv.org/pdf/1804.10752.pdf)：使用Transformer应用在普通话语音识别，数据集是HKUST datasets  |  Shiyu et al,2018

+ [Neural Speech Synthesis with Transformer Network](https://arxiv.org/pdf/1809.08895.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/332316226)：本文受Transformer启发，使用多头自注意力机制取代Tacotron2中的RNN结构和原始注意力机制。 | Naihan et al,2018

+ [A Comparative Study on Transformer vs RNN in Speech Applications](https://arxiv.org/pdf/1909.06317.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/309390439)：Transformer应用在语音领域上与RNN对比的论文，并在ESPnet上面开源了模型代码 | Nanxin et al,2019

# Dataset | 数据集
+ [The Second Dialog State Tracking Challenge](https://www.aclweb.org/anthology/W14-4337.pdf)：DSTC系列语料是专门用于对话状态跟踪的，非常经典，不过它的官网貌似无用了 |  Henderson et al,2014

+ [The Ubuntu Dialogue Corpus: A Large Dataset for Research in Unstructured Multi-Turn Dialogue Systems](https://arxiv.org/pdf/1506.08909.pdf)：Ubuntu 非结构化多轮对话数据集 |  Ryan Lowe et al,2015

+ [DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset](https://arxiv.org/pdf/1710.03957.pdf) | [数据集地址](https://drive.google.com/file/d/1sj3Z_GZfYzrhmleWazA-QawhUEhlNmJd/view?usp=sharing)：包含对话意图和情感信息的多轮对话数据集 | Yanran Li et al, 2017

+ [CrossWOZ: A Large-Scale Chinese Cross-Domain Task-Oriented Dialogue Dataset](https://arxiv.org/pdf/2002.11893.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/259861746)：第一个大规模的中文跨域任务导向对话数据集 | Qi Zhu et al,2020

+ [Improving Dialog Evaluation with a Multi-reference Adversarial Dataset and Large Scale Pretraining](https://scholar.google.com/scholar_url?url=https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl_a_00347/1923874/tacl_a_00347.pdf&hl=zh-CN&sa=T&oi=gsb-gga&ct=res&cd=0&d=13355199831609160829&ei=hXzkYNupCsyO6rQPkrG1wAo&scisig=AAGBfm39FeIrjR-BGf074wiUqDueImjYeA) | [数据集地址](https://github.com/iitmnlp/Dialogue-Evaluation-with-BERT)：DailyDialog数据集的升级版，11K的多轮对话上下文，每个上下文包括五个标准的参考回复、五个不相关的回复、五个随机挑选的回复 | Ananya B. Sai et al, 2020

+ [MuTual: A Dataset for Multi-Turn Dialogue Reasoning](https://arxiv.org/pdf/2004.04494.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/282843192)：MuTual 数据集，用于针对性地评测模型在多轮对话中的推理能力 |  L Cui et al,2020

+ [MultiWOZ 2.2: A Dialogue Dataset with Additional Annotation Corrections and State Tracking Baselines](https://arxiv.org/pdf/2007.12720.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/260097352)：MultiWOZ是一个著名的面向任务的对话数据集，被广泛用作对话状态跟踪的基准，MultiWOZ 2.2是目前最新版本 | Zang et al,2020

# Evaluate | 评估
+ [LogME: Practical Assessment of Pre-trained Models for Transfer Learning](https://arxiv.org/pdf/2102.11005.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/358844524)：一种通用且快速的评估选择适合下游任务的预训练模型的打分方法，logME | Kaichao You et al,2021 

+ [Towards Quantifiable Dialogue Coherence Evaluation](https://arxiv.org/pdf/2106.00507.pdf)：QuantiDCE，一种实现可量化的对话连贯性评估指标模型 | Zheng Ye et al,2021

# Text Similarity | 文本相似度(匹配)
+ [Siamese Recurrent Architectures for Learning Sentence Similarity](https://scholar.google.com/scholar_url?url=https://ojs.aaai.org/index.php/AAAI/article/view/10350/10209&hl=zh-CN&sa=T&oi=gsb-gga&ct=res&cd=0&d=7393466935379636447&ei=KQWzYNL5OYz4yATXqJ6YCg&scisig=AAGBfm0zNEZZez8zh5ZB_iG7UTrwXmhJWg)：Siamese LSTM，一个用来计算句对相似度的模型 | Jonas Mueller et al,2016

# Deep Learning | 深度学习
+ [NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE](https://arxiv.org/pdf/1409.0473.pdf)：Bahdanau Attention的原文 | Bahdanau et al,2014

+ [Convolutional Neural Networks at Constrained Time Cost](https://arxiv.org/pdf/1412.1710.pdf)：针对卷积网络很好地概述了计算成本以及深度，过滤器尺寸之间的权衡 | Kaiming He et al,2014

+ [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/340219662)：经典的Batch Normalization原论文 | Sergey et al,2015

+ [Learning both Weights and Connections for Efficient Neural Networks](https://arxiv.org/pdf/1506.02626.pdf)：有一张表格，其中列出了计算与内存访问的相对成本，除此之外还讨论了怎么精简神经网络 | Song Han et al,2015

+ [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/pdf/1508.04025.pdf)：Luong Attention的原文 | Luong et al,2015

+ [Strategies for Training Large Vocabulary Neural Language Models](https://arxiv.org/pdf/1512.04906.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/397084135)：主要是对当时的一些Softmax和Sampling进行总结，顺便提出了Differentiated Softmax方法 | Wenlin Chen et al,2015

+ [Exploring the Limits of Language Modeling](https://arxiv.org/pdf/1602.02410.pdf)：CNN Softmax方法，虽然还是离不开原始的Softmax，但是换了一个视角效果很好 | Rafal Jozefowicz et al,2016

+ [Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks](https://arxiv.org/pdf/1602.07868.pdf)：Weight Normalization是一种在权值维度上进行归一化的方法 | Tim Salimans et al,2016

+ [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/258977332)：层归一化方法，针对Batch Normalization的改进 | Jimmy et al,2016

+ [Instance Normalization:The Missing Ingredient for Fast Stylization](https://arxiv.org/pdf/1607.08022.pdf)：Instance Normalization是一种不受限于批量大小的算法专门用于Texture Network中的生成器网络 | Dmitry Ulyanov et al,2016

+ [Efficient softmax approximation for GPUs](https://arxiv.org/pdf/1609.04309.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/397084135)：Adaptive Softmax，针对GPU的矩阵计算，实现了多倍与普通Softmax计算效率的提升，值得一看 | Edouard Grave et al,2016

+ [Large-Margin Softmax Loss for Convolutional Neural Networks](https://arxiv.org/pdf/1612.02295.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/397084135)：L-Softmax在原Softmax的基础上增加了控制系数m，使得类内距离尽可能小，类间距离尽可能大 | Weiyang Liu et al,2016

+ [An empirical analysis of the optimization of deep network loss surfaces](https://arxiv.org/pdf/1612.04010.pdf)：论文中得出一个结论，即Batch Normalization更有利于梯度下降 | Shibani et al,2016

+ [Cosine Normalization: Using Cosine Similarity Instead of Dot Product in Neural Networks](https://arxiv.org/pdf/1702.05870v5.pdf)：Cosine Normalization是一种将unbounded的向量点积换成夹角余弦操作，从而进行归一化的方法 | Luo Chunjie et al, 2017

+ [Massive Exploration of Neural Machine Translation Architectures](https://arxiv.org/pdf/1703.03906.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/328801239)：展示了以NMT架构超参数为例的首次大规模分析，实验为构建和扩展NMT体系结构带来了新颖的见解和实用建议。 | Denny et al,2017

+ [SphereFace: Deep Hypersphere Embedding for Face Recognition](https://arxiv.org/pdf/1704.08063.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/397084135)：A-Softmax，思路和L-Softmax差不多，区别是对权重进行了归一化 | Weiyang Liu et al,2017

+ [ProjectionNet: Learning Efficient On-Device Deep Networks Using Neural Projections](https://arxiv.org/pdf/1708.00630.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/295636122)：一种叫ProjectionNet的联合框架，可以为不同机器学习模型架构训练轻量的设备端模型。 | Google et al,2017

+ [Additive Margin Softmax for Face Verification](https://arxiv.org/pdf/1801.05599.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/397084135)：AM-Softmax在A-Softmax的最大区别是AM是角度距离，A是余弦距离

+ [Self-Attention with Relative Position Representations](https://arxiv.org/pdf/1803.02155.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/268649069)：对Transformer里面用到的位置编码进行讨论，对自注意力进行改造，从而使用相对位置编码代替硬位置编码 | Mihaylova et al,2018

+ [Group Normalization](https://arxiv.org/pdf/1803.08494.pdf)：Group Normalization是将输入的通道分成较小的子组，并根据其均值和方差归一化这些值 | Yuxin Wu et al,2018

+ [How Does Batch Normalization Help Optimization?](https://arxiv.org/pdf/1805.11604.pdf)：讨论Batch Normalization是如何帮助优化器工作的，主要结论是BN层能够让损失函数更加平滑 | Shibani et al,2018

+ [Scheduled Sampling for Transformers](https://arxiv.org/pdf/1906.07651.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/267146739)：在Transformer应用Scheduled Sampling | Mihaylova et al,2019

+ [Consistency of a Recurrent Language Model With Respect to Incomplete Decoding](https://arxiv.org/pdf/2002.02492.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/349675973)：讨论Seq2Seq模型解码停不下来的原因 | Sean Welleck et al,2020

+ [PowerNorm: Rethinking Batch Normalization in Transformers](https://arxiv.org/pdf/2003.07845.pdf)：对于Transformer中BN表现不好的原因做了一定的empirical和theoretical的分析 | Sheng Shen et al,2020

+ [Shortcut Learning in Deep Neural Networks](https://arxiv.org/pdf/2004.07780.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/421175552)：对Shortcut Learning问题进行比较详细的解释和剖析，虽然最后没有给出实际的解决方案（Shortcut Learning问题本身就没有一个体系化的策略，需要根据实际任务而定），不过提供了几种解决的视角 | Robert Geirhos et al,2020

+ [Beyond Accuracy: Behavioral Testing of NLP Models with CheckList](https://arxiv.org/pdf/2005.04118.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/421175552)：ACL2020的Best Paper，基于NLP领域提出了测试体系来指导我们了解 NLP 模型的能力，也能够指导我们去理解问题、解决问题。不同于现代 NLP 模型常常仅关注特定的任务，CheckList 希望去评估一个模型的多方面能力，这些能力有的是模型通用的，有的则是面向特定的任务或领域 | Marco Tulio Ribeiro et al,2020

+ [A Theoretical Analysis of the Repetition Problem in Text Generation](https://arxiv.org/pdf/2012.14660.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/349675973)：讨论Seq2Seq模型解码重复生成的原因 | Zihao Fu et al,2020

# Machine Learning | 机器学习

+ [Optimal Whitening and Decorrelation](https://arxiv.org/pdf/1512.00809.pdf)：提供五种白化方法的数学证明 | Agnan Kessy et al,2015

+ [An overview of gradient descent optimization algorithms](https://arxiv.org/pdf/1609.04747.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/343564175)：对当前主流的梯度下降算法进行概述 | Sebastian Ruder et al,2016

+ [Covariate Shift: A Review and Analysis on Classifiers](https://ieeexplore.ieee.org/abstract/document/8978471) | [阅读笔记](https://zhuanlan.zhihu.com/p/339719861)：通过几种分类算法，在四种不同的数据集下验证几种方法处理Covariate Shift问题后的性能分析 | Geeta et al,2019

# Nearest Neighbor | 向量检索
