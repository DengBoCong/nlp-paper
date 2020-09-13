# 快速导航
+ [A Survey on Dialogue Systems:Recent Advances and New Frontiers](#a-survey-on-dialogue-systems:recent-advances-and-new-frontiers)
+ [A Neural Conversational Model](#a-neural-conversational-model)
+ [Attention Is All You Need](#attention-is-all-you-need)


# A Survey on Dialogue Systems:Recent Advances and New Frontiers
[论文地址](https://arxiv.org/pdf/1711.01731.pdf)


# A Neural Conversational Model
[论文地址](https://arxiv.org/pdf/1506.05869.pdf)：在IT Helpdesk Troubleshooting数据集和OpenSubtitles数据集上的seq2seq模型的实验结果。
>会话建模是自然语言理解和机器智能中的重要任务，尽管存在先前的方法，但是它们通常限于特定的领域（例如，预订机票），并且需要手工制定的规则。在本文中，我们提出了一种用于此任务的简单方法，该方法使用最近提出的seq2seq框架。我们的模型通过在对话中给定前一个或多个句子的情况下预测下一个句子来进行交谈。我们的模型的优势在于可以端到端进行训练，因此需要的手工规则要少得多。我们发现，给定大量的会话训练数据集，这种简单的模型可以生成简单的会话。我们的初步结果表明，尽管优化了错误的目标函数，该模型仍能很好地进行对话。它既可以从特定于域的数据集，也可以从大型，嘈杂且通用的电影字幕域数据集中提取知识。在特定领域的IT Helpdesk Troubleshooting数据集上，该模型可以通过对话找到技术问题的解决方案。在嘈杂的开放域OpenSubtitles数据集上，该模型可以执行常识推理的简单形式。不出所料，我们还发现缺乏一致性是我们模型的常见问题。

# Attention Is All You Need
[论文地址](https://arxiv.org/pdf/1706.03762.pdf)
>目前占据主导的序列转换模型主要基于复杂的递归或卷积神经网络，包括编码器和解码器。表现最佳的模型还通过注意力机制连接编码器和解码器。我们提出了一种新的简单网络架构，即Transformer，它完全基于注意力机制，没有使用循环或者卷积操作。在两个机器翻译任务上进行的实验表明，这些模型在质量上具有优势，同时具有更高的可并行性，并且所需的训练时间明显更少。我们的模型在WMT 2014 English-to-German翻译任务上获得了28.4 BLEU，包括集成在内，比现有的最好结果高2 BLEU。在WMT 2014 English-to-French翻译任务上，我们的模型创建了单模型的41.8 BLEU的记录，在八卡GPU训练了3.5天之后，仅有之前最优模型的一小部分训练代价。我们展示了Transformer通过将其成功应用于具有大量训练数据和有限训练数据的英语选区解析，将其很好地应用到了其他任务。

# Self-Attention with Relative Position Representations
[论文地址](https://arxiv.org/pdf/1803.02155.pdf)


# Pretraining Methods for Dialog Context Representation Learning
[论文地址](https://arxiv.org/pdf/1906.00414.pdf)
> 多轮对话中对话历史的表示学习是构建对话系统的基础。为了生成更有质量的回复，对话系统需要聚合来自多轮的信息。之前的研究更关注于单领域内对话系统网络结构的改进，而近年来在大量文本数据上预训练的模型在很多NLP任务中表现突出，比如BERT。目前预训练方法仍处在起步阶段，我们仍然不能完全了解他们的性质。大多数方法都是基于语言模型的，给一个句子，预测当前词，下一个词或者被mask的词。如Word2Vec，Glove，ELMO等。这些方法将自然语言看作是word token的流，需要复杂的模型利用大规模的语料库和庞杂的计算来发现更高级别的依赖关系。BERT模型也是基于语言模型，但是加入了句子对级别的信息，预测两句话是否是连续的。这种方法在预训练时利用了语句之间的关系。但是，在对话上下文建模这种存在多轮的依赖关系的任务上还并没有行之有效的预训练方法，于是本文在这个方面做了一些尝试。