
[A Survey on Dialogue Systems:Recent Advances and New Frontiers](https://arxiv.org/pdf/1711.01731.pdf)


[A Neural Conversational Model](https://arxiv.org/pdf/1506.05869.pdf)：在IT Helpdesk Troubleshooting数据集和OpenSubtitles数据集上的seq2seq模型的实验结果。
>会话建模是自然语言理解和机器智能中的重要任务，尽管存在先前的方法，但是它们通常限于特定的领域（例如，预订机票），并且需要手工制定的规则。在本文中，我们提出了一种用于此任务的简单方法，该方法使用最近提出的seq2seq框架。我们的模型通过在对话中给定前一个或多个句子的情况下预测下一个句子来进行交谈。我们的模型的优势在于可以端到端进行训练，因此需要的手工规则要少得多。我们发现，给定大量的会话训练数据集，这种简单的模型可以生成简单的会话。我们的初步结果表明，尽管优化了错误的目标函数，该模型仍能很好地进行对话。它既可以从特定于域的数据集，也可以从大型，嘈杂且通用的电影字幕域数据集中提取知识。在特定领域的IT Helpdesk Troubleshooting数据集上，该模型可以通过对话找到技术问题的解决方案。在嘈杂的开放域OpenSubtitles数据集上，该模型可以执行常识推理的简单形式。不出所料，我们还发现缺乏一致性是我们模型的常见问题。