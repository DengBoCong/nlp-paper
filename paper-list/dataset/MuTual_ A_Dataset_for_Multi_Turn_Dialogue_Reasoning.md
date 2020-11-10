# 前言

> 标题：MuTual: A Dataset for Multi-Turn Dialogue Reasoning\
> 原文链接：[Link](https://arxiv.org/pdf/2004.04494.pdf)\
> Github：[NLP相关Paper笔记和代码复现](https://github.com/DengBoCong/nlp-paper)\
> 说明：阅读论文时进行相关思想、结构、优缺点，内容进行提炼和记录，论文和相关引用会标明出处，引用之处如有侵权，烦请告知删除。\
> 转载请注明：DengBoCong

# 介绍
面向非任务的对话系统在给定上下文的情况下，当前系统能够产生相关且流畅的回复，但是由于推理能力较弱，有时会出现逻辑错误。为了促进对话推理研究，发布了多轮对话推理数据集 MuTual，针对性地评测模型在多轮对话中的推理能力。它由基于中国学生英语听力理解考试的8,860个手动注释的对话组成，数据集的[GitHub仓库](https://github.com/Nealcly/MuTual)。

神经对话系统在大型对话语料库上进行训练，并用于预测给定上下文的回复。一般情况下，构建聊天机器人主要有两种解决方案：
+ 检索式的方法依赖文本匹配技术，在诸多候选回复中，选择匹配分数最高的作为回复；
+ 生成式的方法使用 Seq2Seq 模型，编码器读取对话历史，解码器直接生成相应回复。

检索式的方法凭借回复相关性高，流利度高等优势，在工业界取得了更多的应用。不过，虽然在以BERT为代表的预训练模型，在检索式多轮对话任务中，已经基本接近了人类的表现。但实际应用中，当前的对话模型选择出的回复往往相关性较好，但是经常出现常识和逻辑错误等问题。由于现有的大部分检索式对话数据集都没有关注这种对话逻辑问题，导致评价指标也无法直接反映模型对对话逻辑的掌握程度。针对此问题，提出了多轮对话推理数据集 MuTual。

相比现有的其他检索式聊天数据集，MuTual 要求对话模型具备常识推理能力；相比阅读理解式的推理数据集，MuTual 的输入输出则完全符合标准检索式聊天机器人的流程。因此，MuTual 也是目前最具挑战性的对话式数据集。由于任务不同，目前现有的推理数据集并不能直接帮助指导训练聊天机器人。下表为对话和基于推理的阅读理解的常用数据集：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201109234130781.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
MuTual是第一个基于人标签推理的多轮对话数据集，用最佳方法在此数据集上运行的 $R@1$ 为71％，明显不如人类的表现（94％）。

# 数据集
原始的听力理解材料和问答对是由语言专家设计的，学生需要根据一段音频从三个选项中选择最佳答案，为了确保学生完全理解音频，大部分问题都需要具备推理能力。原始数据的格式设置为三元组 <Conversation (audio),Question and Choices (text), Answer (image)>
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201110090755102.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
听力考试要求学生根据一段双人多轮对话，回答额外提出的问题（图1左），并通过学生能否正确答对问题衡量学生是否理解了对话内容。为了更自然的模拟开放领域对话，我们进一步将听力题中额外的问题转化为对话中的回复（图1右）。标注者截选原对话中具备回答问题信息的片段，根据正确选项构造正确的回复（图1右回复 A），根据两个错误选项构造两个错误的回复（回复 C 和回复 D）。

为了进一步提升难度，引入额外的推理信息，标注者还需根据正确选项构建一个负面的回复（回复 B）。另外，标注者需要保证在无上文信息情况下，所有候选回复在逻辑上皆合理。这样可以让数据集聚焦于检测模型在多轮对话中的推理能力，而非判断单个句子是否具有逻辑性。作者还在标注过程中控制正确和错误的回复与上文的词汇重叠率相似，防止模型可以通过简单的根据文本匹配选出候选回复。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201110091857685.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
上图是MuTual的详细统计汇总，直观上感觉词汇量比其他数据集要小得多，是由于MuTual是从英语作为外语的听力测试中修改而来的，因此形态和语法的复杂性比其他数据集要简单得多。

为了评估不同推理类型的分布，我们注释了所涉及的特定推理类型，例如，从测试集中取样并将其分为六类。MuTual 数据集主要包含聊天机器人需要的六种推理能力：态度推理(13%)、数值推理(7%)、意图预测(31%)、多事实推理(24%)和常识等其他推理类型（9%）。

+ 态度推理（Attitude Reasoning）：这种类型的实例测试模型是否知道说话者对物体的态度。
+ 数值推理（Algebraic Reasoning）：这种类型的实例测试模型在选择回复时是否具备数值推理能力
+ 意图预测（Intention Prediction）：此类型测试模型是否可以预测说话者接下来要做什么
+ 多事实推理（Situational Reasoning）：在这种类型的实例中考虑情况信息（例如，位置，两个讲话者之间的关系），模型应从先前的上下文中挖掘隐式信息。
+ 常识等其他推理类型（Multi-fact Reasoning and Others）：在这种情况下，正确的响应与上下文中的多个事实有关，这要求模型深刻理解上下文，而不是简单地进行文本匹配。

这六种类型的推理被认为与真正的聊天机器人最相关。例如，如果机器知道用户的姿势，它可使聊天机器人提出个人建议。意图预测功能使聊天机器人在长时间的对话中能够更智能地做出回复。

如下图，所有的回复都与上下文相关，但其中只有一个是逻辑正确的。一些错误的回复在极端情况下可能是合理的，但正确的回复是最合适的。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201110092743942.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
在真实应用场景中，检索式对话模型无法检索所有可能的回复，如果没有检索到合适的回复，系统应具有给予安全回复（safe response）的能力。为了模拟这一场景，我们提出了 MuTual plus。对于每个实例，MuTual plus 随机替换掉 MuTual 中一个候选回复。如果正确回复被替换，安全回复即为新的正确回复。如果错误回复被替换，原正确回复仍为四个回复中最合适的。

这里说明一下论文中R@1、R@2、MRR等指标的含义
>数据集以 Recall@1（正确检索结果出现在检索结果第一位），Recall@2（正确检索结果出现在检索结果前两位），MRR（Mean Reciprocal Rank, 正确检索结果在检索结果中的排名的倒数）作为评价指标。
# 实验结果
我们将数据分为训练集，开发集和测试集，比例分别为80％，10％和10％。我们在拆分过程中打包了从同一会话构造的实例，以避免数据泄漏。
+ 不同方法在 MuTual 数据集上的表现对比。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201110094123186.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
我们发现，选择题方法的性能明显优于个人评分方法。一种可能的解释是，多项选择方法将候选回复同时考虑在内，因此他们可以区分safe response是否是最佳选择。相比之下，个人评分方法并不稳健，在训练阶段，safe response容易使其混淆。

+ 不同方法在 MuTual plus 数据集上的表现对比，实验调查了模型在训练语料库中从未见过的情况下是否能够很好地处理safe response。
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020111010335387.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)

根据表3，4的结果可以看到，之前的检索式对话模型在 MuTual 上，表现只比随机猜的情况好一点。不过预训练模型也不能取得很好的效果，甚至 RoBERTa 也只能达到71%的 Recall@1。然而未经培训的非母语者可以轻松达到94%。

下图，我们发现，不同类别的BERT-MC和RoBERTa-MC的趋势相似，RoBERTa-MC在态度推理和多事实推理方面明显优于BERT-MC
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201110103744714.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)

如下图，通过简单的减法步骤即可得出时间差（5:00 pm-6h = 11:00 am），但这对RoBERTa-MC来说是一个巨大的挑战。模型在不同上下文轮数数据的 R@1 对比。#T 表示上下文轮数。#Instances 表示实例的数量。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201110104418149.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
进一步研究发现，与其他多轮对话数据集不同的是，在 MuTual 中，模型表现不会随着对话轮数增加而变差，RoBERTa 在两轮和六轮以上的数据上 R@1 相似。这表示推理能力并不依赖复杂的对话历史。如下图：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201110104742960.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
在推理类型方面，模型在数值推理和意图推测中表现的较差。上图第一个例子中，时差运算只需简单的减法(5:00pm - 6h = 11:00am)，第二个例子需要推理出对话出现在租房场景中，然而对现有的深度学习模型来说依然十分困难。下图展示了前 n 轮对话被删除情况下模型表现显著下滑，证明了解决 MuTual 中的问题需要依赖多轮推理而不是单轮推理。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201110105031354.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
# 总结
MuTual 数据集，用于针对性地评测模型在多轮对话中的推理能力，该数据集有助于将来进行多轮对话推理问题的研究。