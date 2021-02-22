# 前言

> 标题：Pre-trained Models for Natural Language Processing: A Survey\
> 原文链接：[Link](https://arxiv.org/pdf/2003.08271.pdf)\
> nlp-paper：[NLP相关Paper笔记和代码复现](https://github.com/DengBoCong/nlp-paper)\
> nlp-dialogue：[一个开源的全流程对话系统，更新中！](https://github.com/DengBoCong/nlp-dialogue)\
> 说明：阅读原文时进行相关思想、结构、优缺点，内容进行提炼和记录，原文和相关引用会标明出处，引用之处如有侵权，烦请告知删除。\
> 转载请注明：DengBoCong

预训练模型给下游任务带来的效果不言而喻，有了预训练模型，我们可以使用它来加速解决问题的过程。正如论文中所说的那样，**预训练模型（PTMs）的出现将自然语言处理（NLP）带入了一个新时代**。本篇论文基于分类从四个角度对现有PTMs进行系统分类，描述如何使PTMs的知识适应下游任务，然后概述了PTMs未来研究的一些潜在方向，通过本篇综述，来学习了解相关预训练模型。

# 背景

+ 第一代 PTMs 旨在学习词嵌入，由于下游的任务不再需要这些模型的帮助，因此为了计算效率，它们通常采用浅层模型，如 Skip-Gram 和 GloVe。尽管这些经过预训练的嵌入向量也可以捕捉单词的语义，但它们却不受上下文限制，只是简单地学习「共现词频」。这样的方法明显无法理解更高层次的文本概念，如句法结构、语义角色、指代等等。
+ 第二代 PTMs 专注于学习上下文的词嵌入，如 CoVe、ELMo、OpenAI GPT 以及 BERT。它们会学习更合理的词表征，这些表征囊括了词的上下文信息，可以用于问答系统、机器翻译等后续任务。另一层面，这些模型还提出了各种语言任务来训练 PTMs ，以便支持更广泛的应用，因此它们也可以称为预训练语言模型。

下图说明了NLP的通用神经体系结构，词嵌入有两种：Non-contextual Embeddings（非上下文嵌入）和Contextual Embeddings（上下文嵌入）。它们之间的区别在于，单词的嵌入是否根据出现的上下文而动态变化。
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021022212223824.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
之前 NLP 任务一般会预训练 $e$ 这些不包含上下文信息的词嵌入，我们会针对不同的任务确定不同的上下文信息编码方式，以构建特定的隐藏向量 $h$，从而进一步完成特定任务。但对于预训练语言模型来说，我们的输入也是 $e$ 这些嵌入向量，不同之处在于我们会在大规模语料库上预训练 Contextual Encoder，并期待它在各种情况下都能获得足够好的 $h$，从而直接完成各种 NLP 任务。换而言之，最近的一些 PTMs 将预训练编码的信息，提高了一个层级。


大多数神经上下文编码器可分为两类：sequence模型和graph-based模型，下图说明了这些模型的体系结构：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210222123920428.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
目前Transformer由于其强大的能力，成为了主流的PTMs结构，预训练的优势如下：
+ 在庞大的文本语料库上进行预训练可以学习通用的语言表示形式并帮助完成下游任务。
+ 预训练提供了更好的模型初始化，通常可以带来更好的泛化性能并加快目标任务的收敛速度。
+ 可以将预训练视为一种正则化，以避免对小数据过度拟合。

# PTMs概述
## Pre-training Tasks
+ Language Modeling (LM)
+ Masked Language Modeling (MLM)：被称为完形填空（Cloze）任务，即从输入序列中遮掩一些 token，然后训练模型来通过其余的 token 预测 masked 的 token。
   + Sequence-to-Sequence MLM (Seq2Seq MLM)
   + Enhanced Masked Language Modeling (E-MLM)
+  Permuted Language Modeling (PLM)：给定一个序列，然后从所有可能的排列中随机抽样一个排列。接着将排列序列中的一些 token 选定为目标，同时训练模型以根据其余 token 和目标的正常位置（natural position）来预测这些目标。
+ Denoising Autoencoder (DAE)：接受部分损坏的输入，并以恢复这些未失真的原始输入为目标。这类任务会使用标准 Transformer 等模型来重建原始文本，它与 MLM 的不同之处在于，DAE 会给输入额外加一些噪声。
+ Contrastive Learning (CTL)：相较于语言建模，CTL 的计算复杂度更低，因而在预训练中是理想的替代训练标准。
   + Deep InfoMax (DIM）
   + Replaced Token Detection (RTD）：替换 token 检测（Replaced Token Detection，RTD）与 NCE 相同，但前者会根据上下文语境来预测是否替换 token。
   + Next Sentence Prediction (NSP)：顾名思义，NSP 训练模型以区分两个输入句子是否为训练语料库中的连续片段。
   + Sentence Order Prediction (SOP)：SOP 使用同一文档中的两个连续片段作为正样本，而相同的两个连续片段互换顺序作为负样本。


下图是这些Pre-training Tasks的损失函数汇总：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210222155841685.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
## PTMs分类
+ Representation Type：预训练方法（PTMs）使用的词表征类型
+ Architectures：预训练方法使用的主干网络
+ Pre-Training Task Types：PTMs使用的 预训练任务类型
+ Extensions：为特定场景与输入类型所设计的PTMs

下图详细地展示了各种 PTMs的所属类别，只要看懂了它，基本目前现有的预训练语言模型的整体状态，都能了解了：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210222171337541.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
下表进一步展示了主流预训练方法的更多细节，主流模型、论文、实现，看这张表就足够了：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210222171424152.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
# PTMs扩展
## Knowledge-Enriched PTMs
PTMs通常从通用大型文本语料库中学习通用语言表示，但是缺少特定领域的知识，PTMs中设计一些辅助的预训练任务，将外部知识库中的领域知识整合到PTMs中被证明是有效的

+ ERNIE-THU将在知识图谱中预先训练的实体嵌入与文本中相应的实体提及相结合，以增强文本表示。由于语言表征的预训练过程和知识表征过程有很大的不同，会产生两个独立的向量空间。为解决上述问题，在有实体输入的位置，将实体向量和文本表示通过非线性变换进行融合，以融合词汇、句法和知识信息。
+ LIBERT（语言知识的BERT）通过附加的语言约束任务整合了语言知识。
+ SentiLR集成了每个单词的情感极性，以将MLM扩展到标签感知MLM（LA-MLM），ABSA任务上都达到SOTA。
+ SenseBERT不仅能够预测被mask的token，还能预测它们在给定语境下的实际含义。使用英语词汇数据库 WordNet 作为标注参照系统，预测单词在语境中的实际含义，显著提升词汇消歧能力。
+ KnowBERT与实体链接模型以端到端的方式合并实体表示。
+ KG-BERT显示输入三元组形式，采取两种方式进行预测：构建三元组识别和关系分类，共同优化知识嵌入和语言建模目标。这些工作通过实体嵌入注入知识图的结构信息。
+ K-BERT将从KG提取的相关三元组显式地注入句子中，以获得BERT的扩展树形输入。
+ K-Adapter通过针对不同的预训练任务独立地训练不同的适配器来注入多种知识，从而可以不断地注入知识，以解决注入多种知识时可能会出现灾难性遗忘问题。此外，这类PTMs还有WKLM、KEPLER等。

## Model Compression
由于预训练的语言模型通常包含至少数亿个参数，因此很难将它们部署在现实应用程序中的在线服务和资源受限的设备上，模型压缩是减小模型尺寸并提高计算效率的有效方法，论文中提到的5种PTMs的压缩方法为：
+ pruning（剪枝）：将模型中影响较小的部分舍弃，如Compressing BERT，还有结构化剪枝 LayerDrop ，其在训练时进行Dropout，预测时再剪掉Layer，不像知识蒸馏需要提前固定student模型的尺寸大小。
+ quantization（量化）：将高精度模型用低精度来表示，如Q-BERT和Q8BERT，量化通常需要兼容的硬件。
+ parameter sharing （参数共享）：相似模型单元间的参数共享。ALBERT主要是通过矩阵分解和跨层参数共享来做到对参数量的减少。
+ module replacing（模块替换）：BERT-of-Theseus根据伯努利分布进行采样，决定使用原始的大模型模块还是小模型，只使用task loss。
+ knowledge distillation （知识蒸馏）：通过一些优化目标从大型、知识丰富、fixed的teacher模型学习一个小型的student模型，蒸馏机制主要分为3种类型：
   + 从软标签蒸馏：DistilBERT、EnsembleBERT
   + 从其他知识蒸馏：TinyBERT、BERT-PKD、MobileBERT、 MiniLM、DualTrain
   + 蒸馏到其他结构：Distilled-BiLSTM

下表是一些代表性的压缩PTMs：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210222181417550.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
# 使PTMs适应下游任务
尽管PTMs可以从大型语料库中获取通用语言知识，但是如何有效地将其知识适应下游任务仍然是关键问题。迁移学习旨在使knowledge从源任务（或领域）适应目标任务（或领域），下图给出了迁移学习的示意图：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210222182203577.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
要将PTMs的知识转移到下游NLP任务，我们需要考虑以下问题：
+ 选择合适的预训练任务：语言模型是PTMs是最为流行的预训练任务，相同的预训练任务有其自身的偏置，并且对不同的任务会产生不同的效果。例如，NSP任务可以使诸如问答（QA）和自然语言推论（NLI）之类的下游任务受益。
+ 选择合适的模型架构：例如BERT采用的MLM策略和Transformer-Encoder结构，导致其不适合直接处理生成任务。
+ 选择合适的数据：下游任务的数据应该近似于PTMs的预训练任务，现在已有很多现成的PTMs可以方便地用于各种特定领域或特定语言的下游任务。
+ 选择合适的layers进行transfer：主要包括Embedding迁移、top layer迁移和all layer迁移。如word2vec和Glove可采用Embedding迁移，BERT可采用top layer迁移，Elmo可采用all layer迁移。

特征集成还是fine-tune？对于特征集成预训练参数是freeze的，而fine-tune是unfreeze的。特征集成方式却需要特定任务的体系结构，fine-tune方法通常比特征提取方法更为通用和方便，下表给出了适应性PTMs的一些常见组合：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210222182937367.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
 Fine-Tuning策略，即通过更好的微调策略进一步激发PTMs性能：
+ 两阶段 Fine-Tuning策略：如第一阶段对中间任务或语料进行 Fine-Tuning，第二阶段再对目标任务 Fine-Tuning。第一阶段通常可根据特定任务的数据继续进行Fine-Tuning预训练。
+ 多任务 Fine-Tuning：MTDNN在多任务学习框架下对BERT进行了 Fine-Tuning，这表明多任务学习和预训练是互补的技术。
+ 采取额外的适配器： Fine-Tuning的主要缺点是其参数效率低，每个下游任务都有自己的 Fine-Tuning参数。因此，更好的解决方案是在固定原始参数的同时，将一些可 Fine-Tuning的适配器注入PTMs。
+ 逐层阶段：逐渐冻结而不是同时对所有层进行 Fine-Tuning，也是一种有效的 Fine-Tuning策略。

# 关于PTMs的资源
下表提供了一些受欢迎的存储库，包括第三方实现，论文列表，可视化工具以及PTMs的其他相关资源：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210222183621568.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
# 总结
虽然 PTMs已经在很多 NLP 任务中显示出了他们强大的能力，然而由于语言的复杂性，仍存在诸多挑战，论文给出了五个未来 PTMs发展方向的建议。

+ **PTMs的上限**：目前，PTMs并没有达到其上限。大多数的PTMs可通过使用更长训练步长和更大数据集来提升其性能。目前NLP中的SOTA也可通过加深模型层数来更进一步提升。这将导致更加高昂的训练成本。因此，一个更加务实的方向是在现有的软硬件基础上，设计出更高效的模型结构、自监督预训练任务、优化器和训练技巧等。例如， ELECTRA 就是此方向上很好的一个解决方案。
+ **面向任务的预训练和模型压缩**：在实践中，不同的目标任务需要 PTMs拥有不同功能。而 PTMs与下游目标任务间的差异通常在于两方面：模型架构与数据分布。尽管较大的PTMs通常情况下会带来更好的性能表现，但在低计算资源下如何使用是一个实际问题。例如，对于 NLP 的 PTM 来说，对于模型压缩的研究只是个开始，Transformer 的全连接架构也使得模型压缩具有挑战性。
+ **PTMs的架构设计**：对于PTMs，Transformer 已经被证实是一个高效的架构。然而 Transformer 最大的局限在于其计算复杂度（输入序列长度的平方倍）。受限于 GPU 显存大小，目前大多数 PTMs无法处理超过 512 个 token 的序列长度。打破这一限制需要改进 Transformer 的结构设计，例如 Transformer-XL。关于Transformer结构变体可以参考这篇文章：[Transformer的9种变体概览](https://zhuanlan.zhihu.com/p/351742765)
+ **fine-tune中的知识迁移**：finetune是目前将 PTMs的知识转移至下游任务的主要方法，但效率却很低，每个下游任务都需要有特定的finetune参数。一个可以改进的解决方案是固定PTMs的原始参数，并为特定任务添加小型的finetune适配器，这样就可以使用共享的PTMs 服务于多个下游任务。
+ **PTMs 的解释性与可靠性**：PTMs 的可解释性与可靠性仍然需要从各个方面去探索，它能够帮助我们理解 PTM 的工作机制，为更好的使用及性能改进提供指引。