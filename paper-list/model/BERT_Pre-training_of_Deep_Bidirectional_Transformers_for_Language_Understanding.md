# 前言

> 标题：BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding\
> 原文链接：[Link](https://arxiv.org/pdf/1810.04805.pdf)\
> Github：[NLP相关Paper笔记和实现](https://github.com/DengBoCong/nlp-paper)\
> 说明：阅读论文时进行相关思想、结构、优缺点，内容进行提炼和记录，论文和相关引用会标明出处，引用之处如有侵权，烦请告知删除。\
> 转载请注明：DengBoCong

# 介绍
本文介绍的language representation model就是大名鼎鼎的BERT，其模型结构是利用Transformer的双向Encoder表示。BERT有个明显的特点就是，它通过在所有层的左侧和右侧上下文中共同进行条件预处理，从而在未标记的文本中预训练深层双向表示。

现有两种将预训练语言表示应用于下游任务的策略：**feature-based**和**fine-tuning**，基于特征的方法，比较典型的就是ELMo，使用特定于任务的架构，其中包括预训练的表示形式作为附加功能。微调的方式，众所周知的是GPT了，这种方式通过引入最少的特定于任务的参数，并通过简单地微调所有预训练的参数来对下游任务进行训练。两种方法在预训练过程中具有相同的目标功能，它们使用**单向语言模型**学习通用语言表示形式。而BERT作为预训练模型，就是定位于通过微调的方式，不过和其他模型有个区别就是（前面我加粗字体），BERT是双向结合上下文的架构，论文里面说双向有利于句子级任务。


BERT的特别之处：
+ BERT受到完型填空任务的启发，通过使用一个“masked language model”(MLM)预训练目标来减轻上面提到的单向约束问题。MLM随机masks掉input中的一些tokens，目标是从这些tokens的上下文中预测出它们在原始词汇表中的id。与从左到右的语言模型预训练不同，MLM目标使表示形式能够融合左右上下文，这使我们可以预训练深度双向Transformer。
+ 除了MLM，作者还使用了一个“next sentence prediction”任务，连带的预训练text-pair表征

# 背景知识
+ 深度双向：深度双向和浅度双向的区别在于，后者仅仅是将分开训练好的left-to-right和right-to-left的表征简单的串联，而前者是一起训练得到的。
+ feature-based: 又称feature-extraction 特征提取。就是用预训练好的网络在新样本上提取出相关的特征，然后将这些特征输入一个新的分类器，从头开始训练的过程。也就是说在训练的过程中，网络的特征提取层是被冻结的，只有后面的密集链接分类器部分是可以参与训练的。
+ fine-tuning: 微调。和feature-based的区别是，训练好新的分类器后，还要解冻特征提取层的顶部的几层，然后和分类器再次进行联合训练。之所以称为微调，就是因为在预训练好的参数上进行训练更新的参数，比预训练好的参数的变化相对小，这个相对是指相对于不采用预训练模型参数来初始化下游任务的模型参数的情况。也有一种情况，如果你有大量的数据样本可以训练，那么就可以解冻所有的特征提取层，全部的参数都参与训练，但由于是基于预训练的模型参数，所以仍然比随机初始化的方式训练全部的参数要快的多。对于作者团队使用BERT模型在下游任务的微调时，就采用了解冻所有层，微调所有参数的方法。
+ warmup:学习率热身。规定前多少个热身步骤内，对学习率采取逐步递增的过程。热身步骤之后，会对学习率采用衰减策略。这样训练初期可以避免震荡，后期可以让loss降得更小。
# 相关工作
+ **无监督的基于特征的方法**：ELMo和它的前身从不同的维度概括了传统的词嵌入研究。它们从left-to-right和right-to-left语言模型中提取上下文敏感的特征。每个token(单词、符号等)的上下文表征是通过**串联left-to-right和right-to-left**的表征得到的。Melamud等人在2016年提出了使用LSTMs模型通过一个预测单词左右上下文的任务来学习上下文表征。与ELMo类似，他们的模型也是基于feature-based方法，并且没有深度双向
+ **无监督的微调方法**：与feature-based方法一样，该方向刚开始只是在未标记的文本上预训练词嵌入参数(无监督学习)。最近，句子和文档等生成上下文token表征的编码器已经从未标记的文本中预训练出来，并且通过fine-tuned的方式用在下游任务中。
+ **监督数据的迁移学习**：也有工作展示了从大数据集的监督任务的做迁移学习的有效性，就像自然语言推理(NLI)，和机器翻译。

# 具体实现结构
框架分为两个步骤：
+ pre-training
+ fine-tuning

在预训练期间，BERT模型在不同任务的未标记数据上进行训练。微调的时候，BERT模型用预训练好的参数进行初始化，并且是基于下游任务的有标签的数据来训练的。问答领域的例子作为本节一个运行示例，如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020102911482744.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
BERT的一个特性是其跨不同任务的统一体系结构，即在预训练架构和下游的架构之间的差异很小。BERT是基于原始Transformer实现中的Encoder（Transformer的那篇文章我也有阅读笔记，分别在[CSDN](https://dengbocong.blog.csdn.net/article/details/108639979)和[知乎](https://zhuanlan.zhihu.com/p/250946855)发布），改造成多层双向Transformer Encoder。**BERT Transformer使用双向self-attention，而GPT Transformer 使用带约束的self-attention，每个token只能注意到它左边的上下文。**
> 本文中，$L$表示层数，$H$表示每个隐藏单元的维数大小，$A$表示self-attention头数。BERT有2种参数，分别是BERT($base$，$L=12$, $H=768$, $A=12$, $Total Parameters=110M$)和BERT($large$，$L=24$, $H=1024$, $A=16$, $Total Parameters=340M$)。

使用BERT做各种下游任务，输入表征可以在一个token序列里清楚的表示一个句子或者一对句子(比如<Question,Answer>)。在整个工作中，“句子”可以是任意连续的文本范围，而不是实际的语言句子。论文中BERT关于输入输出的操作：
+ 使用大小为30000的WordPiece作为词嵌入层
+ 句子对打包成一个序列
+ 每个序列的首个token总是一个特定的classification token([CLS])，这个token对应的最后的隐藏状态被用作分类任务的聚合序列表征。
+ 区分句子对中句子的方法有两种：
   + 通过分隔符[SEP]
   + 模型架构中添加了一个经过学习的嵌入(learned embedding)到每个token，以表示它是属于句子A或者句子B。

如上面图1中，$E$表示输入的词嵌入，$C$表示最后隐藏层的[CLS]的向量，$T_i$ 表示第 $i$ 个输入token在最后隐藏层的向量。对于给定的token，其输入表示是通过将相应的token，segment和position embeddings求和而构造的，如下图。
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020102913010227.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
### Pre-training BERT
+  **Masked LM（MLM--掩盖式语言模型--Cloze任务）**：本文作者直觉上认为，深层双向模型比left-to-right模型或left-to-right模型和right-to-left模型的浅层连接更强大。标准条件语言模型只能从左到右或从右到左进行训练，因为双向条件将使模型“间接的看到自己”。为了训练深度双向表示，做法是简单的随机mask一些百分比的输入tokens，然后预测那些被mask掉的tokens。在实验中，作者为每个序列随机mask掉了15%的WordPiece tokens。注意了，**BERT只预测被mask掉的词，而不是重建完整的输入**。但由于[MASK]符号作为token不会出现在微调阶段，所以要想办法让那些被mask掉的词的原本的表征也被模型学习到，所以这里需要采用一些策略，如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201029193110597.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
+ **Next Sentence Prediction（NSP）**：许多下游任务，比如问答，自然语言推理等，需要基于对两个句子之间的关系的理解，而这种关系不能直接通过语言建模来获取到。为了训练一个可以理解句子间关系的模型，做法是为一个二分类的下一个句子预测任务进行了预训练。特别是，当为每个预测样例选择一个句子对A和B，50%的时间B是A后面的下一个句子(标记为IsNext)， 50%的时间B是语料库中的一个随机句子(标记为NotNext)。在图1中，C用来预测下一个句子（NSP）。尽管简单，但是该方法QA和NLI任务都非常有帮助。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201029210744426.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)

+ **Pre-training data**：预训练语料使用了BooksCorpus(800M words)，English Wikipedia(2500M words) 。之所以使用这种文档级语料而不是使用语句级语料，就是为了提取长连续序列。

### Fine-tuning BERT
对于涉及到文本对的应用，常见的模式是先分辨编码文本对中的文本，然后应用双向交叉的注意力。BERT使用self-attention机制统一了这两个步骤，BERT使用self-attention编码一个串联的文本对，其过程中就包含了2个句子之间的双向交叉注意力。
+ 输入端：句子A和句子B可以是：（1）释义句子对（2）假设条件句子对（3）问答句子对 （4）文本分类或序列标注中的text-∅对。
+ 输出端：token表征喂给一个针对token级别的任务的输出层，序列标注和问答是类似的，而[CLS]表征喂给一个分类器输出层，比如情感分析。

BERT在不同任务上微调的图解
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201029195812498.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
# 实验结果
### GLUE
在所有的GLUE任务上，作者使用了$batch-size=32,epochs=3$。对于每个任务，都通过开发集的验证来选择了最佳的微调学习率(在$5e- 5$，$4e - 5$，$3e -5$和$2e-5$之间)。另外，对于BERT的large模型，作者发现微调有时候在小数据集上不稳定，所以随机重启了几次，并选择了开发集上表现最佳的模型。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201029201521980.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
+ 在模型架构方面，除了注意力掩盖之外，BERT base版本的模型架构和OpenAI GPT几乎相同。
+ BERT large 版本明显比base版本要表现的更好。

### SQuAD v1.1
标准的SQuAD v1.1是一个100k的问答对集合，给定一个问题和一篇短文，以及对应的答案，任务是预测出短文中的答案文本span。图1所示，在问答任务中，作者将输入问题和短文表示成一个序列，其中，使用A嵌入表示问题，B嵌入表示短文。在微调的时候，作者引入一个start向量S，和一个end向量E，维数都为H。
+ answer span的起始词 $i$ 的概率计算公式（答案末尾词的概率表示原理一样。）：$P_i=\frac{e^{S\cdot T_i}}{\sum_je^{S\cdot T_j}}$。
+ 位置 $i$ 到位置 $j$ 的候选span的分数定义如下：$S\cdot T_i+E\cdot T_j$。

并将满足 $j>i$ 的最大得分的span最为预测结果。训练目标是正确的开始和结束位置的对数似然估计的和。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201029204255261.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
### SQuAD 2.0
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201029204512621.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)

### SWAG
The Situations With Adversarial Generations (SWAG)数据集包含113k个句子对完整示例，用于评估基于常识的推理。给定一个句子，任务是从四个选项中选择出最有可能是对的的continuation(延续/扩展)。在微调的时候，作者构造了4个输入序列，每个包含给定句子A的序列和continuation(句子B)。引入的唯一特定于任务的参数是一个向量，它与[CLS]token做点积，得到每个选项的分数，该分数会通过一个softmax层来归一化。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201029204522481.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
# Ablation Studies（消融研究）
简单而言就是通过控制变量法证明算法的有效性。
### 预训练任务的影响
通过去掉NSP后，对比BERT的双向表征和Left-to-Right表征，作者得证明了有NSP更好，且双向表征更有效。通过引入一个双向的LSTM，作者证明了BILSTM比Left-to-Right能得到更好的结果，但是仍然没有BERT的base版本效果好，如下图：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201029205352261.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
ELMo是分别训练LTR和RTL的方式，以下列举了其不如BERT的地方：
+ ELMo这种分别训练在串联的方式比单个双向模型代价高的两倍
+ 对于QA这样的任务而言不直观，因为RTL模型将无法确定问题的答案
+ 绝对不如深度双向模型强大，因为深度双向模型可以在每一层使用左右上下文

### 模型大小的影响
论文中训练了一些不同层数、隐藏单元数、注意力头的BERT模型，但使用相同的超参数和训练过程。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201029205942994.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
### 训练步数的影响
从如下图总结出，BERT是需要巨大的预训练量级(128,000 words/batch * 1000,000 steps)进行训练的。还有就是虽然MLM预训练收敛速度比LTR慢（因为每个batch中只有15%的单词被预测，而不是所有单词都参与），但是准确度超过了LTR模型。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201029211329960.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
### 不同Masking过程的影响
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201029211959996.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)

### BERT用于feature-based方法
相对于fine-tuning的方式而言，feature-based的方式也有着其关键的优势。首先，不是所有的任务都可以轻易的表示成Trasformer encoder 架构，所以会有需要添加一个基于特定任务的模型架构的需求。其次，预先计算一次训练数据的昂贵表示，然后在此表示之上使用更便宜的模型运行许多实验，这对计算有很大的好处。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201029210454454.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
# 总结
BERT的大名想必早就知道了，了解BERT相关的结构以及威力还有有必要的，必备不时之需。BERT和GPT相反，使用的Transformer的Encoder（可见Transformer有多牛逼）。BERT的主要贡献是进一步将这些发现推广到深层双向架构，使得相同的预训练模型可以成功应对一组广泛的NLP任务。附录的相关信息我也直接穿插在上面各节的论述中，然后最后补充一下一个BERT,ELMo,OpenAI GPT模型架构对比图
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201029210915719.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
+ BERT使用了双向的Transformer架构
+ OpenAI GPT使用了left-to-right的Transformer
+ ELMo分别使用了left-to-right和right-to-left进行独立训练，然后将输出拼接起来，为下游任务提供序列特征
上面的三个模型架构中，只有BERT模型的表征在每一层都联合考虑到了左边和右边的上下文信息。

除了架构不同，另外的区别在于BERT和OpenAI GPT是基于fine-tuning的方法，而ELMo是基于feature-based的方法。