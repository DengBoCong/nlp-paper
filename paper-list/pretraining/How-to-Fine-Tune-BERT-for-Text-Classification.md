# 前言

> 标题：How to Fine-Tune BERT for Text Classification?
> 原文链接：[Link](https://arxiv.org/pdf/1905.05583.pdf)
> nlp-paper：[NLP相关Paper笔记和代码复现](https://github.com/DengBoCong/nlp-paper)
> nlp-dialogue：[一个开源的全流程对话系统，更新中！](https://github.com/DengBoCong/nlp-dialogue)
> 说明：阅读原文时进行相关思想、结构、优缺点，内容进行提炼和记录，原文和相关引用会标明出处，引用之处如有侵权，烦请告知删除。
> 转载请注明：DengBoCong

预训练语言模型很强，通过微调可以给你的任务模型带来明显的提升，但是针对具体的任务如何进行微调使用，就涉及到了考经验积累的tricks，最近在打文本相关的比赛，正好用预训练模型为基础构建下游任务模型，所以着重的关注一些相关的BERT微调tricks，凑巧看到这篇文章，里面专门介绍 BERT 用于中文文本分类的各种 tricks，所以在此分享一下。这篇文章分别介绍了Fine-Tuning Strategies、Further Pre-training和Multi-Task Fine-Tuning，具体见后文总结介绍。
关于预训练语言模型，可以看论文团队的另一篇文章更新的文章：Pre-trained Models for Natural Language Processing: A Survey（[论文阅读笔记：超详细的NLP预训练语言模型总结清单！](https://zhuanlan.zhihu.com/p/352152573)）

# 前情提要
首先先确定一下BERT在Text Classification上的一般应用，我们都知道BERT喂入的输入有两个特殊的Token，即[CLS]置于开头，[SEP]用于分隔句子，最后的输出取[CLS]的最后隐藏层状态 $h$ 作为整个序列的表示，然后使用全连接层映射到分类任务上，及：
$$p(c|h)=softmax(Wh)$$
基于此，论文分别讨论通用微调BERT的方法流程，Fine-Tuning Strategies、Further Pre-training和Multi-Task Fine-Tuning，如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210319222831458.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
论文分析结果用的实验数据共八个，如下，可以归纳为Sentiment analysis、Question classification、Topic classification、Data preprocessing
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210319232645576.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)

> Warmup是在ResNet论文中提到的一种学习率预热的方法，它在训练开始的时候先选择使用一个较小的学习率，训练了一些epoches或者steps

# Fine-Tuning策略
我们来带着如下几个问题进行思考：
+ BERT的不同层对语义句法信息有着不同的抽取能力，那么那一层更有利于目标任务？
+ 如何选择优化算法和学习率

想要微调BERT适配目标任务，主要有三个因素（和上面思考相匹配）：
+ BERT最大处理序列长度为512
+ BERT-base有12层，需要挑选合适的层用于目标分类任务
+ 过拟合

超参：

```
batch_size = 24; dropout = 0.1; learning-rate=2e-5; warm-up proportion = 0.1; max_epoch = 4;
```

### BERT最大处理序列长度为512
针对长度超过512的文本，实验如下三种转换策略（ 预留[CLS] 和 [SEP]）：
+ head-only：前510 tokens
+ tail-only：后510 tokens;
+ head+tail：根据经验选择前 128 和后 382 tokens
+ 分段：首先将输入文本（长度为L）分成k = L/510个小段落，将它们依次输入BERT得到k个文本段落的表示。每个段落的representation是最后一层[CLS]的hidden state，并分别使用mean pooling, max pooling and self-attention来合并所有段落的representation。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210319235425953.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)

实验证实第三种在错误率上有 0.15~0.20% 的优势，也许是因为重要信息在首尾比较多中间比较少？
### BERT-base有12层，需要挑选合适的层用于目标分类任务
下图是不同层在任务上的测试错误率结果：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210319235624285.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
很明显，用最后一层还是比较靠谱的。


### 过拟合-学习率逐层衰减
针对学习率的衰减策略，本文继承了 [ULM-Fit](https://arxiv.org/pdf/1801.06146.pdf)）中的方案，叫作 Slanted Triangular，这个方案和 BERT 的原版方案类似，都是带 warmup 的先增后减。通常来说，这类方案对初始学习率的设置并不敏感，但是，在 Fine-Tune阶段使用过大的学习率，会打乱 Pre-train 阶段学习到的句子信息，造成“灾难性遗忘”。这里简单描述一下学习率策略：我们将BERT $L$ 层的参数分别表示为 $\{\theta^1,...,\theta^L\}$，参数更新策略如下：
$$\theta^l_t=\theta^l_{t-1}-\eta^l\cdot \triangledown_{\theta^l}J(\theta)$$
其中，$\eta^l$ 表示第 $l$ 层的学习率，并设 $\eta^L$ 为初始学习率，而 $\eta^{k-1}=\xi\cdot\eta^k$，$\xi$ 就是衰减因子，小于或等于 $1$，如何等于 $1$ 那么就是我们所熟悉的SGD了。关于SGD及其扩展的优化算法可以参考我这篇文章：[论文阅读笔记：各种Optimizer梯度下降优化算法回顾和总结](https://zhuanlan.zhihu.com/p/343564175)

这种设置相邻两层的学习率比例，让低层更新幅度更小的方法，系数控制在 0.90 ~ 0.95，整体优化效果不明显，不建议尝试，实验结果如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210320000024861.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210320000253814.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)

上图显示最右边的 4e-4 已经完全无法收敛了，而 1e-4 的 loss 曲线明显不如 2e-5 和 5e-5 的低。根据作者的数据集大小推测，整个 finetune 过程在 1W 步左右，**实测发现，小数据上更多的 epoch 并不能带来效果的提升，不如早一点终止训练省时间**。


# Further Pre-training
+ BERT在通用域数据下进行预训练，其数据分布与目标域不同，所以可以考虑进一步使用目标域数据对BERT进行预训练，先对基线做进一步的 pretrain，能够帮助 finetune 效果提升。而Further Pre-training有几种方案：
+ 使用本任务的训练数据进行预训练
+ 使用有着类似数据分布的相关任务数据进行预训练
+ 使用跨领域任务的数据进行预训练

超参：
```
batch_size = 32; max_length = 128; learning_rate = 5e-5; warmup_steps = 1W; steps = 10W;
```
### 使用本任务的训练数据进行Further Pre-training
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210320000741171.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
上图显示10W步之后达到最优结果，需要注意的是，pretrain 为了提升训练效率，使用的是偏短的 128 个词句子，学习率仍然是带 warmup 的衰减方式，初始值不用像 finetune 那样设置得那么小。整个训练过程为 10W 步，这个值是作者实验测定的，太短或太长都会影响最终模型的准确率。

### 同领域语料和跨领域语料进行Further Pre-training
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210320092603113.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)

上述得到结论，优先使用同领域的无标注数据，其次使用 finetune 的训练数据，什么都没有的话，混一点跨领域数据也可以。

# Multi-Task Fine-Tuning
+ 对目标与中的多个任务同时微调BERT，是否仍然对任务有帮助？

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210320093338603.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
在Multi-Task Fine-Tuning中，所有BERT和Embedding都进行共享，只留最后的分类层来适配不同的任务。由实验结果得出Multi-Task Fine-Tuning并不能给Text Classification相关子任务带来很大的提升。

# 总结
+ 预训练语言模型BERT的最后一层较于其他层来说更加有助于分类
+ 合适的学习率和层宽有助于BERT的catastrophic forgetting 
+ 使用任务语料或者同域语料进行further pre-training效果更好哦
+ 是在任务语料太少，混入跨领域数据也可，但是效果并不明显哦