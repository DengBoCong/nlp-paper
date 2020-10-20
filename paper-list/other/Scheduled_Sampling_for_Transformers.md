
# 前言

> 标题：Scheduled Sampling for Transformers\
> 原文链接：[Link](https://arxiv.org/pdf/1906.07651.pdf)\
> Github：[NLP相关Paper笔记和实现](https://github.com/DengBoCong/nlp-paper)\
> 说明：阅读论文时进行相关思想、结构、优缺点，内容进行提炼和记录，论文和相关引用会标明出处，引用之处如有侵权，烦请告知删除。\
> 转载请注明：DengBoCong

# 介绍
Scheduled sampling(计划采样)是一种避免Exposure Bias的技术，它包括向模型喂入Teacher-Forcing的embeddings和训练时间上一步中的模型预测的混合，该技术已用于通过递归神经网络（RNN）改善模型性能。在Transformer模型中，与RNN不同，新单词的生成会涉及到到目前为止生成的完整句子，而不仅是最后一个单词，致使应用Scheduled sampling技术并非易事。文中提出了一些结构上的更改，以允许通过两次遍历解码策略将Scheduled sampling应用于Transformer架构。
> 由于训练和预测的时候decode行为的不一致， 导致预测单词（predict words）在训练和预测的时候是从不同的分布中推断出来的。而这种不一致导致训练模型和预测模型直接的Gap，就叫做 Exposure Bias。

本文的创新和贡献：
+ 提出了一种在Transformer模型中使用Scheduled sampling的新策略，即在训练阶段内经过decoder两次
+ 比较了使用模型代替标准目标时以模型预测为条件的几种方法
+ 在两个语言对的机器翻译任务中使用Transformer测试了Scheduled sampling，并获得了接近Teacher-Forcing基线的结果（某些模型的改进幅度高达1个BLEU点）。
+ 线性衰减，指数衰减和反sigmoid衰减

# 实现细节
众所周知，Transformer是一个 autoregressive模型，其中，每个单词的生成都取决于序列中所有先前的单词，而不仅是最后生成的单词。单词的顺序是通过将位置嵌入与相应的单词嵌入相加来实现的，而在解码器中使用位置屏蔽可确保每个单词的生成仅取决于序列中的前一个单词，而不取决于随后的单词。由于这些原因，想要将Scheduled sampling应用在Transformer中比较困难，所以需要对Transformer的结构进行一定的修改，更改后的结构图如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201020202136686.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
如果你熟悉Transformer你会发现，其实encoder和decoder的结构都没有变，只是多了一个decoder
## Transformer的Two-decoder
流程分为以下几步
+ 首次通过decoder，获取模型预测
+ 将标准序列与预测序列混合，对于序列中的每个位置，我们以给定的概率选择是使用标准token还是使用来自模型的预测
+ 将标准序列与预测序列的混合再次输入decoder，产生最终预测

注意：重要的是要提到两个decoder是相同的，并且共享相同的参数

## Embedding Mix
对于序列中的每个位置，第一遍解码器都会给出每个词汇词的得分。以下是使用模型预测时使用这些分数的几种方法
+ 完全不混合嵌入并通过模型预测传递argmax，即使用来自解码器的得分最高的词汇词嵌入。
+ 混合top-k嵌入，即使用得分最高的5个词汇词嵌入的加权平均值。
+ 通过将嵌入结合softmax与temperature进行传递，使用较高的temperature参数可使argmax更好地近似，公式如下：其中 $\bar{e}_{i-1}$ 是在当前位置使用的向量，通过所有词汇词的嵌入量之和，以及分数 $s_{i-1}$ 的softmax加权获得。
$$\bar{e}_{i-1}=\sum_ye(y)\frac{exp(as_{i-1}(y))}{\sum_{y^{'}}exp(as_{i-1}(y^{'}))}$$
+ 使用argmax的另一种方法是从softmax分布中采样嵌入，公式如下：其中，$U ∼ Uniform(0, 1)$，$G=-log(-logU)$
$$\bar{e}_{i-1}=\sum_ye(y)\frac{exp(a(s_{i-1}(y))+G_y)}{\sum_{y^{'}}exp(a(s_{i-1}(y^{'})+G_{y^{'}}))}$$
+ 通过嵌入的sparsemax

## 权重更新
基于第二个解码器遍历的输出来计算交叉熵损失。对于将所有词汇单词相加的情况（Softmax，Gumbel softmax，Sparsemax），尝试两种更新模型权重的方法，如下：
+ 仅根据目标与模型预测之间的混合，通过最终做出预测的解码器进行反向传播。
+ 通过第二遍以及第一遍解码器进行反向传播，从而预测模型输出

# 实验
实验在如下两个数据集中进行：
+ IWSLT 2017 German−English 
+ KFTT Japanese−English

使用字节对编码（BPE）进行联合分割，超参数如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201020211057568.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
本论文的数据使用线性衰减最适合：$t(i)=max\{\epsilon, k-ci\}$，其中，$0 \leq \epsilon <1$ 是模型中要使用的最小Teacher-Forcing概率，$k$ 和 $c$ 提供衰减的偏移量和斜率。此函数确定训练步骤 $i$ 的Teacher-Forcing比 $t$，即在序列中每个位置进行Teacher-Forcing的概率，实验结果如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201020211943651.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
仅使用模型预测的得分最高的单词的Scheduled sampling效果不佳，使用混合嵌入（top-k，softmax，Gumbel softmax或sparsemax）并且仅使用第二个解码器通过的反向传播模型，在验证集上的性能略好于基准。
# 总结
这篇论文阐述了在Transformer上使用Scheduled Sampling的思路，对于几种Scheduled策略也进行了实验，说明了效果，值得借鉴。总体来说，实现思路不是很复杂，不过中间的可控性不高，并且可能需要找到符合数据集的一种更佳方式，可能泛化上不是很好。