# 前言

> [CTRL: A Conditional Transformer Language Model For Controllable Generation](https://arxiv.org/pdf/1909.05858.pdf)
> nlp-paper：[NLP相关Paper笔记和代码复现](https://github.com/DengBoCong/nlp-paper)
> nlp-dialogue：[一个开源的全流程对话系统，更新中！](https://github.com/DengBoCong/nlp-dialogue)
> text-similarity：[文本相似度（匹配）计算模型包](https://github.com/DengBoCong/text-similarity)
> 说明：阅读原文时进行相关思想、结构、优缺点，内容进行提炼和记录，原文和相关引用会标明出处，引用之处如有侵权，烦请告知删除。
> 转载请注明：DengBoCong

2019年推出GPT-3，和它以往的风格一样，模型巨大参数超多，在生成文本能力上表现惊人，不过GPT模型生成的文本风格往往由模型自身决定（和训练语料有关，有着一定的随机性）。而本篇所要提的模型CTRL（Conditional Transformer Language Model），对标GPT-2，可以更好的控制文章的内容，生成有价值的文本，且无需进一步训练就可以解决特定领域的具体问题。CTRL模型的最大优势是在生成文本时可指定文章的类型，同一模型可以写作不同风格的文章，可以指定文章的领域、风格、主题、时间、实体，实体间的关系，以及任务相关的行为等等。模型使用的Control Code和sub-reddit data如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/089d5e587cda41fe95526ad9c6f45e6c.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQm9Db25nLURlbmc=,size_20,color_FFFFFF,t_70,g_se,x_16)

不同的角度，有不同的答案。换言之，CTRL关注了语料在不同场景中的不同含义。模型更符合实际应用的场景：使用者在同一时间，只可能生成某一特定类型，同时又希望单个模型支持生成各种类型的文章，CTRL可视为多任务学习。使用 CTRL，只要提供control code，control code可以是 URL、问题和语言，也可以组合（评论、评级和价值）以提供更细粒度的控制，如下图：

# 模型细节
CTRL底层同样也基于Transformer，使用了其中Encoder部分，模型底层改动不大。之前的模型是根据词序列中的前 $n-1$ 个词计算下一个词 $n$ 是哪个词的可能性。例如给定序列为  $x=(x_i,...,x_n)$，数据集 $D=\{x^1,...,x^{|D|}\}$，参数 $\theta$， 即 
$$p(x)=\prod_{n}^{i=1}p(x_i|x<i)$$
$$L(D)=-\sum_{k=1}^{|D|}logp_\theta (x_i^k|x_{<i}^k)$$
CTRL又加入了条件 $c$，即文章的控制信息如类型，在计算概率的同时考虑条件 $c$。具体操作是在每一个序列的具体内容前加了入类型描述，使得在计算Attention过程中，类型与序列中的所有元素建立联系，即：
$$p(x|c)=\prod_{n}^{i=1}p(x_i|x<i,c)$$
$$L(D)=-\sum_{k=1}^{|D|}logp_\theta (x_i^k|x_{<i}^k,c^k)$$
说白了就是在预先考虑控制代码的基础上进行训练，在Encoder block中，还有一个小区别是对点击进行mask：
$$Attention(X,Y,Z)=softmax(\frac{mask(XY^T)}{\sqrt{d}})Z$$

模型的特点就是Controllable Generation，分为Sampling和Control Codes。
### Sampling
从语言模型生成文本时一般会用到 temperature-controlled stochastic sampling 方法，同时，每次生成 token 时在 top-k（而不是所有词表）中随机取，如下：
$$p_{i}=\frac{\exp \left(x_{i} / T\right)}{\sum_{j} \exp \left(x_{j} / T\right)}$$
+ T -> 0 近似贪婪分布，放大了峰值
+ T -> $\infty$ 使得分布更加平坦

其中，$k$ 是启发式的（自适应），$x_i$ 是每个 token 的 score，如果下个词的 confidence 比较高，$k$ 就小一些。在有多个非零的高概率候选 token 时，不采用模型，而是 “贪婪” 地选择下一个 token。对可能会产生的重复 token，文章提出一种新的 sample 方法，既能够近似贪婪 sampling，又能够对重复进行惩罚。惩罚的方法是对已产生的 tokens 进行打折（不在训练中使用），给定一列生成的 tokens g:
$$p_{i}=\frac{\exp \left(x_{i} /(T \cdot I(i \in g))\right.}{\sum_{j} \exp \left(x_{j} /(T \cdot I(j \in g))\right.} \quad I(c)=\theta \text { if } c \text { is True else } 1$$
其中，$\theta\approx 1.2$能够取得不错的平衡。

### Control Codes
+ Style by domain：Wiki，Books，Reviews，Horror，Relationships，Legal
+ More complex control codes：Science Title, Politics Title, Running Text, Horror Text, Reviews Rating；不同的 Link 代表不同的特征（domain, subdomain, entities, entity relations, and even dates）
+ Triggering specific tasks：问答、翻译
+ Zero-shot code-mixing

# 总结
CTRL不仅是一个自然语言处理问题的解决方案，同样也可应用到其它的序列处理问题之中。使用 control code 控制文本生成，控制代码可以是主题、实体、关系、特定任务等等。其实它的本质与之前的 Bert 类似：多任务 + 语言模型；这里的多任务可以看作是一个多分类任务。不过本文的切入角度是 “控制文本生成”，虽然是以类别标签的方式，但不得不说这是一个不错的创新点。