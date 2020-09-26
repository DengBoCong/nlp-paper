
<font color=#999AAA >提示：阅读论文时进行相关思想、结构、优缺点，内容进行提炼和记录，论文和相关引用会标明出处。</font>

# 前言

> 标题：Attention Is All You Need\
> 原文链接：[Link](https://arxiv.org/pdf/1706.03762.pdf)\
> 转载请注明：DengBoCong

# Abstract
序列转导模型基于复杂的递归或卷积神经网络，包括编码器和解码器，表现最佳的模型还通过注意力机制连接编码器和解码器。我们提出了一种新的简单网络架构，即Transformer，它完全基于注意力机制，完全消除了重复和卷积。在两个机器翻译任务上进行的实验表明，这些模型在质量上具有优势，同时具有更高的可并行性，并且所需的训练时间大大减少。我们的模型在WMT 2014英语到德语的翻译任务上达到了28.4 BLEU，比包括集成学习在内的现有最佳结果提高了2 BLEU。在2014年WMT英语到法语翻译任务中，我们的模型在八个GPU上进行了3.5天的训练后，创造了新的单模型最新BLEU分数41.8，比文献中最好的模型的训练成本更小。我们展示了Transformer通过将其成功应用于具有大量训练数据和有限训练数据的英语解析，将其很好地概括了其他任务。
#  Introduction
在Transformer出现之前，RNN、LSTM、GRU等在序列模型和转导问题的方法中占据了稳固的地位，比如语言模型、机器翻译等，人们一直在努力扩大循环语言模型和编码器-解码器体系结构的界限。递归模型通常沿输入和输出序列的符号位置考虑计算。将位置与计算时间中的步骤对齐，它们根据先前的隐藏状态ht-1和位置t的输入生成一系列隐藏状态ht。这种固有的顺序性导致了没办法并行化进行训练，这在较长的序列长度上变得至关重要。最近的工作通过分解技巧和条件计算大大提高了计算效率，同时在后者的情况下还提高了模型性能，但是，顺序计算的基本约束仍然存在。注意力机制已成为各种任务中引人注目的序列建模和转导模型不可或缺的一部分，允许对依赖项进行建模，而无需考虑它们在输入或输出序列中的距离。在这项工作中，我们提出了一种Transformer，一种避免重复的模型体系结构，而是完全依赖于注意力机制来绘制输入和输出之间的全局依存关系。

# Background
减少顺序计算的目标也构成了扩展神经GPU，ByteNet和ConvS2S的基础，它们全部使用卷积神经网络作为基本构建块，并行计算所有输入和输出的隐藏表示。在这些模型中，关联来自两个任意输入或输出位置的信号所需的操作数在位置之间的距离中增加，对于ConvS2S线性增长，而对于ByteNet则对数增长，这使得学习远处位置之间的依存关系变得更加困难。在Transformer中，此操作被减少为恒定的操作次数，尽管以平均注意力加权位置为代价，导致有效分辨率降低，但是我们用多头注意力抵消了这种代价。

Self-attention（有时称为d intra-attention）是一种与单个序列的不同位置相关的注意力机制，目的是计算序列的表示形式。Self-attention已成功用于各种任务中，包括阅读理解，抽象摘要和学习与任务无关的句子表示。Transformer是第一个完全依靠Self-attention来计算其输入和输出表示的转导模型，而无需使用序列对齐的RNN或卷积。

# Model Architecture
Transformer依旧是遵循encoder-decoder结构，其模型的每一步都是自回归的，在生成下一个模型时，会将先前生成的符号用作附加输入。在此基础上，使用堆叠式Self-attention和point-wise，并在encoder和decoder中使用全连接层，结构图如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200917160031494.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
##  Encoder and Decoder Stacks
+ Encoder
   + 编码器由$N = 6$个相同层的堆栈组成，每层有两个子层，分别是Self-attention机制和位置完全连接的前馈网络
   + 每个子层周围都使用残差连接并进行归一化，也就是说每个子层的输出为$LayerNorm(x+Sublayer(x))$
   + 为了促进这些残差连连接，模型中的所有子层以及嵌入层均产生尺寸为dmodel = 512的输出

+ Decoder
   + 解码器还由N = 6个相同层的堆栈组成
   + 除了每个编码器层中的两个子层之外，解码器还插入一个第三子层，该子层对编码器堆栈的输出执行多头注意力
   + 对编码器堆栈的输出执行多头注意力时，要注意使用mask，保证预测只能依赖于小于当前位置的已知输出。
   + 每个子层周围都使用残差连接并进行归一化

## Attention
注意力方法可以描述为将query和一组key-value映射到输出，其中query，key，value和输出都是向量。输出是计算value的加权总和，其中分配给每个value的权重是通过query与相应key的方法来计算的。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200917162420181.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
### Scaled Dot-Product Attention
它的输入是$d_k$维的queries和keys组成，使用所有key和query做点积，并除以$\sqrt{d_k}$，然后应用softmax函数获得value的权重，公式如下：
$$Attention(Q,K,V)=softmax(\frac{QK^{T}}{\sqrt{d_k}})V$$
+ 常用注意力方法
   + 相加（在更大的$d_k$下，效果更好）
   + 点积（更快一些）
   + 所以为了在较大的$d_k$下，点积也能工作的好，在公式中才使用了$\frac{1}{\sqrt{d_k}}$

### Multi-Head Attention
多头注意力使模型可以共同关注来自不同位置的不同表示子空间的信息，最后取平均：
$$
MultiHead(Q,K,V)=Concat(head_1,...,head_h)W^{O}\\
head_1=Attention(QW_{i}^{Q},K_{i}^{K},V_{i}^{V})
$$
论文中使用$h=8$注意力层，其中$d_k=d_v=\frac{d_{model}}{h}=64$
### Applications of Attention in our Model
Transformer以三种不同方式使用多头注意力:
+ 在“encoder-decoder注意”层中，queries来自先前的decoder层，而keys和values来自encoder的输出，这允许解码器中的每个位置都参与输入序列中的所有位置。
+ encoder包含self-attention层。 在 self-attention层中，所有key，value和query都来自同一位置，在这种情况下，是编码器中前一层的输出。
+ 类似地，decoder中的self-attention层允许decoder中的每个位置都参与decoder中直至并包括该位置的所有位置。我们需要阻止decoder中的向左信息流，以保留自回归属性。

## Position-wise Feed-Forward Networks
除了关注子层之外，我们的encoder和decoder中的每个层还包含一个完全连接的前馈网络，该网络分别应用于每个位置。 这由两个线性变换组成，两个线性变换之间有ReLU激活。
$$
FNN(x)=max(0,xW_1+b_1)W_2+b_2
$$
虽然线性变换在不同位置上相同，但是它们使用不同的参数
## Embeddings and Softmax
与其他序列转导模型类似，使用学习的嵌入将输入标记和输出标记转换为维dmodel的向量。我们还使用通常学习的线性变换和softmax函数将解码器输出转换为预测的下一个token概率
## Positional Encoding
位置编码的维数dmodel与嵌入的维数相同，因此可以将两者相加，位置编码有很多选择，可以学习和固定。在这项工作中，我们使用不同频率的正弦和余弦函数，其中pos是位置，i是维度。
$$
PE_{(pos,2i)}=sin(pos/10000^{2i/d_{model}})
PE_{(pos,2i+1)}=sin(pos/10000^{2i/d_{model}})
$$
也就是说，位置编码的每个维度对应于一个正弦曲线，波长形成从2π到10000·2π的几何级数。当然还有其他的方法，不过选择正弦曲线版本是因为它可以使模型外推到比训练期间遇到的序列长的序列长度

# Why Self-Attention
考虑一下三点：
+ 每层的总计算复杂度
+ 可以并行化的计算量，以所需的最少顺序操作数衡量
+ 网络中远程依赖关系之间的路径长度，在许多序列转导任务中，学习远程依赖性是一项关键挑战。影响学习这种依赖性的能力的一个关键因素是网络中前向和后向信号必须经过的路径长度。输入和输出序列中位置的任意组合之间的这些路径越短，学习远程依赖关系就越容易

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200917175632580.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
作为附带的好处，自我关注可以产生更多可解释的模型
# Training
##  Training Data and Batching
我们对标准WMT 2014英语-德语数据集进行了培训，该数据集包含约450万个句子对。句子是使用字节对编码的，字节对编码具有大约37000个token的共享源目标词汇。
## Hardware and Schedule
大型模型接受了300,000步（3.5天）的训练。
## Optimizer
我们使用Adam优化器，其中β1= 0.9，β2= 0.98和$\xi $= 10-9。 根据公式，我们在训练过程中改变了学习率：
$$lrate=d_{model}^{-0.5}\cdot min(step\_num^{-0.5},step\_num\cdot warmup\_steps^{-1.5})$$
这对应于第一个warmup_steps训练步骤的线性增加学习率，此后与步骤数的平方根的平方成反比地降低学习率，我们使用的warmup_steps=4000。
## Regularization
+ Residual Dropout
+ Label Smoothing

#  Results
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200917203338187.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
# Conclusion
在这项工作中，我们介绍了Transformer，这是完全基于注意力的第一个序列转导模型，用多头自注意力代替了编码器-解码器体系结构中最常用的循环层。对于翻译任务，与基于循环层或卷积层的体系结构相比，可以比在体系结构上更快地训练Transformer。 在WMT 2014英语到德语和WMT 2014英语到法语的翻译任务中，我们都达到了最新水平。 在前一项任务中，我们最好的模型甚至胜过所有先前报告。我们对基于注意力的模型的未来感到兴奋，并计划将其应用于其他任务。 我们计划将Transformer扩展到涉及文本以外的涉及输入和输出形式的问题，并研究局部受限的注意机制，以有效处理大型输入和输出，例如图像，音频和视频。 使生成减少连续性是我们的另一个研究目标。