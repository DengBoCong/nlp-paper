# 前言

> 标题：Neural Speech Synthesis with Transformer Network\
> 原文链接：[Link](https://arxiv.org/pdf/1809.08895.pdf)\
> Github：[NLP相关Paper笔记和代码复现](https://github.com/DengBoCong/nlp-paper)\
> 说明：阅读论文时进行相关思想、结构、优缺点，内容进行提炼和记录，论文和相关引用会标明出处，引用之处如有侵权，烦请告知删除。\
> 转载请注明：DengBoCong

# 介绍
虽然像Tacotron2这样的TTS模型实现了最新的性能，但它们仍然存在两个问题：
+ 在训练和推理过程中效率低下（巨慢）
+ 难以使用当前的递归神经网络（RNN）对长期依赖性进行建模

本文受Transformer启发，使用多头自注意力机制取代Tacotron2中的RNN结构和原始注意力机制。借助多头自注意力机制，可以并行构造编码器和解码器中的隐藏状态，从而提高训练效率，同时，不同时间步的任意两个输入通过自注意力机制直接连接，有效解决了远程依赖问题。使用phoneme（音素）序列作为输入，Transformer TTS网络生成梅尔频谱图，然后通过WaveNet声码器以输出最终的音频结果。
> phoneme音素：能区分意义的最小声音单位，比如dog和fog中，d和f只要改变一个就改变了意义。

+ Tacotron2模型结构
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201205235515714.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
+ Transformer模型结构
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201206000024694.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70)


# 相关知识
这里对一些语音方面的相关概念进行说明，如果不感兴趣想要直接看论文内容，可以直接跳过这一节。
+ speech wave：语音波是一种compound wave，即包含各种频率的波。因此在频域上表示语音更为合适。
+ pitch音高：声音的尖锐程度，在频域中表现为频率的高低。
+ 基础频率F0：浊音中存在基础频率，而清音中不存在，F0决定了声音的音高。
+ formants共振峰：是一种元音特有的在频域中的现象，因为只有元音有基础频率。每个元音都有两个共振峰，可以用来区分元音，记为F1和F2。F1,F2取决于基础频率，如果基础频率太高，共振峰可能会消失，这种情况下就区分不出来元音，这种现象在各种女高音身上比较常见。
+ timbre音色：音色在广义上是指声音不同于其它的特点，在语音中不同的音节都有不同的特点，这可以通过频域观察出来，另外，特别地，对于元音我们可以通过共振峰来分辨音色。
+ noise：噪音、辅音(摩擦音)都会有broad spectrum，也就是说我们无法通过共振峰来识别它们。
+ envelope包络：在波的时域和频域图中，用来形容图形的整体形状的叫做包络。
+ frontend：主要是文字处理，使用NLP技术，从离散到离散，包括基本的分词、text normalization、POS以及特有的Pronunciation标注。
+ backend：根据前端结果生成语音，从离散到连续
+ segmentation & normalization：去噪、分句、分词以及把缩写、日期、时间、数字还有符号都换成可发音的词，这一步叫spell out。基本都基于规则。


# 模型结构
与基于RNN的模型相比，在神经TTS中使用Transformer具有两个优点：
+ 因为可以并行提供解码器输入序列的帧，它可以通过取代循环连接来进行并行训练。
+ 自注意力将整个序列的全局上下文注入到每个输入帧中，直接建立了长距离依赖关系

在本节中，将介绍Transformer TTS模型的体系结构，并分析每个部分的功能，整个模型结构图如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020120609460844.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
+ Text-to-Phoneme转换器：由于训练数据不足的情况下很难学习语言的所有规律性，并且某些例外情况很少出现，而无法通过神经网络学习。因此，作者建立了一个规则系统并将其实现为文本到音素的转换器，它可以覆盖绝大多数情况。
+ Scaled Positional编码：采用的位置编码是Transformer的正弦位置编码：
$$PE(pos,2i)=sin(\frac{pos}{10000^{\frac{2i}{d_{model}}}})$$    $$PE(pos,2i+1)=cos(\frac{pos}{10000^{\frac{2i}{d_{model}}}})$$
其中 $pos$ 是时间步长索引，$2i$ 和 $2i+1$ 是通道索引，$d_{model}$ 是每个帧的维数。不过有一点要注意的是，不像文本训练那样，source和target都是一个语言空间，embedding的编码是相似的。TTS中使用固定的位置嵌入可能会对编码器和解码器的pre-nets都施加严格的约束（将在后面描述），因此作者使用具有可训练权重的位置编码，以便这些位置编码可以自适应地匹配编码器和解码器pre-nets输出的比例，公式如下：
$$x_i=prenet(phoneme_i)+\alpha PE(i)$$
其中$\alpha$是可训练权重。
+ Encoder Pre-net：在Tacotron2中，将三层CNN应用于输入文本嵌入，它可以对输入字符序列中的上下文进行建模。在这里的Transformer TTS模型中，将 phoneme序列输入到同一网络中，这称为“encoder pre-net”。每个phoneme具有512维的可训练嵌入，每个卷积层的输出具有512个通道，然后进行batch normalization、ReLU激活以及dropout。此外，由于ReLU的输出范围是 $[0,\infty]$，而这些位置编码的每个维数都在 $[-1,1]$ 中，所以作者在最终ReLU激活后添加了线性层。在实验中证明，在Nonnegative Embeddings中添加以0为中心的位置信息将影响模型的性能。
+ Decoder Pre-net：梅尔频谱图首先被具有ReLU激活的，由两个全连接层（每个层都有256个隐藏单元）组成的神经网络处理，称为“decoder pre-net”，它在TTS系统中起着重要作用。phonemes具有可训练的嵌入，因此其子空间是自适应的，而梅尔频谱图的是固定的。**作者推断decoder pre-net负责将梅尔频谱图投影到与phonemes嵌入相同的子空间中，从而可以计算 $\left \langle phoneme, mel\ frame \right \rangle$ 对的相似性，从而使注意力机制发挥作用**。此外，还尝试了2个没有非线性激活的全连接层，但是无法生成合理的注意力矩阵来对齐编码器和解码器的隐藏状态。此外，**作者推测梅尔谱图具有一个紧凑且低维的子空间，其中256个隐藏单元足以映射**。同encoder pre-net一样，还添加了一个附加的线性层，不仅用于中心一致性，而且还获得与位置编码相同的尺寸。
+ Encoder：在Tacotron2中，编码器是双向RNN，而这里使用Transformer编码器代替它。与原始的双向RNN相比，多头注意力将注意力分散到几个子空间中，从而可以在多个不同方面建模帧关系，并直接建立任意两个帧之间的长依赖关系，因此每个帧都被视为整个序列的全局上下文。这对于合成音频韵律至关重要，尤其是在句子较长的情况下。
+ Decoder：在Tacotron2中，解码器是一个结合location sensitive attention的2层RNN，而这里使用Transformer解码器代替它。
+  Mel Linear、Stop Linear和Post-net：与Tacotron2相同，我们分别使用两个不同的线性层来预测梅尔频谱图和停止标记，并使用5层CNN产生残差来完善mel频谱图的重建。值得一提的是，对于停止标记的线性层而言，每个序列的末尾只有一个正样本，表示“停止”，而其他帧则是负样本，这种不平衡可能导致无法停止的推断。在计算二元交叉熵损失时，作者在停止标记的正样本上施加正权重 $(5.0\sim 8.0)$，从而有效地解决了这个问题。

# 实验结果
实验使用25小时的专业语音对测试Transformer TTS模型，并通过人工测试在MOS和CMOS中评估音频质量。**由于训练样本的长度相差很大，因此，如果以长样本为准扩大batch尺寸将占用很大内存，而如果以短样本为准缩小batch尺寸则会浪费并行计算能力，因此，作者使用动态batch大小，其中最大总的Mel光谱图帧数是固定的，并且一个batch应包含尽可能多的样本。**

Tacotron2使用字符序列作为输入，而本文的模型是根据pre-normalized phoneme序列训练的。自回归WaveNet包含2个QRNN层和20个扩张层，所有残差通道和扩张通道的大小均为 $256$。QRNN最终输出的每一帧均被复制200次，以具有与音频样本相同的空间分辨率且条件为20扩张层。

下表是MOS和CMOS指标的对比结果：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201206111801454.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
下图是模型生成的梅尔频谱的结果对比：如我们所见，论文模型在重建以红色矩形标记的细节方面做得更好，而Tacotron2在高频区域则省略了细节纹理。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201206112001432.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
下表显示了中心一致的位置编码效果稍好：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020120611221863.png#pic_center)

下图表明编码器和解码器的最终位置编码比例不同的对比：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201206114754487.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)



下表显示了具有可缩放比例的模型，其性能略有提高：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201206112337437.png#pic_center)

下面3张表是比较具有不同层数和头注意力数的性能和训练速度。发现减少层数和头注意力数均可以提高训练速度，但另一方面，会在不同程度上损害模型性能。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201206113930677.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201206113940804.png#pic_center)

# 总结
值得再次提及的是batch大小对于训练稳定性至关重要，并且更多的层可以完善生成的mel频谱图的细节，尤其是在高频区域，从而提高模型性能。论文作者对这一模型做了很多的实验，总的来说，训练时期的速度大大提高，加快了2到3倍，生成语音的质量也好于传统RNN结构模型（存疑，复现版本仅仅能做到效果相接近，可能是作者的调参技艺比较高超）。基于Transformer的TTS模型已是现在主流的End-to-End TTS系统的baseline，它的实现必不可少，而且因为Transformer本身优异的结构，也能大大加快实验的速度。