# 前言

> [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/pdf/1704.04368.pdf)
> nlp-paper：[NLP相关Paper笔记和代码复现](https://github.com/DengBoCong/nlp-paper)
> nlp-dialogue：[一个开源的全流程对话系统，更新中！](https://github.com/DengBoCong/nlp-dialogue)
> text-similarity：[文本相似度（匹配）计算模型包](https://github.com/DengBoCong/text-similarity)
> Algorithm：[对ML、DL和数据结构算法进行整理讲解，同时含C++/Java/Python三个代码版本](https://github.com/DengBoCong/Algorithm)s
> 说明：阅读原文时进行相关思想、结构、优缺点，内容进行提炼和记录，原文和相关引用会标明出处，引用之处如有侵权，烦请告知删除。
> 转载请注明：DengBoCong

最近看2021ACL的文章，碰到了Copying Mechanism和Coverage mechanism两种技巧，甚是感兴趣的翻阅了原文进行阅读，Copying Mechanism的模型CopyNet已经进行阅读并写了阅读笔记，如下：
[论文阅读笔记：Copying Mechanism缓解未登录词问题的模型--CopyNet](https://zhuanlan.zhihu.com/p/415502906)
而本篇文章则是讲Coverage mechanism，当然这篇并不是Coverage mechanism最初的技巧原文（最早出现在这一篇：[Statistical machine translation](http://mt-class.org/jhu/assets/papers/neural-network-models.pdf)），本篇只是将这个技巧进行改进使其更加适用于RNN-base的Seq2Seq模型。Copying Mechanism和Coverage mechanism两个技巧的提出都比较早，但是其应用得当，在特定任务上给模型带来效果提升会令人意想不到。

本论文主要围绕解决Seq2Seq模型应用于摘要生成时主要存在两个问题：
+ 难以准确复述原文的事实细节、无法处理原文中的未登录词(OOV)
+ 生成的摘要中存在重复的片段

对于OOV的问题，一种很自然的想法就是将source doc也纳入输出词的考虑范围，即可以直接从source doc中复制相关相应的token作为输出，这一点在CopyNet中应用的效果很不错。而对于重复词的问题，需要通过一种手段，利用之前所生成的token来影响当前time step的决策（可以认为是已出现的token概率进行惩罚），从而避免产生重复词，不过论文作者为了避免影响模型的效果，对不同的模型任务进行了改进，比如额外加上了coverage loss来将token位置也给考虑进去。

# 模型细节
![在这里插入图片描述](https://img-blog.csdnimg.cn/62e262c6edf34c4e826fc9268258d823.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQm9Db25nLURlbmc=,size_20,color_FFFFFF,t_70,g_se,x_16)

encoder部分采用一个单层双向LSTM，输入原文的词向量序列，输出一个编码后的隐层状态序列 $h_i$。decoder部分采用一个单层单向LSTM，每一步的输入是前一步预测的词的词向量，同时输出一个解码的状态序列 $s_t$，用于当前步的预测。attention具体的计算公式为：
$$e_i^t=v^Ttanh(W_hh_i+W_ss_t+b_{attn})$$
$$a_t=softmax(e_t)$$
其中$h_i,s_t$分别是source doc进行双向LSTM编码的hidden state和cell state，$W,b$则是参数。在计算出当前步的attention分布后，对encoder输出的隐层做加权平均，获得输入序列的动态表示，即context-vector：
$$h_t^*=\sum_ia_i^th_i$$
在不使用Copy Mechanism的情况下，我们的Seq2Seq是依靠decoder输出的隐层和context-vector，共同决定当前time step预测在词表上的概率分布：
$$P_{vocab}=softmax(V^{'}(V[s_t,h_t^*]+b)+b^{'})$$
#### Copying Mechanism
而论文则是在预测的每一个time step，通过动态计算一个生成概率 $p_{gen}$，巧妙的把seq2seq模型和pointer network结合起来，使得即保留了seq2seq模型保持抽象生成的能力，也保留了pointer network直接从原文中取词的Copy能力：
$$p_{gen}=\sigma(w_{h^*}^Th_t^*+w_s^Ts_t+w_x^T+b_{ptr})$$
$$P(W)=p_{gen}P_{vocab}(w)+(1-p_{gen})\sum_{i:w_i}a_i^t$$
其中，$\sigma$ 是sigmoid激活函数，这样就直接把seq2seq模型计算的attention分布作为pointer network的输出，源代码实现上通过参数复用，大大降低了模型的复杂度，如下：

```
with tf.variable_scope('calculate_pgen'):
p_gen = linear([context_vector, state.c, state.h, x], 1, True) # Tensor shape (batch_size, 1)
p_gen = tf.sigmoid(p_gen)
p_gens.append(p_gen)
```
#### Coverage mechanism
除此之外，针对重复词问题，论文使用Coverage mechanism，Coverage模型的重点在于预测过程中，维护一个coverage vector：
$$c^t=\sum_{t^{'}=0}^{t-1}a^{t^{'}}$$
这个向量是过去所有预测步计算的attention分布的累加和，记录着模型已经关注过source doc的哪些token，并且让这个coverage vector影响当前time step的attention计算：
$$e_i^t=v^Ttanh(W_hh_i+W_ss_t+w_cc_i^t+b_{attn})$$
这样做的目的在于，在模型进行当前time step进行attention计算的时候，告诉它之前它已经关注过的token，希望避免出现连续attention到某几个token上的情况。同时，考虑到重复token的位置的影响，coverage模型还添加一个额外的coverage loss，来对重复的attention作惩罚：
$$covloss_t=\sum_imin(a_i^t,c_i^t)$$
这样这个loss只会对重复的attention产生惩罚，并不会强制要求模型关注原文中的每一个词。加上词表预测的损失函数采用交叉熵：
$$loss=-\frac{1}{T}\sum_{t=0}^TlogP(w_t^*)$$
最终，模型的整体损失函数为：
$$loss_t=-logP(w_t^*)+\lambda\sum_imin(a_i^t,c_i^t)$$
文章在实验部分提到，如果移除了covloss，单纯依靠coverage vector去影响attention的计算并不能缓解重复token的问题，模型还是会重复地attention到某些token上。而加上covloss的模型训练上也比较trick，需要先用主函数训练好一个收敛的模型，然后再把covloss加上，做个finetune，不然的话效果还是不好。

# 实验结果
论文用的数据集是CNN/DailyMail数据集，可以看到论文的模型在该任务上有着明显的提升。
![在这里插入图片描述](https://img-blog.csdnimg.cn/048b56aeec7b4ad19019d50d65ba66be.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQm9Db25nLURlbmc=,size_20,color_FFFFFF,t_70,g_se,x_16)
下面是三种模型对同一篇原文生成的摘要，橘色的是最终coverage vector在原文上的分布，红色的是事实细节和OOV问题，绿色的是生成摘要时 $p_{gen}$ 的大小。
![在这里插入图片描述](https://img-blog.csdnimg.cn/1b6a871ff5904bad87c5e09d7d674b9f.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQm9Db25nLURlbmc=,size_20,color_FFFFFF,t_70,g_se,x_16)


# 总结
![在这里插入图片描述](https://img-blog.csdnimg.cn/aa0ba1662218416ea1bd64b31a7b3a17.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQm9Db25nLURlbmc=,size_20,color_FFFFFF,t_70,g_se,x_16)
![在这里插入图片描述](https://img-blog.csdnimg.cn/8764eb049d1343509cf00f7d71ddcf15.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQm9Db25nLURlbmc=,size_20,color_FFFFFF,t_70,g_se,x_16)

本文模型改善了抽象文本摘要中存在的主要问题，但与具象摘要结果相比仍然存在差距，同时考虑到新闻文章重要信息普遍集中分布于前部分的特性，抽象摘要模型受到了一定影响，模型的普适性需要进一步地验证。
