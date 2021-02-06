
# 前言

> Paper 1：[Consistency of a Recurrent Language Model With Respect to Incomplete Decoding](https://arxiv.org/pdf/2002.02492.pdf)
> Paper 2：[A Theoretical Analysis of the Repetition Problem in Text Generation](https://arxiv.org/pdf/2012.14660.pdf)
> Github：[NLP相关Paper笔记和代码复现](https://github.com/DengBoCong/nlp-paper)
> 说明：阅读论文时进行相关思想、结构、优缺点，内容进行提炼和记录，论文和相关引用会标明出处，引用之处如有侵权，烦请告知删除。
> 转载请注明：DengBoCong

Seq2Seq模型在深度学习中已经屡见不鲜了，而且在各种任务上表现很出色，但是其中存在一个相信很多人都遇到的问题，也就是Seq2Seq在解码过程中，某些token反复出现，而且结束解码符号一直都不出现。针对这两个问题，上述两篇论文分别从“解码不停止”以及“重复解码”这两个方面进行分析陈述（其实这两个问题是同理的），而本篇文章针对两篇论文进行学习和总结。

# 介绍
用最大似然估计（MLE）训练的神经序列模型已经成为各种自然语言应用中的标准做法，但是其被证明存在length bias 和 degenerate repetition 问题（可能是与最大似然目标的局部归一化有关，即真实输出分布和算法学习到的分布不一致）。


# “解码不停止”
为了分析“解码不停止”问题，论文首先定义了解码算法的不一致性，即**该算法可以产生一个概率为零的无限长序列**。对于一致的递归语言模型与不完整的解码算法配合使用后，会导致不一致的序列分布的观点，论文中对此提出了两种解决不一致问题的方法
+ consistent sampling：以确保在解码期间不会将终止token排除在选择之外
+ self-terminating recurrent language model：该模型保证最终将终止token排在所有其他token之上，从而确保在不完全解码下的一致性

所以在进一步分析解码算法之前，需要先建立recurrent language model，方便后面分析，即条件语言模型在每个时间步的计算如下：
$$p_{\theta}(y_t=v|y_{<t},C)=\frac{exp(u_v^Th_t+c_v)}{\sum_{v^{'}\in V}exp(u_{v^{'}}^Th_t+c_{v^{'}})}$$
其中，$h_t=f_{\theta}(y_t,h_{t-1}),h_0=g_{\theta}(C),u,c,\theta$ 都是参数，$C$ 是上下文分布 $p(C)$ 的token集合中的元素。由此可以计算一条完整的序列的概率 $Y=(y_1,...,y_T)$ ：
$$p_{\theta}(Y|C)=\prod_{t=1}^{T}p_{\theta}(y_t|y_{<t},C)$$

其中，$y_{<t}=(y_1,...,y_{t-1})$。有了这些，我们可以开始来讨论解码算法，解码算法大致可以分为两类：
+ Stochastic decoding：随机性解码算法
+ Deterministic decoding：确定性解码算法

# Stochastic decoding
随机性解码算法，通俗点理解就是哪怕输入文本固定了，解码出来的输出文本也不是固定的。对于Seq2Seq来说，我们很多时候希望得到确定性的结果，所以多数场景下我们都是用Beam Search。但是由于Beam Search的生成结果可能会出现过于单一的现象，又或者有时候我们希望增加输出的多样性，这时候就需要随机性解码算法，论文中提到三种随机性解码算法
+ 原生随机解码：原生随机解码算法其实就是从文字表面意思理解的逻辑，即每步按概率随机采样一个token，直到采样到终止token才停止，这看着就和它的名字一样随意。
+ [top-k随机解码](https://arxiv.org/pdf/1805.04833.pdf)：其在原生随机采样上加以完善，也就是限定随机采样的范围，即只在每步最高的前k个token中，重新进行归一化之后再随机采样，如下，其中 $V_k=arg\underset{v^{'}}{top-k}p_{\theta}(v^{'}|y_{<t},C)$：
$$q(v)∝\left\{\begin{matrix}  p_{\theta}(v|y_{<t},C),\ if\ v\in V_k \\ 0,\ otherwise \end{matrix}\right.$$
+ [Nucleus随机解码](https://arxiv.org/pdf/1904.09751.pdf)：跟top-k随机解码类似，也是对采样空间做了个截断，截断方式是：固定 $p\in (0,1)$，然后只保留概率最高的、概率和刚好超过 $p$ 的若干个token，所以它也叫top-p采样。公式和上面相似，不过在于另 $v\in V_{\mu}$，其中$V_{\mu}=\{v_1,...,v_{k_{\mu}}\}$，则
$$k_{\mu}=min\{k|\sum_{i=1}^kp_{\theta}(v_i|y_{<t},C)>\mu\}$$


# Deterministic decoding
确定性解码算法就是当输入文本固定之后，解码出来的输出文本也是固定的，这类算法就包含比较常见的Greedy Search 和Beam Search，事实上Greedy Search是 Beam Search 的一个特例，所以只需要讨论Beam Search，如下图：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210206150514589.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)

正如上图所示，每个序列直到出现终止token就停止，最后从这k个已经完成终止的序列中选最优的那个输出。一般有两种选择，一是输出总得分最大的，二是输出平均得分最大的（处以各自token数），有时候还会根据实际需求加入长度惩罚等。

# 解码算法的一致性
正如我们前面定义的解码算法的不一致性，从Seq2Seq的模型设计和上面介绍的解码算法来看，并没有任何的理论保证解码过程一定能停下来，也就是说并没有东西保证一定会出现终止token标记，这只能靠模型自己学出来，而当模型学得不够好时，就会出现“根本停不下来”的现象了。针对这问题，论文提出了对应的策略，接下来要提到的对对策内容理解来自文章：[如何应对Seq2Seq中的“根本停不下来”问题？](https://kexue.fm/archives/7500)
## 有界的隐向量
在对最上面我们事先定义好的条件语言模型进行建模的一个经典方法就是：
$$p(y_t|y_{\lt t}, x)=softmax(Wh_t+b),\quad h_t=f(y_{\lt t}, x)$$
也就是说，先算一个隐向量 $h_t$，然后接一个全连接，然后softmax激活，在这种形式下，原论文提到：
> 如果对于任意的 $t$，$∥h_t∥$是有上界的，那么原生随机解码就能够“适可而止”。

看上去很强很实用的一个结论是不是？让$∥h_t∥$是有上界是一个很简单的事情，比如加个Layer Norm就可以了，那是不是说加个Layer Norm就可以解决所有的问题呢？并不是。上述结论理论上是对的，推理过程是：因为 $∥h_t∥$ 是有上界的，所以对于任意的 $t$、任意的token，$p(y_t|y_{\lt t}, x)$ 是有正的下界的（因为 $Wh_t$不会无穷大，所以 $e^{Wh_t}$ 也不会无穷大，归一化后也不会无限接近于0），那也就意味着存在一个正数 $\epsilon > 0$，总有 $p(\text{<eos>}|y_{\lt t}, x)\geq \epsilon$，因为概率是一个正数，因此只要你采样足够多步，总有机会采样到`<eos>`的，所以不会永远停不下来，示意图如下。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210206153804221.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)

这推理过程是不是有点让人啼笑皆非？没错，是能停，但是要采样足够多步，感觉就像是“只要你买足够多张彩票就一定能中头奖”一样，并没什么确切的实际价值。采样足够多步之后，该循环的、该重复的token可能都已经重复多次了，就算能停下来，得到的输出可能也没有意义了，或许还不如直接按长度截断。
## 主动添加`<eos>`
注意上述结论还只是对原生随机解码成立，对于top-k随机解码和Nucleus随机解码不一定成立，因为经过截断后`<eos>`就不一定出现在采样空间中了，当然，我们可以手动把<eos>添加进采样空间，所以就有了如下的结论：
> 如果对于任意的$t$，$∥h_t∥$是有上界的，并且我们把`<eos>`也加入到采样空间中，那么top-k随机解码和Nucleus随机解码就能够“适可而止”。


## 自截断设计
注意，上面的两个结论都只能用于随机解码，对于确定性解码来说，因为没有了随机性，所以我们没法保证一定能碰到`<eos>`。为此，原论文提出了一个自截断的设计：想办法让 $p(\text{<eos>}|y_{\lt t}, x)$ 有正的下界，而且这个下界随着 $t$ 的增大而增大，最终逐渐趋于1。这种自截断的设计也不复杂，就是定义 $p(\text{<eos>}|y_{\lt t}, x) = 1 - \alpha(h_t)$，其中
$$\alpha(h_0)=\sigma\left(w_{\text{<eos>}}^{\top} h_0 + b_{\text{<eos>}}\right)$$    $$\alpha(h_t)=\sigma\left(w_{\text{<eos>}}^{\top} h_t + b_{\text{<eos>}}\right)\left[1 - p(\text{<eos>}|y_{\lt {t-1}}, x)\right]$$
这里的 $\sigma(\cdot)$ 负责将 $\mathbb{R}$ 映射到 $[0, 1-\epsilon]$，比如可以用 $\sigma(\cdot)=(1-\epsilon)\text{sigmoid}(\cdot)$。设计好 $p(\text{<eos>}|y_{\lt t}, x)$ 后，剩下的token概率还是按照原来的softmax方式计算，然后乘以 $\alpha(h_t)$ 即可。现在我们有：
$$p(\text{<eos>}|y_{\lt t}, x)=1 - \sigma\left(w_{\text{<eos>}}^{\top} h_t + b_{\text{<eos>}}\right)\left[1 - p(\text{<eos>}|y_{\lt {t-1}}, x)\right]\\ =1 - \prod_{i=0}^t\sigma\left(w_{\text{<eos>}}^{\top} h_i + b_{\text{<eos>}}\right)\\ \geq 1 - (1 - \epsilon)^{t+1} $$
显然只要 $t > -\ln 2/\ln (1-\epsilon)$，$p(\text{<eos>}|y_{\lt t}, x) > 0.5$，也就是说，对于贪心搜索来说必然在 $-\ln 2/\ln (1-\epsilon)$ 步内停止，而对随着 $p(\text{<eos>}|y_{\lt t}, x)$ 越来越接近1，显然Beam Search也能在有限步停止。

示意图如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210206153904241.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)

# 实验结果
下图使用论文中讨论的解码算法的递归语言模型的非终止率：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210206154144299.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
下图是Paper中的示例：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210206154256330.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
下图展示了Paper中自截断设计的优势：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210206154439581.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)

关于第二篇文章，可以看这一篇解析：
[AAAI21 | Seq2Seq模型成为“复读机”的原因找到了？](https://mp.weixin.qq.com/s?__biz=MzIwNzc2NTk0NQ==&mid=2247508015&idx=1&sn=cdd96d721dbca5fc5bb0390536e1a18c&chksm=970f88f9a07801effb8994afae258b8e2b51bf8b499599133f3501d4ad92e6378f0568edd6ce&mpshare=1&scene=23&srcid=02035hH26qrD9MDND1bsXTaP&sharer_sharetime=1612281796520&sharer_shareid=e7514cbb57886566ce01f800bc56a3ad#rd)



参考文献：
+ [Consistency of a Recurrent Language Model With Respect to Incomplete Decoding](https://arxiv.org/pdf/2002.02492.pdf)
+ [A Theoretical Analysis of the Repetition Problem in Text Generation](https://arxiv.org/pdf/2012.14660.pdf)
+ [如何应对Seq2Seq中的“根本停不下来”问题？](https://kexue.fm/archives/7500)