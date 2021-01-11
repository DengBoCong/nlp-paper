# 前言

> 标题：Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift\
> 原文链接：[Link](https://arxiv.org/pdf/1502.03167.pdf)\
> Github：[NLP相关Paper笔记和代码复现](https://github.com/DengBoCong/nlp-paper)\
> 说明：阅读论文时进行相关思想、结构、优缺点，内容进行提炼和记录，论文和相关引用会标明出处，引用之处如有侵权，烦请告知删除。\
> 转载请注明：DengBoCong

训练深度神经网络非常复杂，因为在训练过程中，随着先前各层的参数发生变化，各层输入的分布也会发生变化，导致调参工作要做的很小心，训练更加困难，论文中将这种现象称为“internal covariate shift”，而Batch Normalization正式用来解决深度神经网络中internal covariate shift现象的方法。有关covariate shift的内容，可以参阅我另一篇[论文阅读笔记](https://zhuanlan.zhihu.com/p/339719861)。

# 介绍
Batch Normalization是在每个mini-batch进行归一化操作，并将归一化操作作为模型体系结构的一部分，使用BN可以获得如下的好处：
+ **可以使用更大的学习率**，训练过程更加稳定，极大提高了训练速度。
+ **可以将bias置为0**，因为Batch Normalization的Standardization过程会移除直流分量，所以不再需要bias。
+ **对权重初始化不再敏感**，通常权重采样自0均值某方差的高斯分布，以往对高斯分布的方差设置十分重要，有了Batch Normalization后，对与同一个输出节点相连的权重进行放缩，其标准差也会放缩同样的倍数，相除抵消。
+ **对权重的尺度不再敏感**。
+ **深层网络可以使用sigmoid和tanh了**，BN抑制了梯度消失。
+ **Batch Normalization具有某种正则作用，不需要太依赖dropout，减少过拟合。**

我们从梯度计算开始看起，如在SGD中是优化参数 $\theta$，从而最小化损失，如下公式：
$$\theta=arg\underset{\theta}{min}\frac{1}{N}\sum_{i=1}^{N}l(x_i,\theta)$$
其中，$x_1...x_N$是训练数据集。使用SGD，训练将逐步进行，并且在每个步骤中，我们考虑大小为 $m$ 的mini-batch，即$x_1...m$，通过计算$\frac{1}{m}\frac{\partial(x_i,\theta)}{\partial\theta}$，使用小批量数据来近似损失函数关于参数的梯度。使用小批量样本，而不是一次一个样本，在一些方面是有帮助的。首先，小批量数据的梯度损失是训练集上的梯度估计，其质量随着批量增加而改善。第二，由于现代计算平台提供的并行性，对一个批次的计算比单个样本计算 $m$ 次效率更高。

虽然随机梯度是简单有效的，但它需要仔细调整模型的超参数，特别是优化中使用的学习速率以及模型参数的初始值。训练的复杂性在于每层的输入受到前面所有层的参数的影响——因此当网络变得更深时，网络参数的微小变化就会被放大。如果我们能保证非线性输入的分布在网络训练时保持更稳定，那么优化器将不太可能陷入饱和状态，训练将加速。

# BN之前的一些减少Covariate Shift的方法
对网络的输入进行白化，网络训练将会收敛的更快——即输入线性变换为具有零均值和单位方差，并去相关。当每一层观察下面的层产生的输入时，实现每一层输入进行相同的白化将是有利的。通过白化每一层的输入，我们将采取措施实现输入的固定分布，消除Internal Covariate Shift的不良影响。那么如何消除呢？考虑在每个训练步骤或在某些间隔来白化激活值，通过直接修改网络或根据网络激活值来更改优化方法的参数，但这样会弱化梯度下降步骤。

例如：例如，考虑一个层，其输入u加上学习到的偏置 $b$，通过减去在训练集上计算的激活值的均值对结果进行归一化：$\hat x=x - E[x]$，$x = u+b$，$X={x_{1\ldots N}}$ 是训练集上$x$ 值的集合，$E[x] = \frac{1}{N}\sum_{i=1}^N x_i$。如果梯度下降步骤忽略了 $E[x]$ 对 $b$的依赖，那它将更新$b\leftarrow b+\Delta b$，其中$\Delta b\propto -\partial{\ell}/\partial{\hat x}$。然后 $u+(b+\Delta b) -E[u+(b+\Delta b)] = u+b-E[u+b]$。因此，结合 $b$ 的更新和接下来标准化中的改变会导致层的输出没有变化，从而导致损失没有变化。随着训练的继续，$b$ 将无限增长而损失保持不变。如果标准化不仅中心化而且缩放了激活值，问题会变得更糟糕。在最初的实验中，当标准化参数在梯度下降步骤之外计算时，模型会爆炸。

总结而言就是使用白话来缓解ICS问题，白化是机器学习里面常用的一种规范化数据分布的方法，主要是PCA白化与ZCA白化。白化是对输入数据分布进行变换，进而达到以下两个目的：
+ 使得输入特征分布具有相同的均值与方差，其中PCA白化保证了所有特征分布均值为0，方差为1，而ZCA白化则保证了所有特征分布均值为0，方差相同。
+ 去除特征之间的相关性。

**通过白化操作，我们可以减缓ICS的问题，进而固定了每一层网络输入分布，加速网络训练过程的收敛。但是白话过程的计算成本太高，并且在每一轮训练中的每一层我们都需要做如此高成本计算的白化操作，这未免过于奢侈。而且白化过程由于改变了网络每一层的分布，因而改变了网络层中本身数据的表达能力，底层网络学习到的参数信息会被白化操作丢失掉。**

# BN算法描述
文中使用了类似z-score的归一化方式：每一维度减去自身均值，再除以自身标准差，由于使用的是随机梯度下降法，这些均值和方差也只能在当前迭代的batch中计算，故作者给这个算法命名为Batch Normalization。BN变换的算法如下所示，其中，为了数值稳定，$\epsilon$ 是一个加到小批量数据方差上的常量。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201228154526445.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
我们可以将上面的算法总结为两步：
+ **Standardizatio**n：首先对 $m$ 个 $x$ 进行Standardization，得到 zero mean unit variance的分布 $\hat{x}$。
+ **scale and shift**：然后再对 $\hat{x}$ 进行scale and shift，缩放并平移到新的分布 $y$，具有新的均值 $\beta$ 方差 $\gamma$。

更形象一点，假设BN层有 $d$ 个输入节点，则 $x$ 可构成 $d\times m$大小的矩阵 $X$，BN层相当于通过行操作将其映射为另一个 $d\times m$ 大小的矩阵$Y$，如下所示：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201228154941139.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
将2个过程写在一个公式里如下：
$$y_i^{(b)}=BN(x_i)^{(b)}=\gamma (\frac{x_i^{(b)}-\mu(x_i)}{\sqrt{\sigma(x_i)^2+\epsilon}})+\beta$$

其中，$x_i^{(b)}$ 表示输入当前batch的b-th样本时该层i-th输入节点的值，$x_i$ 为 $[x_i^{(1)},x_i^{(2)},…,x_i^{(m)}]$ 构成的行向量，长度为batch size $m$，$\mu$和$\sigma$为该行的均值和标准差，$\epsilon$ 为防止除零引入的极小量（可忽略），$\gamma$和$\beta$为该行的scale和shift参数，可知
+ $\mu$ 和 $\sigma$ 为当前行的统计量，不可学习。
+ $\gamma$ 和 $\beta$ 为待学习的scale和shift参数，用于控制 $y_i$ 的方差和均值。
+ BN层中，$x_i$ 和 $x_j$ 之间不存在信息交流 $(i\neq j)$

可见，**无论xi原本的均值和方差是多少，通过BatchNorm后其均值和方差分别变为待学习的 $\gamma$ 和 $\beta$**。为什么需要 $\gamma$ 和 $\beta$ 的可训练参数？Normalization操作我们虽然缓解了ICS问题，让每一层网络的输入数据分布都变得稳定，但却导致了数据表达能力的缺失。也就是我们通过变换操作改变了原有数据的信息表达（representation ability of the network），使得底层网络学习到的参数信息丢失。另一方面，单纯通过让每一层的输入分布均值为0，方差为1，而不做缩放和移位，会使得输入在经过sigmoid或tanh激活函数时，容易陷入非线性激活函数的线性区域。

> 在训练初期，分界面还在剧烈变化时，计算出的参数不稳定，所以退而求其次，**在 $Wx+b$ 之后，ReLU激活层前面进行归一化**。因为初始的 $W$ 是从标准高斯分布中采样得到的，而 $W$ 中元素的数量远大于 $x$，$Wx+b$ 每维的均值本身就接近 $0$、方差接近 $1$，所以在 $Wx+b$ 后使用Batch Normalization能得到更稳定的结果，如下图所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201228154233138.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)

# Batch Normalization的反向传播
讲反向传播之前，我们先来简单的写一下正向传递的代码，如下：

```python
def batchnorm_forward(x, gamma, beta, eps):
    N, D = x.shape
    # 第一步：计算平均
    mu = 1./N * np.sum(x, axis=0)
    # 第二步：每个训练样本减去平均
    xmu = x - mu
    # 第三步：计算分母
    sq = xmu ** 2
    # 第四步：计算方差
    var = 1./N * np.sum(sq, axis=0)
    # 第五步：加上eps保证数值稳定性，然后计算开方
    sqrtvar = np.sqrt(var + eps)
    # 第六步：倒转sqrtvar
    ivar = 1./sqrtvar
    # 第七步：计算归一化
    xhat = xmu * ivar
    # 第八步：加上两个参数
    gammax = gamma * xhat
    out = gammax + beta
    # cache储存计算反向传递所需要的一些内容
    cache = (xhat, gamma, xmu, ivar, sqrtvar, var, eps)

    return out, cache
```

我们都知道，对于目前的神经网络计算框架，一个层要想加入到网络中，要保证其是可微的，即可以求梯度。BatchNorm的梯度该如何求取？反向传播求梯度只需抓住一个关键点，如果一个变量对另一个变量有影响，那么他们之间就存在偏导数，找到直接相关的变量，再配合链式法则，公式就很容易写出了。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201228161718394.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
根据反向传播的顺序，首先求取损失 $l$ 对BN层输出 $y_i$ 的偏导 $\frac{\partial l}{\partial y_i}$，然后是对可学习参数的偏导 $\frac{\partial l}{\partial \gamma}$ 和 $\frac{\partial l}{\partial \beta}$，用于对参数进行更新，想继续回传的话还需要求对输入 $x$ 偏导，于是引出对变量 $\mu$、$\sigma^2$ 和 $\hat{x}$ 的偏导，根据链式法则再求这些变量对 $x$ 的偏导，计算图如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201228205034787.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
通过链式法则，我们可以对上面的正向传递的代码进行运算，得到反向传播的代码，如下（结合代码理解更方便）：

```python
def batchnorm_backward(dout, cache):
    # 展开存储在cache中的变量
    xhat, gamma, xmu, ivar, sqrtvar, var, eps = cache
    # 获得输入输出的维度
    N, D = dout.shape
    dbeta = np.sum(dout, axis=0)
    dgammax = dout
    dgamma = np.sum(dgammax * xhat, axis=0)
    dxhat = dgammax * gamma
    divar = np.sum(dxhat * xmu, axis=0)
    dxmu1 = dxhat * ivar
    dsqrtvar = -1./(sqrtvar ** 2) * divar
    dvar = 0.5 * 1. / np.sqrt(var + eps) * dsqrtvar
    dsq = 1. / N * np.ones((N, D)) * dvar
    dxmu2 = 2 * xmu * dsq
    dx1 = (dxmu1 + dxmu2)
    dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)
    dx2 = 1. / N * np.ones((N, D)) * dmu
    dx = dx1 + dx2

    return dx, dgamma, dbeta
```

# Batch Normalization的预测阶段
在预测阶段，所有参数的取值是固定的，对BN层而言，意味着$\mu$、$\sigma$、$\gamma$、$\beta$ 都是固定值。$\gamma$和$\beta$ 比较好理解，随着训练结束，两者最终收敛，预测阶段使用训练结束时的值即可。对于 $\mu$ 和 $\sigma$，在训练阶段，它们为当前mini batch的统计量，随着输入batch的不同， $\mu$ 和 $\sigma$ 一直在变化。在预测阶段，输入数据可能只有1条，该使用哪个 $\mu$ 和 $\sigma$ ，或者说，每个BN层的 $\mu$ 和 $\sigma$ 该如何取值？可以采用训练收敛最后几批mini batch的 $\mu$ 和 $\sigma$ 的期望，作为预测阶段的 $\mu$ 和 $\sigma$ ，如下所示：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201228162916396.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
因为Standardization和scale and shift均为线性变换，在预测阶段所有参数均固定的情况下，参数可以合并成 $y=kx+b$ 的形式，如上图中行号11所示。

这里多说一句，BN在卷积中使用时，1个卷积核产生1个feature map，1个feature map有1对 $\gamma$和$\beta$ 参数，同一batch同channel的feature map共享同一对 $\gamma$和$\beta$ 参数，若卷积层有 $n$ 个卷积核，则有 $n$ 对 $\gamma$和$\beta$ 参数。

# 实验结果
下图是使用三层全连接层，在每层之后添加BN以及无添加的实验对比：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201228170635438.png#pic_center)
下图是训练步和精度的实验结果：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201228172923853.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
下图是使用BN在Inception上的相关实验结果：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201228195302919.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)

## 关于BN的几个讨论
+ 没有scale and shift过程可不可以？
BatchNorm有两个过程，Standardization和scale and shift，前者是机器学习常用的数据预处理技术，在浅层模型中，只需对数据进行Standardization即可，Batch Normalization可不可以只有Standardization呢？答案是可以，但网络的表达能力会下降。直觉上理解，浅层模型中，只需要模型适应数据分布即可。对深度神经网络，每层的输入分布和权重要相互协调，强制把分布限制在zero mean unit variance并不见得是最好的选择，加入参数 $\gamma$和$\beta$ ，对输入进行scale and shift，有利于分布与权重的相互协调，特别地，令 $\gamma=1$和$\beta=0$ 等价于只用Standardization，令 $\gamma=\sigma$和$\beta=\mu$ 等价于没有BN层，scale and shift涵盖了这2种特殊情况，在训练过程中决定什么样的分布是适合的，所以使用scale and shift增强了网络的表达能力。表达能力更强，在实践中性能就会更好吗？并不见得，就像曾经参数越多不见得性能越好一样。在[caffenet-benchmark-batchnorm](https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md)中，作者实验发现没有scale and shift性能可能还更好一些，如下图：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201228195444340.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
+ BN层放在ReLU前面还是后面？实验表明，放在前后的差异似乎不大，甚至放在ReLU后还好一些（如上图），放在ReLU后相当于直接对每层的输入进行归一化，这与浅层模型的Standardization是一致的。[caffenet-benchmark-batchnorm](https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md)中有很多组合实验结果，可以看看。BN究竟应该放在激活的前面还是后面？以及，BN与其他变量，如激活函数、初始化方法、dropout等，如何组合才是最优？可能只有直觉和经验性的指导意见，具体问题的具体答案可能还是得实验说了算

# 写在最后
BN层的有效性已有目共睹，但为什么有效可能还需要进一步研究，还需要进一步研究，这里整理了一些关于BN为什么有效的论文，贴在这：
+ [How Does Batch Normalization Help Optimization?](https://arxiv.org/pdf/1805.11604.pdf)：**BN层让损失函数更平滑**。
论文中通过分析训练过程中每步梯度方向上步长变化引起的损失变化范围、梯度幅值的变化范围、光滑度的变化，认为添加BN层后，损失函数的landscape(loss surface)变得更平滑，相比高低不平上下起伏的loss surface，平滑loss surface的梯度预测性更好，可以选取较大的步长。如下图所示：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201228201948418.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
+ [An empirical analysis of the optimization of deep network loss surfaces](https://arxiv.org/pdf/1612.04010.pdf)：**BN更有利于梯度下降**。
论文中绘制了VGG和NIN网络在有无BN层的情况下，loss surface的差异，包含初始点位置以及不同优化算法最终收敛到的local minima位置，如下图所示。没有BN层的，其loss surface存在较大的高原，有BN层的则没有高原，而是山峰，因此更容易下降。
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020122820245825.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)




参考文献：
+ [caffenet-benchmark-batchnorm](https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md)
+ [Understanding the backward pass through Batch Normalization Layer](https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html)
+ [Why Does Batch Normalization Work?](https://abay.tech/blog/2018/07/01/why-does-batch-normalization-work/)
+ [Batch Normalization详解](https://www.cnblogs.com/shine-lee/p/11989612.html)
+ [Batch Normalization — What the hey](https://gab41.lab41.org/batch-normalization-what-the-hey-d480039a9e3b)
+ [How does Batch Normalization Help Optimization?](http://gradientscience.org/batchnorm/)
+ [How does Batch Normalization Help Optimization?](https://www.microsoft.com/en-us/research/video/how-does-batch-normalization-help-optimization/)
