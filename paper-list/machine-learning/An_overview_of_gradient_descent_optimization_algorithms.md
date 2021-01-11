# 前言

> 标题：An overview of gradient descent optimization algorithms\
> 原文链接：[Link](https://arxiv.org/pdf/1609.04747.pdf)\
> Github：[NLP相关Paper笔记和代码复现](https://github.com/DengBoCong/nlp-paper)\
> 说明：阅读论文时进行相关思想、结构、优缺点，内容进行提炼和记录，论文和相关引用会标明出处，引用之处如有侵权，烦请告知删除。\
> 转载请注明：DengBoCong

不管是使用PyTorch还是TensorFlow，用多了Optimizer优化器封装好的函数，对其内部使用的优化算法却没有仔细研究过，也很难对其优点和缺点进行实用的解释。所以打算以这一篇论文为主线并结合多篇优秀博文，回顾和总结目前主流的优化算法，对于没有深入了解过的算法，正好借这个机会学习一下。

# 写在前面

当前使用的许多优化算法，是对梯度下降法的衍生和优化。在微积分中，对多元函数的参数求 $\theta$ 偏导数，把求得的各个参数的导数以向量的形式写出来就是梯度。梯度就是函数变化最快的地方。梯度下降是迭代法的一种，在求解机器学习算法的模型参数 $\theta$ 时，即无约束问题时，梯度下降是最常采用的方法之一。 

这里定义一个通用的思路框架，方便我们后面理解各算法之间的关系和改进。首先定义待优化参数 $\theta$ ，目标函数 $J(\theta)$，学习率为 $\eta$，然后我们进行迭代优化，假设当前的epoch为 $t$ ，则有：
+ 计算目标函数关于当前参数的梯度： $g_t = \triangledown_{\theta_t} J(\theta_t)$
+ 根据历史梯度计算一阶动量和二阶动量：$m_t=\phi(g_1,g_2,...,g_t);V_t=\psi(g_1,g_2,...,g_t)$，
+ 计算当前时刻的下降梯度： $\triangledown_t=\eta\cdot \frac{m_t}{\sqrt{V_t}}$
+ 根据下降梯度进行更新： $\theta_{t+1} = \theta_t -\triangledown_t$

其中，$\theta_{t+1}$为下一个时刻的参数，$\theta_t$为当前时刻 $t$ 参数，后面的描述我们都将结合这个框架来进行。

这里提一下一些概念：
+ 鞍点：一个光滑函数的鞍点邻域的曲线，曲面，或超曲面，都位于这点的切线的不同边。例如这个二维图形，像个马鞍：在x-轴方向往上曲，在y-轴方向往下曲，鞍点就是（0，0）。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210111155320118.png#pic_center)
+ 指数加权平均、偏差修正：可参见这篇文章：[什么是指数加权平均、偏差修正？](https://www.cnblogs.com/guoyaohua/p/8544835.html)


# Gradient Descent（GD）
在GD中没有动量的概念，也就是说在上述框架中：$m_t=g_t；V_t=I^2$，则我们在当前时刻需要下降的梯度就是 $\triangledown_t=\eta\cdot g_t$ ，则使用梯度下降法更新参数为（假设当前样本为 $(x_i,y_i)$，每当样本输入时，参数即进行更新）：
$$\theta_{t+1}=\theta_t-\triangledown_t=\theta_t-\eta\cdot g_t=\theta_t-\eta\cdot \triangledown_{\theta_t} J_i(\theta_t,x_i,y_i)$$

梯度下降算法中，模型参数的更新调整，与代价函数关于模型参数的梯度有关，即沿着梯度的方向不断减小模型参数，从而最小化代价函数。基本策略可以理解为”在有限视距内寻找最快路径下山“，因此每走一步，参考当前位置最陡的方向(即梯度)进而迈出下一步，更形象的如下图：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210111142747267.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
标准的梯度下降主要有两个缺点：
+ 训练速度慢：在应用于大型数据集中，每输入一个样本都要更新一次参数，且每次迭代都要遍历所有的样本，会使得训练过程及其缓慢，需要花费很长时间才能得到收敛解。
+ 容易陷入局部最优解：由于是在有限视距内寻找下山的反向，当陷入平坦的洼地，会误以为到达了山地的最低点，从而不会继续往下走。所谓的局部最优解就是鞍点，落入鞍点，梯度为0，使得模型参数不在继续更新。

# Batch Gradient Descent（BGD）
BGD相对于标准GD进行了改进，改进的地方通过它的名字应该也能看出来，也就是不再是想标准GD一样，对每个样本输入都进行参数更新，而是针对一个批量的数据输入进行参数更新。我们假设**批量训练样本总数**为 $n$，样本为 $\{(x_1,y_1),..,(x_n, y_n)\}$  ，则在第 $i$ 对样本 $(x_i,y_i)$ 上损失函数关于参数的梯度为 $\triangledown_\theta J_i(\theta, x_i, y_i)$ , 则使用BGD更新参数为：
$$\theta_{t+1}=\theta_t-\eta\cdot\frac{1}{n}\cdot\sum_{i=1}^{n}\triangledown_{\theta_t} J_i(\theta_t, x_i, y_i)$$
从上面的公式我们可以看到，BGD其实是在一个批量的样本数据中，求取该批量样本梯度的均值来更新参数，即每次权值调整发生在批量样本输入之后，而不是每输入一个样本就更新一次模型参数，这样就会大大加快训练速度，但是还是不够，我们接着往下看。
# Stochastic Gradient Descent（SGD）
随机梯度下降法，不像BGD每一次参数更新，需要计算整个数据样本集的梯度，而是每次参数更新时，仅仅选取一个样本 $(x_i,y_i)$ 计算其梯度，参数更新公式为：

$$\theta_{t+1}=\theta_t-\eta\cdot\triangledown_{\theta_t} J_i(\theta_t, x_i, y_i)$$

公式看起来和上面标准GD一样，但是注意了，这里的样本是从批量中随机选取一个，而标准GD是所有的输入样本都进行计算。可以看到BGD和SGD是两个极端，SGD由于每次参数更新仅仅需要计算一个样本的梯度，训练速度很快，即使在样本量很大的情况下，可能只需要其中一部分样本就能迭代到最优解，由于每次迭代并不是都向着整体最优化方向，导致梯度下降的波动非常大（如下图），更容易从一个局部最优跳到另一个局部最优，准确度下降。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210111150957576.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
论文中提到，当缓慢降低学习率时，SGD会显示与BGD相同的收敛行为，几乎一定会收敛到局部（非凸优化）或全局最小值（凸优化）。

SGD的优点：
+ 虽然看起来SGD波动非常大，会走很多弯路，但是对梯度的要求很低（计算梯度快），而且对于引入噪声，大量的理论和实践工作证明，只要噪声不是特别大，SGD都能很好地收敛。
+ 应用大型数据集时，训练速度很快。比如每次从百万数据样本中，取几百个数据点，算一个SGD梯度，更新一下模型参数。相比于标准梯度下降法的遍历全部样本，每输入一个样本更新一次参数，要快得多。

SGD的缺点：
+ SGD在随机选择梯度的同时会引入噪声，使得权值更新的方向不一定正确（次要）。
+ SGD也没能单独克服局部最优解的问题（主要）。

# Mini-batch Gradient Descent（MBGD，也叫作SGD）
小批量梯度下降法就是结合BGD和SGD的折中，对于含有 $n$ 个训练样本的数据集，每次参数更新，选择一个大小为 $m(m<n)$的mini-batch数据样本计算其梯度，其参数更新公式如下：
$$\theta_{t+1}=\theta_t-\eta\cdot\frac{1}{m}\cdot\sum_{i=x}^{i=x+m-1}\triangledown_{\theta_t} J_i(\theta_t, x_i, y_i)$$
小批量梯度下降法即保证了训练的速度，又能保证最后收敛的准确率，目前的SGD默认是小批量梯度下降算法。常用的小批量尺寸范围在50到256之间，但可能因不同的应用而异。


MBGD的缺点：
+ Mini-batch gradient descent 不能保证很好的收敛性，learning rate 如果选择的太小，收敛速度会很慢，如果太大，loss function 就会在极小值处不停地震荡甚至偏离（有一种措施是先设定大一点的学习率，当两次迭代之间的变化低于某个阈值后，就减小 learning rate，不过这个阈值的设定需要提前写好，这样的话就不能够适应数据集的特点）。对于非凸函数，还要避免陷于局部极小值处，或者鞍点处，因为鞍点所有维度的梯度都接近于0，SGD 很容易被困在这里（会在鞍点或者局部最小点震荡跳动，因为在此点处，如果是BGD的训练集全集带入，则优化会停止不动，如果是mini-batch或者SGD，每次找到的梯度都是不同的，就会发生震荡，来回跳动）。
+ SGD对所有参数更新时应用同样的 learning rate，如果我们的数据是稀疏的，我们更希望对出现频率低的特征进行大一点的更新， 且learning rate会随着更新的次数逐渐变小。

# Momentum
momentum算法思想：参数更新时在一定程度上保留之前更新的方向，同时又利用当前batch的梯度微调最终的更新方向，简言之就是通过积累之前的动量来加速当前的梯度。从这里开始，我们引入一阶动量的概念（在mini-batch SGD的基础之上），也就是说，在最开始说的框架中， $m_{t+1}=\beta_1\cdot m_{t}+(1-\beta_1)\cdot g_t$，而 $V_t=I^2$ 不变，参数更新公式如下：
$$m_{t+1}=\beta_1\cdot m_{t}+(1-\beta_1)\cdot \triangledown_{\theta_t} J_i(\theta_t)$$        $$\theta_{t+1}=\theta_t-m_{t+1}$$

一阶动量是各个时刻梯度方向的指数移动平均值，约等于最近 $\frac{1}{(1-\beta_1)}$ 个时刻的梯度向量和的平均值（移动平均是啥看最上面的文章）。也就是说，$t$ 时刻的下降方向，不仅由当前点的梯度方向决定，而且由此前累积的下降方向决定。 $\beta_1$ 的经验值为0.9，这就意味着下降方向主要是此前累积的下降方向，并略微偏向当前时刻的下降方向。在梯度方向改变时，momentum能够降低参数更新速度，从而减少震荡，在梯度方向相同时，momentum可以加速参数更新， 从而加速收敛，如下图：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210111162859514.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
动量主要解决SGD的两个问题：
+ 随机梯度的方法（引入的噪声）
+ Hessian矩阵病态问题（可以理解为SGD在收敛过程中和正确梯度相比来回摆动比较大的问题）。

# Nesterov Accelerated Gradient
牛顿加速梯度（NAG, Nesterov accelerated gradient）算法，是Momentum动量算法的变种。momentum保留了上一时刻的梯度  $\triangledown_\theta J(\theta)$  ，对其没有进行任何改变，NAG是momentum的改进，在梯度更新时做一个矫正，具体做法就是在当前的梯度上添加上一时刻的动量 $\beta_1\cdot m_t$ ，梯度改变为  $\triangledown_\theta J(\theta-\beta_1\cdot m_t)$  ，参数更新公式如下：
$$m_{t+1}=\beta_1\cdot m_{t}+(1-\beta_1)\cdot \triangledown_{\theta_t} J(\theta_t-\beta_1\cdot m_t)$$    $$\theta_{t+1}=\theta_t-m_{t+1}$$

加上nesterov项后，梯度在大的跳跃后，进行计算对当前梯度进行校正。 下图是momentum和nesterrov的对比表述图如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210111164437433.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
Nesterov动量梯度的计算在模型参数施加当前速度之后，因此可以理解为往标准动量中添加了一个校正因子。在凸批量梯度的情况下，Nesterov动量将额外误差收敛率从 $O(1/k)$ (k步后)改进到 $O(1/k^2)$，然而，在随机梯度情况下，Nesterov动量对收敛率的作用却不是很大。

Momentum和Nexterov都是为了使梯度更新更灵活。但是人工设计的学习率总是有些生硬，下面介绍几种自适应学习率的方法。

# Adagrad
Adagrad其实是对学习率进行了一个约束，对于经常更新的参数，我们已经积累了大量关于它的知识，不希望被单个样本影响太大，希望学习速率慢一些；对于偶尔更新的参数，我们了解的信息太少，希望能从每个偶然出现的样本身上多学一些，即学习速率大一些。而该方法中开始使用二阶动量，才意味着“自适应学习率”优化算法时代的到来。

我们前面都没有好好的讨论二阶动量，二阶动量是个啥？它是用来度量历史更新频率的，二阶动量是迄今为止所有梯度值的平方和，即 $V_t = \sum_{i=1}^tg_t^2$，在最上面的框架中 $\triangledown_t=\eta\cdot \frac{m_t}{\sqrt{V_t}}$（在这里$m_t=I$）， 也就是说，我们的学习率现在是 $\frac{\eta}{\sqrt{V_t+\epsilon}}$（一般为了避免分母为0，会在分母上加一个小的平滑项 $\epsilon$），从这里我们就会发现 $\sqrt{V_t+\epsilon}$ 是恒大于0的，而且参数更新越频繁，二阶动量越大，学习率就越小，这一方法在稀疏数据场景下表现非常好，参数更新公式如下：
$$V_t = \sum_{i=1}^tg_t^2$$   $$\theta_{t+1}=\theta_t-\eta\frac{1}{\sqrt{V_t+\epsilon}}$$

细心的小伙伴应该会发现Adagrad还是存在一个很明显的缺点：
+ 仍需要手工设置一个全局学习率 $\eta$ , 如果 $\eta$ 设置过大的话，会使regularizer过于敏感，对梯度的调节太大
+ 中后期，分母上梯度累加的平方和会越来越大，使得参数更新量趋近于0，使得训练提前结束，无法学习

# Adadelta
由于AdaGrad调整学习率变化过于激进，我们考虑一个改变二阶动量计算方法的策略：不累积全部历史梯度，而只关注过去一段时间窗口的下降梯度，即Adadelta只累加固定大小的项，并且也不直接存储这些项，仅仅是近似计算对应的平均值（指数移动平均值），这就避免了二阶动量持续累积、导致训练过程提前结束的问题了，参数更新公式如下：
$$V_t = \beta_2\cdot V_{t-1} + (1-\beta_2)(\triangledown_{\theta_t} J(\theta_t))^2$$   $$\theta_{t+1}=\theta_t-\eta\frac{1}{\sqrt{V_t+\epsilon}}$$

观察上面的参数更新公式，我们发现还是依赖于全局学习率 $\eta$ ，但是原作者在此基础之上做出了一定的处理，上式经过牛顿迭代法之后，得到Adadelta最终迭代公式如下式，其中 $g_t = \triangledown_{\theta_t} J(\theta_t)$：
$$E[g_t^2]_t=\rho\cdot E[g_t^2]_{t-1}+(1-\rho)\cdot g_t^2$$    $$\triangledown_t=\frac{\sum_{i=1}^{t-1}\triangle\theta_r}{\sqrt{E[g_t^2]_t+\epsilon}}$$

**此时可以看出Adadelta已经不依赖全局learning rate了**，Adadelta有如下特点：
+ 训练初中期，加速效果不错，很快
+ 训练后期，反复在局部最小值附近抖动

# RMSprop
RMSProp算法修改了AdaGrad的梯度平方和累加为指数加权的移动平均，使得其在非凸设定下效果更好。设定参数：全局初始率 $\eta$ , 默认设为0.001，decay rate $\rho$ ，默认设置为0.9，一个极小的常量 $\epsilon$ ，通常为10e-6，参数更新公式如下，其中 $g_t = \triangledown_{\theta_t} J(\theta_t)$：
$$E[g_t^2]_t=\rho\cdot E[g_t^2]_{t-1}+(1-\rho)\cdot g_t^2$$    $$\triangledown_t=\frac{\eta}{\sqrt{E[g_t^2]_t+\epsilon}}\cdot g_t$$

+ 其实RMSprop依然依赖于全局学习率 $\eta$
+ RMSprop算是Adagrad的一种发展，和Adadelta的变体，效果趋于二者之间
+ 适合处理非平稳目标(包括季节性和周期性)——对于RNN效果很好

# Adaptive Moment Estimation（Adam）
其实有了前面的方法，Adam和Nadam的出现就很理所当然的了，因为它们结合了前面方法的一阶动量和二阶动量。我们看到，SGD-M和NAG在SGD基础上增加了一阶动量，AdaGrad和AdaDelta在SGD基础上增加了二阶动量，参数更新公式如下（按照最开始总结的计算框架）：
$$m_{t+1}=\beta_1\cdot m_{t}+(1-\beta_1)\cdot \triangledown_{\theta_t} J_i(\theta_t)$$      $$V_{t+1} = \beta_2\cdot V_{t} + (1-\beta_2)(\triangledown_{\theta_t} J(\theta_t))^2$$     $$\theta_{t+1}=\theta_t-\eta\frac{m_{t+1}}{\sqrt{V_{t+1}+\epsilon}}$$

通常情况下，默认值为$\beta_1=0.9$、$\beta_2=0.999$ 和 $\epsilon=10^{-8}$，Adam通常被认为对超参数的选择相当鲁棒，特点如下：
+ Adam梯度经过偏置校正后，每一次迭代学习率都有一个固定范围，使得参数比较平稳。
+ 结合了Adagrad善于处理稀疏梯度和RMSprop善于处理非平稳目标的优点
+ 为不同的参数计算不同的自适应学习率
+ 也适用于大多非凸优化问题——适用于大数据集和高维空间。

# AdaMax
Adamax是Adam的一种变体，此方法对学习率的上限提供了一个更简单的范围，即使用无穷范式，参数更新公式如下：
$$m_{t+1}=\beta_1\cdot m_{t}+(1-\beta_1)\cdot \triangledown_{\theta_t} J_i(\theta_t)$$      $$V_{t+1} = \beta_2^\infty\cdot V_{t} + (1-\beta_2^\infty)(\triangledown_{\theta_t} J(\theta_t))^\infty=max(\beta_2\cdot V_t, |\triangledown_{\theta_t}J(\theta_t)|)$$     $$\theta_{t+1}=\theta_t-\eta\frac{m_{t+1}}{\sqrt{V_{t+1}+\epsilon}}$$

通常情况下，默认值为$\beta_1=0.9$、$\beta_2=0.999$ 和 $\eta=0.002$

# Nadam
其实如果说要集成所有方法的优点于一身的话，Nadam应该就是了，Adam遗漏了啥？没错，就是Nesterov项，我们在Adam的基础上，加上Nesterov项就是Nadam了，参数更新公式如下：
$$m_{t+1}=\beta_1\cdot m_{t}+\frac{(1-\beta_1)}{(1-\beta_1^t)}\cdot \triangledown_{\theta_t} J_i(\theta_t)$$      $$V_{t+1} = \beta_2\cdot V_{t} + (1-\beta_2)(\triangledown_{\theta_t} J(\theta_t))^2$$     $$\theta_{t+1}=\theta_t-\eta\frac{m_{t+1}}{\sqrt{V_{t+1}+\epsilon}}$$

可以看出，Nadam对学习率有更强的约束，同时对梯度的更新也有更直接的影响。一般而言，在使用带动量的RMSprop或Adam的问题上，使用Nadam可以取得更好的结果。

来张直观的动态图展示上述优化算法的效果：
+ 下图描述了在一个曲面上，6种优化器的表现：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210111213136496.gif#pic_center)
+ 下图在一个存在鞍点的曲面，比较6中优化器的性能表现：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210111212630400.gif#pic_center)
+ 下图图比较了6种优化器收敛到目标点（五角星）的运行过程
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210111212612731.gif#pic_center)



# 总结
那种优化器最好？该选择哪种优化算法？目前还没能够达达成共识。Schaul et al (2014)展示了许多优化算法在大量学习任务上极具价值的比较。虽然结果表明，具有自适应学习率的优化器表现的很鲁棒，不分伯仲，但是没有哪种算法能够脱颖而出。

目前，最流行并且使用很高的优化器（算法）包括SGD、具有动量的SGD、RMSprop、具有动量的RMSProp、AdaDelta和Adam。在实际应用中，选择哪种优化器应结合具体问题；同时，也优化器的选择也取决于使用者对优化器的熟悉程度（比如参数的调节等等）。

+ 对于稀疏数据，尽量使用学习率可自适应的优化方法，不用手动调节，而且最好采用默认值
+ SGD通常训练时间更长，但是在好的初始化和学习率调度方案的情况下，结果更可靠
+ 如果在意更快的收敛，并且需要训练较深较复杂的网络时，推荐使用学习率自适应的优化方法。
+ Adadelta，RMSprop，Adam是比较相近的算法，在相似的情况下表现差不多。
+ 在想使用带动量的RMSprop，或者Adam的地方，大多可以使用Nadam取得更好的效果
+ 如果验证损失较长时间没有得到改善，可以停止训练。
+ 添加梯度噪声（高斯分布$N(0,\sigma_t^2)$）到参数更新，可使网络对不良初始化更加健壮，并有助于训练特别深而复杂的网络。


*参考文献*：
+ [An overview of gradient descent optimization algorithms](https://ruder.io/optimizing-gradient-descent/)
+ [深度学习最全优化方法总结比较（SGD，Adagrad，Adadelta，Adam，Adamax，Nadam）](https://zhuanlan.zhihu.com/p/22252270)
+ [visualize_optimizers](https://github.com/snnclsr/visualize_optimizers)
+ [lossfunctions](https://lossfunctions.tumblr.com/)
+ [优化算法Optimizer比较和总结](https://zhuanlan.zhihu.com/p/55150256)
+ [一个框架看懂优化算法之异同 SGD/AdaGrad/Adam](https://zhuanlan.zhihu.com/p/32230623)
+ [深度学习——优化器算法Optimizer详解（BGD、SGD、MBGD、Momentum、NAG、Adagrad、Adadelta、RMSprop、Adam）](https://www.cnblogs.com/guoyaohua/p/8542554.html)
+ [机器学习：各种优化器Optimizer的总结与比较](https://blog.csdn.net/weixin_40170902/article/details/80092628)
+ [optimizer优化算法总结](https://blog.csdn.net/muyu709287760/article/details/62531509#%E4%B8%89%E7%A7%8Dgradient-descent%E5%AF%B9%E6%AF%94)