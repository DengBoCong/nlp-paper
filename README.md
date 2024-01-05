<h1 align="center">NLP-Paper</h1>
<div align="center">

[![Blog](https://img.shields.io/badge/blog-@DengBoCong-blue.svg?style=social)](https://www.zhihu.com/people/dengbocong)
[![Paper Support](https://img.shields.io/badge/paper-repo-blue.svg?style=social)](https://github.com/DengBoCong/nlp-paper)
![Stars Thanks](https://img.shields.io/badge/Stars-thanks-brightgreen.svg?style=social&logo=trustpilot)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=social&logo=appveyor)

</div>


更新一些在我学习过程中阅读过的且感觉不错的论文，对于一些经典或思路很新奇的论文，会进行精读，并写一些阅读笔记同步上传更新。(内容同步更新到[知乎](https://www.zhihu.com/people/dengbocong)、[CSDN](https://dengbocong.blog.csdn.net/))，**论文按照时间顺序排放**。

**注：**
+ 文本相似度计算相关的复现代码以及工具包（Tf/Pytorch双版本）在这个仓库 ☞ [Text-Similarity](https://github.com/DengBoCong/text-similarity)
+ 对话系统构建项目在这个仓库 ☞ [Nlp-Dialogue](https://github.com/DengBoCong/nlp-dialogue)
+ 对部分复现论文代码以及NLP其他工具代码放在这 ☞ [paper-code](https://github.com/DengBoCong/paper/tree/master/paper-code)

为了方便查找论文以及归档，提供了搜索工具，使用方式如下：
```
python3 search_kits.py
```
<div align=center>
<img height="350" src="https://github.com/DengBoCong/nlp-paper/blob/master/paper-code/image/preview.gif" alt="Search kits" title="Search kits">
</div><br>

# Contents | 内容
<div align="center">
    
&nbsp;&nbsp;[大模型](https://github.com/DengBoCong/nlp-paper)&nbsp;&nbsp;•&nbsp;&nbsp;[聚类](https://github.com/DengBoCong/nlp-paper)&nbsp;&nbsp;•&nbsp;&nbsp;[向量召回](https://github.com/DengBoCong/nlp-paper)&nbsp;&nbsp;•&nbsp;&nbsp;[对话系统](https://github.com/DengBoCong/nlp-paper)&nbsp;&nbsp;•&nbsp;&nbsp;[对话状态管理](https://github.com/DengBoCong/nlp-paper)&nbsp;&nbsp;•&nbsp;&nbsp;[机器学习](https://github.com/DengBoCong/nlp-paper)&nbsp;&nbsp;•&nbsp;&nbsp;[语言模型](https://github.com/DengBoCong/nlp-paper)&nbsp;&nbsp;<br>&nbsp;&nbsp;[数据集](https://github.com/DengBoCong/nlp-paper)&nbsp;&nbsp;•&nbsp;&nbsp;[文本相似度/匹配/分类](https://github.com/DengBoCong/nlp-paper)&nbsp;&nbsp;•&nbsp;&nbsp;[深度学习](https://github.com/DengBoCong/nlp-paper)&nbsp;&nbsp;•&nbsp;&nbsp;[语音系统](https://github.com/DengBoCong/nlp-paper)&nbsp;&nbsp;•&nbsp;&nbsp;[语音识别](https://github.com/DengBoCong/nlp-paper)&nbsp;&nbsp;•&nbsp;&nbsp;[模型](https://github.com/DengBoCong/nlp-paper)&nbsp;&nbsp;<br>&nbsp;&nbsp;[预训练](https://github.com/DengBoCong/nlp-paper)&nbsp;&nbsp;•&nbsp;&nbsp;[Subword](https://github.com/DengBoCong/nlp-paper)&nbsp;&nbsp;•&nbsp;&nbsp;[任务型对话](https://github.com/DengBoCong/nlp-paper)&nbsp;&nbsp;•&nbsp;&nbsp;[对话状态跟踪](https://github.com/DengBoCong/nlp-paper)&nbsp;&nbsp;•&nbsp;&nbsp;[对话意图识别](https://github.com/DengBoCong/nlp-paper)&nbsp;&nbsp;•&nbsp;&nbsp;[对话槽位填充](https://github.com/DengBoCong/nlp-paper)&nbsp;&nbsp;<br>&nbsp;&nbsp;[GNN](https://github.com/DengBoCong/nlp-paper)&nbsp;&nbsp;•&nbsp;&nbsp;[检索式对话系统](https://github.com/DengBoCong/nlp-paper)&nbsp;&nbsp;•&nbsp;&nbsp;[语音合成](https://github.com/DengBoCong/nlp-paper)&nbsp;&nbsp;•&nbsp;&nbsp;[综述](https://github.com/DengBoCong/nlp-paper)&nbsp;&nbsp;•&nbsp;&nbsp;[无监督](https://github.com/DengBoCong/nlp-paper)&nbsp;&nbsp;•&nbsp;&nbsp;[数据增强](https://github.com/DengBoCong/nlp-paper)&nbsp;&nbsp;<br>&nbsp;&nbsp;[阅读理解模型](https://github.com/DengBoCong/nlp-paper)&nbsp;&nbsp;•&nbsp;&nbsp;[可解释性](https://github.com/DengBoCong/nlp-paper)&nbsp;&nbsp;•&nbsp;&nbsp;[Prompt](https://github.com/DengBoCong/nlp-paper)&nbsp;&nbsp;•&nbsp;&nbsp;[评估](https://github.com/DengBoCong/nlp-paper)&nbsp;&nbsp;•&nbsp;&nbsp;[对话策略学习](https://github.com/DengBoCong/nlp-paper)&nbsp;&nbsp;•&nbsp;&nbsp;[关系抽取](https://github.com/DengBoCong/nlp-paper)&nbsp;&nbsp;<br>&nbsp;&nbsp;[蒸馏](https://github.com/DengBoCong/nlp-paper)&nbsp;&nbsp;•&nbsp;&nbsp;[异常检测](https://github.com/DengBoCong/nlp-paper)&nbsp;&nbsp;•&nbsp;&nbsp;[自监督](https://github.com/DengBoCong/nlp-paper)&nbsp;&nbsp;•&nbsp;&nbsp;[损失函数](https://github.com/DengBoCong/nlp-paper)&nbsp;&nbsp;•&nbsp;&nbsp;[半监督](https://github.com/DengBoCong/nlp-paper)&nbsp;&nbsp;•&nbsp;&nbsp;[社区发现](https://github.com/DengBoCong/nlp-paper)&nbsp;&nbsp;<br>&nbsp;&nbsp;[图算法](https://github.com/DengBoCong/nlp-paper)&nbsp;&nbsp;•&nbsp;&nbsp;[搜排](https://github.com/DengBoCong/nlp-paper)&nbsp;&nbsp;•&nbsp;&nbsp;[文本摘要](https://github.com/DengBoCong/nlp-paper)&nbsp;&nbsp;

</div>

# Paper List | 论文列表
```
注：论文按时间排序，并进行分类归档，可直接在本页Ctrl+F查询，或使用上述搜索工具查询（推荐）
    下述列表项格式：<标签 | 论文 | 阅读笔记 | 简述 | 作者时间>
```
+ [图算法-搜排] | [The PageRank Citation Ranking: Bringing Order to the Web](http://ilpubs.stanford.edu:8090/422/1/1999-66.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/137561088) | 经典的PageRank算法，最初是作为互联网网页的重要度计算方法，被用于谷歌搜索引擎网页排序。该算法的核心思想就是在有向图（带权）上定义一个随机游走模型（一阶马尔可夫链），在一定的条件下，使得极限情况访问每个节点的概率收敛到平稳分布，节点上的平稳概率值就是PageRank值，用于表示节点的重要度 | L Page et al, 1998

+ [聚类] | [Accelerating exact k-means algorithms with geometric reasoning](http://portal.acm.org/citation.cfm?doid=312129.312248) | [阅读笔记](https://www.zhihu.com/question/494753171/answer/2204649294) | K-Means引入计算机中的那片论文，K-Means属于Partition-based methods，思想是初始化中心点，然后通过启发式算法，达到”类内的点都足够近，类间的点都足够远“的目标 | et al Dan Pelleg,1999

+ [异常检测-机器学习] | [LOF: Identifying Density-Based Local Outliers](https://dl.acm.org/doi/pdf/10.1145/342009.335388) | [阅读笔记](https://zhuanlan.zhihu.com/p/28178476) | 本文提出的LOF算法是基于密度的算法，其优势在于同时考虑了数据集的局部和全局属性（其中局部可达密度的定义其实暗含了一个假设，即不存在大于等于K个重复的点），异常值不是按绝对值确定的，而是相对于它们的领域点密度确定的。因此，当数据集中存在不同密度的不同集群时，LOF算法表现良好，比较适合中等高维的数据集 | Markus M. Breunig et al, 2000

+ [聚类] | [Mean Shift: A Robust Approach toward Feature Space Analysis](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.76.8968&rep=rep1&type=pdf) | [阅读笔记](https://www.zhihu.com/question/494753171/answer/2204649294) | 实现的方法是滑动窗口的算法，在每次迭代中，通过将中心点移动到窗口内所有点的平均值位置（因此得名），将滑动窗口中心移向密度较高的区域。滑动窗口内的密度与其内部的点数成正比。通过转换到窗口内点的平均值位置，窗口将逐渐移动到有着更高点密度的区域。优点：Mean-Shift的最大优势就是可以自动发现簇的数量而不需要人工选择；簇的中心向最大密度点聚合的事实也是非常令人满意的，因为它可被非常直观地理解并很自然地契合数据驱动；可以处理任意形状的簇类；算法只需设置半径这一个参数，半径影响数据集的核密度估计；算法结果稳定，不需要进行类似K均值的样本初始化；缺点：不足就是窗口大小/半径“r”的选择可能是非平凡的；半径设置的太小，收敛太慢，簇类个数过多；半径设置的太大，一些簇类可能会丢失。对于较大的特征空间，计算量非常大 | Dorin Comaniciu et al,2002

+ [向量召回] | [similarity estimation techniques from rounding algorithms](https://www.cs.princeton.edu/courses/archive/spring04/cos598B/bib/CharikarEstim.pdf) | [阅读笔记](http://tangxman.github.io/2015/12/01/simhash/) | 论文提出的SimHash是当年Google用来文本去重的算法。主要做法是将文档提取出一定数量的关键词，然后转换成哈希码并按列相加，1+weight，0-weight，得到的结果按照整数为1，负数为0得到最终的哈希码，然后将哈希码分为m个table，并分别记性计算检索 | Moses S. Charikar et al,2002

+ [图算法-文本摘要-无监督] | [TextRank: Bringing Order into Texts](https://aclanthology.org/W04-3252.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/126733456) | 本文提出的是一种基于图的用于关键词抽取和文档摘要的排序算法，由PageRank算法改进而来，它利用一篇文档内部的词语间的共现信息(语义)便可以抽取关键词，并使用抽取式的自动文摘方法抽取出该文本的关键句，相对于TF-IDF方法，可以更充分的利用文本元素之间的关系。当然，它也同样存在受分词、停用词、文本清洗的影响 | Rada Mihalcea et al, 2004

+ [聚类] | [k-means++: The Advantages of Careful Seeding](http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf) | [阅读笔记](https://www.zhihu.com/question/494753171/answer/2204649294) | 原始K-Means（随机选择簇中心）对于初始化簇中心敏感，因此k-means++进行了改进，随机选择第一个初始中心点，计算其他点与该中心点的距离，按照距离远的以较大的概率被选中来选择第二个初始中心点，一次类推 | et al David Arthur,2006

+ [聚类] | [Clustering by Passing Messages Between Data Points](https://warwick.ac.uk/fac/sci/dcs/research/combi/seminars/freydueck_affinitypropagation_science2007.pdf) | [阅读笔记](https://www.zhihu.com/question/494753171/answer/2204649294) | 其基本思想是将全部样本看作网络的节点，然后通过网络中各条边的消息传递计算出各样本的聚类中心。聚类过程中，共有两种消息在各节点间传递，分别是responsibility和availability 。AP算法通过迭代过程不断更新每一个点的吸引度和归属度值，直到产生m个高质量的Exemplar。优点是无需指定聚类“数量”参数；聚类中心点由实际的样本点中产生；初始值不敏感，且对距离矩阵的对称性没要求。AP通过输入相似度矩阵来启动算法，因此允许数据呈非对称，数据适用范围非常大，鲁棒性很好；误差低；缺点是AP聚类应用中需要手动指定Preference和Damping factor，这其实是原有的聚类“数量”控制的变体，且算法复杂度较高 | Brendan J. Frey et al，2007

+ [聚类-社区发现-图算法] | [Maps of random walks on complex networks reveal community structure](https://arxiv.org/pdf/0707.0609.pdf) | [阅读笔记1](https://kexue.fm/archives/7006) / [阅读笔记2](https://zhuanlan.zhihu.com/p/53085574) | 经典的infomap算法，其属于动态社区发现算法。infoma的核心思想是通过构造转移概率，在图上进行随机游走来生成序列，再通过对序列做层次编码，最小化目标，从而完成聚类。其中有几个点需要说明的是：（1）转移概率的构造，例如在带权图的基础上，通过对权重的归一化得到概率（由于优化目标中只看相对概率，所以事实上归不归一化都行）；（2）随机游走是指在图中按照概率，从一个点跳到另一点，从而得到的路径序列（实现上不需要真的生成序列，解概率方程就行，目标就是优化到这个随机序列达到平稳）；（3）所谓最小化的目标，是使用层次编码的方案下，得到的最小信息熵（最短编码长度）目标函数；（4）节点合并到类的环节，是按顺序依次尝试将每个节点赋给邻居节点所在的类，取平均比特下降最大时的类赋给该节点，如果没有下降，该节点的类不变。infomap算法有很清晰的信息论解释，还几乎没有任何超参（唯一一个“穿越概率”参数） | M. Rosvall et al, 2007

+ [社区发现-聚类-图算法] | [Near linear time algorithm to detect community structures in large-scale networks](https://journals.aps.org/pre/pdf/10.1103/PhysRevE.76.036106) | [阅读笔记](https://www.cnblogs.com/LittleHann/p/10699988.html) | LPA是基于标签传播的社区发现算法，其核心的思想不复杂，通过给图中节点初始化唯一标签（PUSH算法），然后再每次迭代中随机选取节点，根据与其相连的节点所属的标签改变自己的标签，选择方式可以根据数量、权重等，如果存在多个相同则随机选取，直到多次迭代后稳定。LPA算法简单，且不需要指定社区个数，但是缺点在于算法过程中的更新顺序和随机选择，使得算法并不稳定，改进的切入点自然就是从这两个方面入手 | Usha Nandini Raghavan et al, 2007

+ [聚类] | [A Tutorial on Spectral Clustering](https://arxiv.org/pdf/0711.0189.pdf) | [阅读笔记](https://www.zhihu.com/question/494753171/answer/2204649294) | 不是原论文，但是这一篇对Spectral Clustering讲的非常好，谱聚类（Spectral Clustering），就是先用Laplacian eigenmaps对数据降维（简单地说，就是先将数据转换成邻接矩阵或相似性矩阵，再转换成Laplacian矩阵，再对Laplacian矩阵进行特征分解，把最小的K个特征向量排列在一起），然后再使用k-means完成聚类。谱聚类是个很好的方法，效果通常比k-means好，计算复杂度还低，这都要归功于降维的作用。优点：谱聚类只需要数据之间的相似度矩阵，因此对于处理稀疏数据的聚类很有效。这点传统聚类算法比如K-Means很难做到；由于使用了降维，因此在处理高维数据聚类时的复杂度比传统聚类算法好。缺点：如果最终聚类的维度非常高，则由于降维的幅度不够，谱聚类的运行速度和最后的聚类效果均不好；聚类效果依赖于相似矩阵，不同的相似矩阵得到的最终聚类效果可能很不同 | Ulrike von Luxburg et al,2007

+ [异常检测-模型-机器学习] | [Isolation Forest](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf?q=isolation-forest) | [阅读笔记1](https://zhuanlan.zhihu.com/p/492469453) / [阅读笔记2](https://zhuanlan.zhihu.com/p/25040651) / [阅读笔记3](https://zhuanlan.zhihu.com/p/74508141) | 经典的孤立森林算法的初版paper，2012发表了扩充版本（Isolation-based anomaly detection）。孤立森林是一个基于Ensemble的快速离群点检测方法，主要针对的是连续型结构化数据中的异常点，具有线性的时间复杂度和高精准度。它的理论基础是（1）异常数据占样本量的比例很小；（2）异常点的特征值与正常点的差异很大。孤立森林简单高效，但是在一些情况下，比如说数据的分布不是沿着特征轴，而是随意分布，或者流型分布，孤立森林效果就不好，就需要考虑选择别的方式了 | Fei Tony Liu et al, 2008

+ [社区发现-图算法] | [Fast unfolding of communities in large networks](https://arxiv.org/pdf/0803.0476.pdf) | [阅读笔记1](https://zhuanlan.zhihu.com/p/178790546) | [阅读笔记2](https://zhuanlan.zhihu.com/p/556291759) | 经典的Louvain算法，其是基于模块度优化的社区发现算法，且是一种启发式贪婪算法。流程是：（1）初始时将每个顶点当作一个社区，社区个数与顶点个数相同；（2）依次将每个顶点与之相邻顶点合并在一起，计算它们最大的模块度增益是否大于0，如果大于0，就将该结点放入模块度增量最大的相邻结点所在社区；（3）迭代第二步，直至算法稳定，即所有顶点所属社区不再变化；（4）将各个社区所有节点压缩成为一个结点，社区内点的权重转化为新结点环的权重，社区间权重转化为新结点边的权重；（5）重复步骤1-3，直至算法稳定。一般认为用于评估效果的模块化指数在0.3~0.7就有明显的社区结构出现。Louvain算法的优点是时间复杂度低（nlogn），适合大规模的网络、社区划分结果稳定且有具体指标、天然自带层次化。而缺点在于容易导致”过拟合“。 | Vincent D. Blondel et al, 2008

+ [对话系统-对话状态管理] | [The Hidden Information State model: A practical framework for POMDP-based spoken dialogue management](https://www.sciencedirect.com/science/article/abs/pii/S0885230809000230) | 关于对话状态管理的文章，可以用来补充相关背景知识 | Young et al,2010

+ [向量召回] | [Product quantization for nearest neighbor search](https://lear.inrialpes.fr/pubs/2011/JDS11/jegou_searching_with_quantization.pdf) | [阅读笔记](http://vividfree.github.io/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/2017/08/05/understanding-product-quantization) | 这篇Paper是PQ算法的成功开始，PQ是一种量化方法，本质上是数据的一种压缩表达方式（本篇论文使用了KMeans算法，得到质心的codebook），先将向量分成m段，每段分别根据codebook转换成压缩向量，然后使用SDC或ADC算法进行相似搜索。不过论文中进一步进行了改进，提出了IVFADC算法，一种基于倒排索引的ADC算法，分两步，第一步是PQ一遍（成为coarse quantizer），然后用向量减去量化后的向量得到残差，第二步就是在所有得到的残差集合上在进行一次PQ，最后用得到的向量建立倒排索引 | Herve Jegou et al,2011

+ [聚类] | [Scalable K-Means++](https://theory.stanford.edu/~sergei/papers/vldb12-kmpar.pdf) | [阅读笔记](https://www.zhihu.com/question/494753171/answer/2204649294) | K-Means++由于它的采样策略，所以难以并行，限制了其用于大规模数据集上。为了解决这个问题，k-means II 改变取样策略（以oversampling的方式），初始化一个中心点，然后循环log(n)次，每次按照一个概率计算公式选择多个point加入到中心集，最后得到的候选中心集再通过k-means++对候选中心集进行聚类，选出k个簇中心 | Bahman Bahmani et al,2012
  
+ [向量召回] | [Fast Search in Hamming Space with Multi-Index Hashing](https://www.cs.toronto.edu/~norouzi/research/papers/multi_index_hashing.pdf) | [阅读笔记](https://tangxman.github.io/2015/12/03/mih/) | 主要是解决在汉明空间上的R-Neighbors of query和KNN query，论文提出了一种多分段索引的哈希方法，查询效率达到了次线性，做法是r为查询的汉明距离，将汉明码切分成m段，快速找出每段中汉明距离小于r/m的结果，合并所有结果即为候选集 | Mohammad Norouzi et al,2012

+ [向量召回] | [Learning Deep Structured Semantic Models for Web Search using Clickthrough Data](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/421944601) | 经典的DSSM语义相似度匹配模型，就是通常我们所说的双塔模型。使用Word Hashing的n-gram，在那个时候还是很独到的，其核心思想是将query和doc映射到到共同维度的语义空间中，通过最大化query和doc语义向量之间的余弦相似度，从而训练得到隐含语义模型，达到检索的目的。负采样1:4 | Po-Sen Huang et al,2013

+ [机器学习] | [Parameter Server for Distributed Machine Learning](http://www.cs.cmu.edu/~feixia/files/ps.pdf) | [阅读笔记](https://www.zybuluo.com/Dounm/note/517675) | 论文里说本篇所介绍的Parameter Server属于第三代PS，提供了更加通用的设计，架构上包括一个Server Group和若干个Worker Group，提供了如下几个特点：Efficient Communication、Elastic Scalability、Fault Tolerance and Durability、Ease of Use | Mu Li et al,2013

+ [向量召回] | [Optimized Product Quantization](https://www.microsoft.com/en-us/research/wp-content/uploads/2013/11/pami13opq.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/213395313) | PQ的做法是直接简单粗暴的分割原始特征向量，并没有去除相关性，而如果去除相关性之后再进行PQ可以使得检索效果更好，OPQ就提供了是的每个子空间信息均衡的方法，即使用一个正交矩阵来对聚类中心进行旋转，并提供了Non-Parametric和Parametric的两种算法思路 | Tiezheng Ge et al,2013
  
+ [语言模型] | [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/26306795) | Word2vec开山之作之一，专门讲训练中的两个trick：hierarchical softmax 和 negative sampling | Tomas Mikolov et al,2013

+ [语言模型] | [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/pdf/1310.4546.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/26306795) | Word2vec开山之作之一，在前人基础上提出更精简的语言模型框架并用于生成词向量，这个框架就是 Word2vec | Tomas Mikolov et al,2013

+ [向量召回] | [Speeding Up the Xbox Recommender System Using a Euclidean Transformation for Inner-Product Spaces](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/XboxInnerProduct.pdf) | 微软的Inner Product快速计算的方法，主要解决的是Inner Product Top-K Search的问题。通过各种公式证明，将问题简化到一个欧氏距离搜索问题后，使用一个PCA-Tree来求解 | Yoram Bachrach et al,2014

+ [机器学习] | [Scaling Distributed Machine Learning with the Parameter Server](https://www.cs.cmu.edu/~muli/file/parameter_server_osdi14.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/21569493) | 在PS中，每个 server 实际上都只负责分到的部分参数（servers共同维持一个全局的共享参数），而每个 work 也只分到部分数据和处理任务。较它之前一篇PS进行了更加细节的描述，并对一些细节进行了扩展，两篇结合起来看受益颇多 | Mu Li et al,2014

+ [向量召回] | [Approximate nearest neighbor algorithm based on navigable small world graphs](sciencedirect.com/science/article/abs/pii/S0306437913001300) | [阅读笔记](https://blog.csdn.net/u011233351/article/details/85116719) | 经典的NSW算法，在构建近似DG图的基础上，加入Expressway mechanism。构建时，在朴素插入选近邻连接的思路上，使用废弃列表和动态列表提速 Yury Malkov et al,2014

+ [数据集] | [The Second Dialog State Tracking Challenge](https://www.aclweb.org/anthology/W14-4337.pdf) | DSTC系列语料是专门用于对话状态跟踪的，非常经典，不过它的官网貌似无用了 |  Henderson et al,2014

+ [向量召回] | [Locally Optimized Product Quantization for Approximate Nearest Neighbor Search](https://openaccess.thecvf.com/content_cvpr_2014/papers/Kalantidis_Locally_Optimized_Product_2014_CVPR_paper.pdf) | LOPQ实在OPQ的基础上进一步优化，OPQ仅考虑了CodeBook的旋转问题，LOPQ考虑的是每个子空间进行不同的旋转 | Yannis Kalantidis et al,2014

+ [向量召回] | [Asymmetric LSH (ALSH) for Sublinear Time Maximum Inner Product Search (MIPS)](https://arxiv.org/pdf/1405.5869.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/111502331) | 传统的MIPS问题找不到LSH函数，为此论文提出了一种“非对称LSH”的算法，其核心技巧就是通过“非对称变换”构造向量从而消除待查集合X的向量模长对MIPS结果的影响。巧妙的将问题转换为欧氏距离下，通过LSH函数求出NN的近似解的问题 | Anshumali Shrivastava et al,2014

+ [图算法-GNN-模型-无监督] | [DeepWalk: Online Learning of Social Representations](https://arxiv.org/pdf/1403.6652.pdf) | [阅读笔记1](https://zhuanlan.zhihu.com/p/45167021) / [阅读笔记2](https://zhuanlan.zhihu.com/p/56380812) | 本文提出的DeepWalk是我们了解Graph Embedding无法绕过的一个方法。其核心思想是通过使用随机游走(RandomWalk)的方式在图中进行节点采样，从而使用图中节点与节点的共现关系来学习节点的向量表示（思想来源于Word2Vec的skip-gram）。总体分为两步，第一步就是随机游走采样节点序列，然后使用skip-gram来学习表示向量 | Bryan Perozzi et al, 2014

+ [语言模型-文本相似度/匹配/分类] | [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf) | 经典的TextCNN，static/non-static几种特征向量学习方式 | Yoon Kim et al,2014

+ [深度学习] | [Neural Machine Translation Bu Jointly Learning To Align And Translate](https://arxiv.org/pdf/1409.0473.pdf) | Bahdanau Attention的原文 | Bahdanau et al,2014

+ [深度学习] | [Convolutional Neural Networks at Constrained Time Cost](https://arxiv.org/pdf/1412.1710.pdf) | 针对卷积网络很好地概述了计算成本以及深度，过滤器尺寸之间的权衡 | Kaiming He et al,2014

+ [语音系统-语音识别-模型] | [Attention-Based Models for Speech Recognition](https://proceedings.neurips.cc/paper/2015/file/1068c6e4c8051cfd4e9ea8072e3189e2-Paper.pdf) | Tacotron2使用的Location Sensitive Attention  |  Chorowski et al,2015
  
+ [对话系统] | [Context Sensitive Spoken Language Understanding Using Role Dependent LSTM Layers](https://www.merl.com/publications/docs/TR2015-134.pdf) | 使用LSTM在SLU方面做的工作，通过agent和client角色划分，能够解决多轮对话中的歧义问题 | Hori et al,2015
  
+ [深度学习] | [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/340219662) | 经典的Batch Normalization原论文 | Sergey et al,2015

+ [蒸馏-预训练] | [Distilling the Knowledge in a Neural Network](https://arxiv.org/pdf/1503.02531.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/75031938) | 蒸馏方法的开山之作啦，将一个复杂模型的knowledge，transfer到一个简单的模型上。具体做法是给复杂分类模型（teacher）的softmax加上一个temperature参数，然后用hard target训练好，模型的softmax的输出就是我们需要的soft target。然后用一个simple模型，基于soft和hard target进行训练，simple模型在soft target训练时，softmax的temperature设置和teacher一样，在hard target训练时，temperature设置1即可，然后loss计算取两个目标的交叉熵的加权平均（soft targets和小模型的输出数据的交叉熵，hard targets和小模型的输出数据的交叉熵）。除此之外，通过梯度计算公式转换，我当temperature特别大的时候（且模型产生的logits为0），知识蒸馏就相当于大模型的logits和小模型的logits的MSE | Geoffrey Hinton et al,2015

+ [GNN-图算法-模型-无监督] | [LINE: Large-scale Information Network Embedding](https://arxiv.org/pdf/1503.03578.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/56478167) | 本文提出的LINE方法是应用于graph embedding，是一种采用基于领域相似假设的方法。文中总共提出了两个维度的相似计算视角：（1）一度相似性（First-order）适用于无向图，是认为图中存在直接连接的两个点是相似的，因此目的是使得两个点的向量表示分布尽可能相似；（2）二度相似性（Second-order）适用于无向图或有向图，认为的是一个节点，需要学习自己的表示向量之外，还需要一个用于表示与其直接相邻节点的表示（作为上下文向量），当两个没有直接连接的节点时，如果它们的邻居节点重合，便可以将上下文向量用于计算相似性。节点数字序列编号，并用embedding层编码，两种方法都是通过KL散度作为目标函数进行优化计算 | Jian Tang et al, 2015
  
+ [模型] | [Highway Networks](https://arxiv.org/pdf/1505.00387.pdf) | [阅读笔记](https://www.zhihu.com/question/279426970/answer/614880515) | Highway Networks名字取得很有意思，整个网络结构思想也是符合取名的。简单来说就是通过设置一个函数T来限制网络的输出（借鉴LSTM中gate思想），其中T取0时，输出y=x，这个时候梯度直接传到下一层，也就是说，可以通过T来控制梯度传递，从而一定程度上解决梯度消失的问题。Highway的参数较少，适合single nonlinear layer的transform | Rupesh Kumar Srivastava et al,2015

+ [深度学习] | [Learning both Weights and Connections for Efficient Neural Networks](https://arxiv.org/pdf/1506.02626.pdf) | 有一张表格，其中列出了计算与内存访问的相对成本，除此之外还讨论了怎么精简神经网络 | Song Han et al,2015

+ [模型] | [Pointer Networks](https://arxiv.org/pdf/1506.03134.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/48959800) | 原文是围绕解决凸包而设计的的网络结构，直接使用Attention的权重用于预测，能够适应输入的规模，后面许多网络结构应用发展成了Copying Mechanism来解决OOV问题 | Oriol Vinyals et al,2015

+ [对话系统-模型] | [A Neural Conversational Model](https://arxiv.org/pdf/1506.05869.pdf) | Seq2Seq结构的对话模型 | Oriol et al,2015
  
+ [数据集] | [The Ubuntu Dialogue Corpus: A Large Dataset for Research in Unstructured Multi-Turn Dialogue Systems](https://arxiv.org/pdf/1506.08909.pdf) | Ubuntu 非结构化多轮对话数据集 |  Ryan Lowe et al,2015
  
+ [向量召回] | [Clustering is Efficient for Approximate Maximum Inner Product Search](https://arxiv.org/pdf/1507.05910.pdf) | K-Means Tree，使用K-Means进行建树 | Alex Auvolat et al,2015
  
+ [模型] | [Training Very Deep Networks](https://arxiv.org/pdf/1507.06228.pdf) | [阅读笔记](https://cloud.tencent.com/developer/article/1148375) | 经典的Highway networks，基于深层的CNN堆叠网络，使用transform gate和carry gate（其实后来被统一称为Shortcut），将浅层特征信息带到深层中，以此来解决深度网络中梯度发散，难以训练的问题 | Rupesh Kumar Srivastava et al,2015

+ [深度学习] | [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/pdf/1508.04025.pdf) | Luong Attention的原文 | Luong et al,2015

+ [预训练-语言模型] | [Character-Aware Neural Language Models](https://arxiv.org/pdf/1508.06615.pdf) | 提供一种功能强大，功能强大的语言模型，其可编码子词相关性，同时解决先前模型的罕见字问题，使用更少的参数获得可比较的表现力 | Yoon et al,2015

+ [模型-Subword] | [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/pdf/1508.07909.pdf) | 就是我们所熟知的Byte Pair Encoding，是一种使用一些出现频率高的byte pair来组成新的byte的方法 | Sennrich et al,2015

+ [向量召回] | [Deep Compression: Ccompressing Deep Neural Networks With Pruning, Trained Quantization And Huffman Coding](https://arxiv.org/pdf/1510.00149.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/21574328) | ICLR的best paper，主要内容是关于深度学习网络参数的压缩。分为三步，Network pruning，即先训练一个网络，然后把一些权重很小的连接进行剪枝（通过阈值来决定），然后retrain网络。第二步，量化权重；第三步则是使用Huffman coding进行无损编码 | Song Han et al,2015

+ [机器学习] | [Optimal Whitening and Decorrelation](https://arxiv.org/pdf/1512.00809.pdf) | 提供五种白化方法的数学证明 | Agnan Kessy et al,2015

+ [深度学习] | [Strategies for Training Large Vocabulary Neural Language Models](https://arxiv.org/pdf/1512.04906.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/397084135) | 主要是对当时的一些Softmax和Sampling进行总结，顺便提出了Differentiated Softmax方法 | Wenlin Chen et al,2015

+ [机器学习-模型] | [XGBoost: A Scalable Tree Boosting System](https://dl.acm.org/doi/pdf/10.1145/2939672.2939785) | [阅读笔记1](https://zhuanlan.zhihu.com/p/89572181) / [阅读笔记2](https://zhuanlan.zhihu.com/p/87885678) | 本文提出的XGBoost是基于Boosting的集成算法，更确切的说，XGBoost包括了数学原理和工程实现的优化，有着包括精度更高、灵活性强、并行化计算的诸多优点。一般在一些业务场景作为baseline（在数据科学竞赛做集成更是可以无脑上一波），实现包xgboost | Tianqi Chen et al, 2016

+ [聚类] | [Approximate K-Means++ in Sublinear Time](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewFile/12147/11759) | [阅读笔记](https://www.zhihu.com/question/494753171/answer/2204649294) | K-MC2区别于k-means II的采样方法，使用MCMC采样，其主要思想是将K-Means++中的采样方法替换为基于MCMC（马尔科夫链蒙特卡洛）采样方法（MCMC的介绍可以参考：[MCMC随机采样](https://zhuanlan.zhihu.com/p/30003899)）。用MCMC的方法采样出长为M的数列，取最后（K-1）个数作为中心点初始化，target distribution是距离的函数，满足距离越远，概率越大(表达的含义同k-means++)，proposal distribution是一个常函数，1/样本数。 | Olivier Bachem et al,2016

+ [聚类] | [Fast and Provably Good Seedings for k-Means](https://proceedings.neurips.cc/paper/2016/file/d67d8ab4f4c10bf22aa353e27879133c-Paper.pdf) | [阅读笔记](https://www.zhihu.com/question/494753171/answer/2204649294) | AFK-MC2基于K-MC2改进，由于K-MC2的proposal distribution是常函数，不够鲁棒，因此AFK-MC2将与距离有关的分布作为一个term加入原始的分布中，优化proposal distribution | Olivier Bachem et al,2016

+ [模型] | [Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) [阅读笔记](https://cloud.tencent.com/developer/article/1148375) | 经典的ResNet，基于深层的CNN堆叠网络，利用了残差连接（ResNet中是跨越了2层或3层），解决深度模型中的退化问题，最优的残差结构是把BN和ReLU都提前，成为pre-activation | Kaiming He et al,2016

+ [模型-文本相似度/匹配/分类] | [Siamese Recurrent Architectures for Learning Sentence Similarity](https://scholar.google.com/scholar_url?url=https://ojs.aaai.org/index.php/AAAI/article/view/10350/10209&hl=zh-CN&sa=T&oi=gsb-gga&ct=res&cd=0&d=7393466935379636447&ei=KQWzYNL5OYz4yATXqJ6YCg&scisig=AAGBfm0zNEZZez8zh5ZB_iG7UTrwXmhJWg) | Siamese LSTM，一个用来计算句对相似度的模型 | Jonas Mueller et al,2016

+ [模型-文本相似度/匹配/分类] | [Learning Text Similarity with Siamese Recurrent Networks](https://aclanthology.org/W16-1617.pdf) | 网络包含4层BiLSTM（64-d hidden），最后一层的BiLSTM的hidden state和cell state进行concat，然后在timestep维度进行average处理，并接一个Dense层（激活函数为tanh），得到的两个Embedding Space进行Cosine sim计算，得到的相似度分数E用于损失函数计算，损失函数使用对比损失函数，计算方法为，损失函数正例：1/4(1-E)^2，负例：E^2(如果E<m)，否则0 | Paul Neculoiu et al,2016

+ [深度学习] | [Exploring the Limits of Language Modeling](https://arxiv.org/pdf/1602.02410.pdf) | CNN Softmax方法，虽然还是离不开原始的Softmax，但是换了一个视角效果很好 | Rafal Jozefowicz et al,2016

+ [深度学习] | [Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks](https://arxiv.org/pdf/1602.07868.pdf) | Weight Normalization是一种在权值维度上进行归一化的方法 | Tim Salimans et al,2016

+ [模型] | [Incorporating Copying Mechanism in Sequence-to-Sequence Learning](https://arxiv.org/pdf/1603.06393.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/415502906) | CopyNet模型，使用Copying Mechanism来缓解未登录词问题的模型，在文本摘要等生成词多含输入词的任务中，效果不错 | Jiatao Gu et al,2016
  
+ [向量召回] | [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/pdf/1603.09320.pdf) | [阅读笔记](https://blog.csdn.net/u011233351/article/details/85116719) | HNSW算法，在NSW的基础上，引入层次结构实现Expressway mechanism，达到顶层粗查，底层细查的思路 | Yu. A. Malkov et al,2016
  
+ [模型-Subword] | [Achieving Open Vocabulary Neural Machine Translation with Hybrid Word-Character Models](https://arxiv.org/pdf/1604.00788.pdf) | 一个非常出色的框架，主要是在word-level进行翻译，但是在有需要的时候可以很方便的使用Character-level的输入 | Luong et al,2016

+ [对话系统-任务型对话] | [A Network-based End-to-End Trainable Task-oriented Dialogue System](https://arxiv.org/pdf/1604.04562.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/261701071) | 非常值得一读的任务型对话模型架构 | Wen et al,2016
  
+ [深度学习] | [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/pdf/1604.06174.pdf) | [keras_recompute](https://github.com/bojone/keras_recompute) | 这篇论文整体上讲了一个以时间换空间的省显存的trick，即gradient checkpointing，通过丢弃低运算成本操作的结果，也就是把中间结果feature map 一个都不保留，全部干掉，反向传播时重新计算出来（当然，论文折中是每隔 sqrt(n)保留一个feature map）。能够把内存降低 sqrt(n) 分之一，超越大多数节省内存的奇淫巧技，具体实现可参考tf.recompute_grad，或者的一个开源实现 | Tianqi Chen et al,2016
  
+ [模型-Subword] | [Learning Character-level Representations for Part-of-Speech Tagging](http://proceedings.mlr.press/v32/santos14.pdf) | Character-level去构建word-level，该网络结构主要是对字符进行卷积以生成单词嵌入，同时使用固定窗口对PoS标记的字嵌入进行操作 | Jason et al,2016

+ [语言模型-文本相似度/匹配/分类] | [Very Deep Convolutional Networks for Text Classification](https://arxiv.org/pdf/1606.01781.pdf) | VDCNN，character level，只利用小尺度的卷积核池化操作，包含了29个卷积层。效果提升并不是很明显吧，不过亮点在于CNN层数比较深，从某种程度上证明了类似ResNet那样的Shortcut connections可以降低梯度消失带来的影响，从而提升效果 | Alexis Conneau et al, 2016
  
+ [模型-语言模型] | [A Joint Model for Word Embedding and Word Morphology](https://arxiv.org/pdf/1606.02601.pdf) | 该模型的目标与word2vec相同，但是使用的是Character-level的输入，它使用了双向的LSTM结构尝试捕获形态并且能够推断出词根 | Kris et al,2016

+ [对话系统-对话状态跟踪] | [Neural Belief Tracker: Data-Driven Dialogue State Tracking](https://arxiv.org/pdf/1606.03777.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/262289823) | NBT框架，理解Belief state和tracking的好文 | Young et al,2016
  
+ [机器学习] | [Gaussian Error Linear Units (GELUS)](https://arxiv.org/pdf/1606.08415.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/349492378) | GELU的目标就是希望在激活（拥有非线性拟合的能力）中加入正则化的思想。ReLU会确定性的将输入乘上一个0或者1，Dropout则是随机乘上0。而GELU也是通过将输入乘上0或1来实现这个功能，但是输入是乘以0还是1，是在同时取决于输入自身分布的情况下随机选择的。换句话说，是0还是1取决于当前的输入有多大的概率大于其余的输入。而由于神经元的输入x往往遵循正态分布（尤其是深度网络中普遍存在Normalization），所以GELU就可以被定义为“标准正态分布的累积分布函数”，利用erf就可以得到公式：x/2*(1+erf(x/sqrt(2))) | Dan Hendrycks et al,2016

+ [GNN-图算法-模型-无监督] | [node2vec: Scalable Feature Learning for Networks](https://arxiv.org/pdf/1607.00653.pdf) | [阅读笔记1](https://zhuanlan.zhihu.com/p/56542707) / [阅读笔记2](https://toutiao.io/posts/y84ifco/preview) | 本文提出的node2vec是一种综合考虑DFS邻域和BFS邻域的graph embedding方法，整体算法思路框架继承了DeepWalk，通过随机游走获取序列，并通过Word2Vec学习表示向量。不同于DeepWalk的是，其使用了有偏的随机游走，同时通过p和q两个参数，以alias采样的方式来控制序列游走的方向（是选择邻接节点还是二度节点） | Aditya Grover et al, 2016

+ [GNN-图算法-模型-无监督] | [Structural Deep Network Embedding](http://www.shichuan.org/hin/time/2016.%20Structural%20Deep%20Network%20Embedding.pdf) | [阅读笔记1](https://zhuanlan.zhihu.com/p/56637181) / [阅读笔记2](https://mp.weixin.qq.com/s?__biz=MzIzOTU0NTQ0MA==&mid=2247486868&idx=1&sn=c2d3e38e9ab7cc61f2a2ffaeecf0febe&chksm=e929309bde5eb98dfa657c7fd1bf7e80495f9c9ad0cde2ee78b36a1f0a453d2cc322948cb3a3&mpshare=1&scene=1&srcid=0213HJqpvPzBLS7AW7L6K3Zz#rd) | 本文提出的SDNE模型是和Node2Vec同年提出的graph embedding方法，可以看作是基于LINE方法的扩展。SDNE使用一个自动编码器结构来同时优化1阶和2阶相似度(LINE是分别优化的)，学习得到的向量表示能够保留局部和全局结构，并且对稀疏网络具有鲁棒性。通过输入的邻接矩阵和网络重构出的邻接矩阵计算一阶二阶损失函数，并配合一个正则项组成联合损失函数进行优化 | Daixin Wang et al, 2016
  
+ [模型-文本相似度/匹配/分类] | [Bag of Tricks for Efficient Text Classification](https://arxiv.org/pdf/1607.01759.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/31118235) | 比较经典的FastText，比较依赖Word Embedding的质量（槽点本身难点就在于embedding，结果文章不谈这个），整个网络结构使用N-gram，对得到的Embedding求和，并过两个Dense然后输出，本身网络结构就那没啥，当然fast啦，外加论文具体加了hashing trick，hierarchical softmax等进行加速、内存优化 | Armand Joulin et al,2016
  
+ [模型-语言模型] | [Enriching Word Vectors with Subword Information](https://arxiv.org/pdf/1607.04606.pdf) | word2vec的升级版，对于具有大量形态学的稀有词和语言有更好的表征，它也可以说是带有字符n-gram的w2v skip-gram模型的扩展 | Piotr et al,2016

+ [深度学习] | [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/258977332) | 层归一化方法，针对Batch Normalization的改进 | Jimmy et al,2016

+ [深度学习] | [Instance Normalization:The Missing Ingredient for Fast Stylization](https://arxiv.org/pdf/1607.08022.pdf) | Instance Normalization是一种不受限于批量大小的算法专门用于Texture Network中的生成器网络 | Dmitry Ulyanov et al,2016

+ [对话系统-对话意图识别-对话槽位填充] | [Attention-Based Recurrent Neural Network Models for Joint Intent Detection and Slot Filling](https://arxiv.org/pdf/1609.01454.pdf) | 使用Attention-Based的RNN模型进行联合意图识别和槽位填充，达到不错的效果 | Bing Liu et al,2016
  
+ [GNN-文本相似度/匹配/分类-图算法] | [Semi-supervised classification with graph convolutional networks](https://arxiv.org/pdf/1609.02907.pdf) | [阅读笔记1](https://zhuanlan.zhihu.com/p/78466344) / [阅读笔记2](https://www.zhihu.com/question/54504471/answer/332657604) | 这就是非常经典的GCN啦，GCN对非结构化数据构造特征节点，进而构造graph，通过使用邻接矩阵、度矩阵等图结构对输入的节点embedding进行优化学习（本质上是一种局部加权求和的方式，类似Attention的思想，不过有很多trick在里面，比如对称归一化等），能够通过相邻节点传递特征信息。GCN能够有效地提取空间特征来进行机器学习，虽然目前在NLP任务中的表现不算特别突出，但是它的功劳在于提供一种处理、研究的模型，扩广了解决方案的思路 | Thomas N. Kipf et al,2016
  
+ [深度学习] | [Efficient softmax approximation for GPUs](https://arxiv.org/pdf/1609.04309.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/397084135) | Adaptive Softmax，针对GPU的矩阵计算，实现了多倍与普通Softmax计算效率的提升，值得一看 | Edouard Grave et al,2016
  
+ [机器学习] | [An overview of gradient descent optimization algorithms](https://arxiv.org/pdf/1609.04747.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/343564175) | 对当前主流的梯度下降算法进行概述 | Sebastian Ruder et al,2016
  
+ [模型-Subword] | [Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/pdf/1609.08144.pdf) | wordpiece作为BERT使用的分词方式，其生成词表的方式和BPE非常相近，区别在于BPE选择频率最高的相邻字符对进行合并，而wordpiece是基于概率生成的 | Yonghui et al,2016

+ [模型-Subword] | [Fully Character-Level Neural Machine Translation without Explicit Segmentation](https://arxiv.org/pdf/1610.03017.pdf) | 比较经典的Character-Level的Subword算法模型 | Jason et al,2016

+ [深度学习] | [Categorical Reparameterization With Gumbel-Softmax](https://arxiv.org/pdf/1611.01144.pdf) | [阅读笔记](https://www.zhihu.com/question/422373907/answer/2260975090) | Gumbel Max由来已久，而这篇文章就是基于Gumbel Max，首次提出并应用Gumbel Softmax的。目标就是使用梯度估计的方法，来解决Categorical Distribution中，使用类似argmax操作导致网络不可微的问题。文章主要探讨了部分隐变量是离散型变量的变分推断问题，比如基于VAE的半监督学习 | Eric Jang et al,2016

+ [对话系统-检索式对话系统] | [Sequential Matching Network: A New Architecture for Multi-turn Response Selection in Retrieval-Based Chatbots](https://arxiv.org/pdf/1612.01627v2.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/270554147) | SMN检索式对话模型，多层多粒度提取信息 | Devlin et al,2016
  
+ [深度学习] | [Large-Margin Softmax Loss for Convolutional Neural Networks](https://arxiv.org/pdf/1612.02295.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/397084135) | L-Softmax在原Softmax的基础上增加了控制系数m，使得类内距离尽可能小，类间距离尽可能大 | Weiyang Liu et al,2016
  
+ [深度学习] | [An empirical analysis of the optimization of deep network loss surfaces](https://arxiv.org/pdf/1612.04010.pdf) | 论文中得出一个结论，即Batch Normalization更有利于梯度下降 | Shibani et al,2016
  
+ [模型-语言模型] | [Language Modeling with Gated Convolutional Networks](https://arxiv.org/pdf/1612.08083.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/395977833) | 受LSTM门控机制的启发，将线性门控机制应用于卷积结构，文中对比GLU、GTU等结构性能 | Yann N. Dauphin et al,2016
  
+ [语音系统-语音合成] | [Tacotron: A Fully End-To-End Text-To-Speech Synthesis Model](http://bengio.abracadoudou.com/cv/publications/pdf/wang_2017_arxiv.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/337042442) | Tacotron，端到端的语音合成系统 | Yuxuan et al,2017
  
+ [模型] | [Densely Connected Convolutional Networks](https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf) | [阅读笔记](https://cloud.tencent.com/developer/article/1148375) | CVPR 2017的Best Paper，提出了DenseNet，借鉴highway networks和ResNet的思路，DenseNet将shortcut用到了“极致”——每两层之间都添加shortcut，当然具体实现中使用了一些tricks防止模型过大的问题 | Gao Huang et al,2017
  
+ [模型-语言模型] | [A Simple But Tough-To-Beat Baseline For Sentence Embeddings](https://openreview.net/pdf?id=SyK00v5xx) | Smooth Inverse Frequency，一种简单但是效果好的Sentence Embedding方法 | Sanjeev Arora et al,2017

+ [深度学习] | [Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning](https://arxiv.org/pdf/1702.03118.pdf) | 提出SILU激活函数，其实从某种角度讲就是GELU激活的一种近似，x*sigmoid(x) | Stefan Elfwing et al,2017

+ [深度学习] | [Cosine Normalization: Using Cosine Similarity Instead of Dot Product in Neural Networks](https://arxiv.org/pdf/1702.05870v5.pdf) | Cosine Normalization是一种将unbounded的向量点积换成夹角余弦操作，从而进行归一化的方法 | Luo Chunjie et al, 2017

+ [深度学习] | [Massive Exploration of Neural Machine Translation Architectures](https://arxiv.org/pdf/1703.03906.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/328801239) | 展示了以NMT架构超参数为例的首次大规模分析，实验为构建和扩展NMT体系结构带来了新颖的见解和实用建议。 | Denny et al,2017

+ [GNN-图算法-模型-无监督] | [struc2vec: Learning Node Representations from Structural Identity](https://arxiv.org/pdf/1704.03165.pdf) | [阅读笔记1](https://zhuanlan.zhihu.com/p/63175042) / [阅读笔记2](https://zhuanlan.zhihu.com/p/56733145) | | 本文提出的struc2vec方法是用于graph embedding，相较于经典的DeepWalk和Node2Vec方法（关注存在直连节点的相似性），struc2vec如它名字一样，关注的是节点的结构相似性，这使得它可以表示两个距离很远但结构（连接度，局部拓扑结构）相似的节点。大体的流程分为四步：（1）根据不同距离的邻居信息分别算出每个节点对的结构相似度，这涉及到了不同层次的结构相似度的计算，其中使用DTW（一种动态规划方法）计算有序度序列的距离；（2）构建一个多层次的带权重网络M，每个层次中的节点皆由原网络中的节点构成，距离计算对应其层数的有序度序列的距离；（3）在M中生成随机游走，为每个节点采样出上下文；（4）使用word2vec的方法对采样出的随机游走序列学习出每个节点的节点表示 | Leonardo F. R. Ribeiro et al, 2017

+ [模型] | [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/pdf/1704.04368.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/419659043) | 结合Copying Mechanism和Coverage mechanism两种技巧的LSTM-Base模型，一定程度上解决OOV和重复词问题，经典值得一读 | Abigail See et al,2017

+ [深度学习] | [SphereFace: Deep Hypersphere Embedding for Face Recognition](https://arxiv.org/pdf/1704.08063.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/397084135) | A-Softmax，思路和L-Softmax差不多，区别是对权重进行了归一化 | Weiyang Liu et al,2017

+ [模型-语言模型] | [Supervised Learning of Universal Sentence Representations from Natural Language Inference Data](https://arxiv.org/pdf/1705.02364.pdf) | InferSent，通过不同的encoder得到Sentence Embedding，并计算两者差值、点乘得到交互向量，从而得到相似度 | Alexis Conneau et al,2017

+ [对话系统-对话意图识别] | [Latent Intention Dialogue Models](https://arxiv.org/pdf/1705.10229.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/263034049) | 离散潜在变量模型学习对话意图的框架 | Wen et al,2017
  
+ [模型-预训练-语言模型] | [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/250946855) | Transformer的开山之作，值得精读 | Ashish et al,2017

+ [社区发现-综述] | [Network Community Detection: A Review and Visual Survey](https://arxiv.org/ftp/arxiv/papers/1708/1708.00977.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/141401358) | 一篇关于社区发现的综述，文章对社区发现概念和发展进行了介绍，并对当下社区发现算法按照分类进行了细致的介绍，包括传统的社区发现技术、基于分裂的社区发现技术、基于模块化优化的社区发现技术、重叠社区发现技术、动态社区发现技术等 | Bisma S. Khan et al, 2017
  
+ [深度学习] | [ProjectionNet: Learning Efficient On-Device Deep Networks Using Neural Projections](https://arxiv.org/pdf/1708.00630.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/295636122) | 一种叫ProjectionNet的联合框架，可以为不同机器学习模型架构训练轻量的设备端模型。 | Google et al,2017

+ [深度学习-损失函数] | [Focal Loss for Dense Object Detection](https://arxiv.org/pdf/1708.02002.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/260407405) | 分类问题我们一般使用交叉熵损失函数，交叉熵损失函数是平等对待正负样本的，当正负样本不均衡的时候，正样本总的损失会淹没小量负样本总的损失，导致模型最终的学习方向并不会把重点放在负样本上，解决方案就是增加负样本的权重，减少正样本的权重，从而是模型重点倾向于学习负样本的规律。这种方式还是不能解决easy/heard samples的问题，Focal loss对交叉熵损失函数增加了一个调制因子，实现对easy samples的降权，从而使模型训练的损失可以集中在比较难学习的负样本上 | Tsung-Yi Lin et al,2017
  
+ [对话系统-任务型对话-对话状态跟踪] | [An End-to-End Trainable Neural Network Model with Belief Tracking for Task-Oriented Dialog](https://arxiv.org/pdf/1708.05956.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/260345363) | 面向任务的对话系统的新型端到端可训练神经网络模型 | Liu et al,2017
  
+ [数据集] | [DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset](https://arxiv.org/pdf/1710.03957.pdf) | [数据集地址](https://drive.google.com/file/d/1sj3Z_GZfYzrhmleWazA-QawhUEhlNmJd/view?usp=sharing) | 包含对话意图和情感信息的多轮对话数据集 | Yanran Li et al, 2017
  
+ [机器学习] | [Swish: A Self-Gated Activation Function](https://arxiv.org/pdf/1710.05941v1.pdf) | 提出的Swish激活函数，通SILU激活函数一样，没啥差别，x*sigmoid(x) | Prajit Ramachandran et al,2017
  
+ [综述-对话系统] | [A Survey on Dialogue Systems:Recent Advances and New Frontiers](https://arxiv.org/pdf/1711.01731.pdf) | 对话系统的最新研究和方向 | Chen et al,2017

+ [语音系统-语音合成] | [Natural TTS Synthesis By Conditioning Wavenet On Mel Spectrogram Predictions](https://arxiv.org/pdf/1712.05884.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/337042442) | Tacotron2，相较于Tacotron有着更好的性能，使用WaveNet作为Vocoder | Jonathan et al,2017

+ [异常检测-机器学习] | [XGBOD: Improving Supervised Outlier Detection with Unsupervised Representation Learning](https://arxiv.org/ftp/arxiv/papers/1912/1912.00290.pdf) | 本文主要是提出一种通过无监督算法来增广特征，进而融合原始特征用于XGB的有监督的训练思路，命名为XGBOD。在ADBench中，半监督的总体效果来讲好于直接使用XGB，监督学习的情况下，指标提升差不了多少（然后XGBOD在训练代价上更大），因此在有监督的情况下，直接使用XGB作为baseline更加简单直接一些 | Yue Zhao et al, 2018

+ [数据集] | [LCQMC: A Large-scale Chinese Question Matching Corpus](https://aclanthology.org/C18-1166.pdf) | LCQMC，开放域的中文语义相似度语料，更加侧重于intent相似，总共26万的文本对 | Xin Liu et al,2018

+ [数据集] | [The BQ Corpus: A Large-scale Domain-specific Chinese Corpus For Sentence Semantic Equivalence Identification](https://aclanthology.org/D18-1536.pdf) | 关于Bank Question的中文语义相似度语料，总共12万的文本对 | Jing Chen et al,2018

+ [聚类] | [Robust and Rapid Clustering of KPIs for Large-Scale Anomaly Detection](https://netman.aiops.org/~peidan/ANM2018/8.DependencyDiscovery/LectureCoverage/2018IWQOS_ROCKA.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/50698719) | 关于快速时序聚类的文章，提出ROCKA系统架构，包括了数据预处理、基线提取、相似性度量、基于密度的聚类算法。ROCKA算法仅仅是使用了派发策略，单是并未在有效的利用过程中的计算结果，导致在派发过程中复杂度较高 | Zhihan Li et al,2018

+ [对话系统-检索式对话系统] | [Multi-Turn Response Selection for Chatbots with Deep Attention Matching Network](https://www.aclweb.org/anthology/P18-1103.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/306846122) | DAM检索式对话模型，完全基于注意力机制的多层多粒度提取信息 | Xiangyang et al,2018
  
+ [对话系统-对话意图识别-对话槽位填充] | [Slot-Gated Modeling for Joint Slot Filling and Intent Prediction](https://aclanthology.org/N18-2118.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/413261222) | 提出了Slot-Gated机制，联合意图识别和槽位填充效果提升 | Chih-Wen Goo et al,2018
  
+ [模型-语言模型-无监督] | [Unsupervised Random Walk Sentence Embeddings: A Strong but Simple Baseline](https://www.aclweb.org/anthology/W18-3012.pdf) | Unsupervised Smooth Inverse Frequency，USIF改进SIF对句向量长度敏感，在相似度任务上提升很大 | Kawin Ethayarajh Arora et al,2018
  
+ [深度学习] | [Fixing Weight Decay Regularization in Adam](https://openreview.net/pdf?id=rk6qdGgCZ) | [原英文版阅读笔记](https://www.fast.ai/2018/07/02/adam-weight-decay/) | [阅读笔记](https://zhuanlan.zhihu.com/p/39543160) | 论文提出Adam在算法实现上的改进方法--AdamW（注意是算法实现）。Adam相较于传统的GD算法来说，增加了一阶动量（各时刻方向的指数移动平均值）和二阶动量（历史梯度平方和），在算法库的具体实现中，一般是通过在计算梯度之初就加上了正则项，这就导致这个正则项随着梯度一同计算，而AdamW的做法则是在梯度计算完之后，在加上这个正则项（称为weight decay）。论文中比较了SGD和SGDW、Adam和AdamW，通过实验证明了weight decay相较于一般实现的l2正则效果更好 | Anonymous authors et al, 2018
  
+ [深度学习] | [Additive Margin Softmax for Face Verification](https://arxiv.org/pdf/1801.05599.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/397084135) | AM-Softmax在A-Softmax的最大区别是AM是角度距离，A是余弦距离

+ [预训练-语言模型] | [Deep contextualized word representations](https://arxiv.org/pdf/1802.05365.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/51679783) | ELMo模型原paper，想了想还是放在预训练里吧。ELMo模型很经典了，在Transformer这个大杀器提出后，依旧以LSTM为核心结构提出新的SOTA语义编码结构，还是尤其独到之处（ps：同年BERT也被提出了）。ELMo的结构可以分为两侧各一个多层LSTM，左侧的多层LSTM负责编码文本的正向语义，右侧的负责编码反向语义，然后对左右两边每一层的输出进行concat并乘上一个权重，最后的ELMo向量就是每一层输出的和。ELMo最大的亮点就是编码了文本的双向语义，因此相对于一些单向、静态编码器来讲，效果更好，但是问题也在这，这种将正反向的语义分开编码方式，就比不上BERT这种融合式的双向编码了，事实上也证明了这一点 | Matthew E. Peters et al,2018

+ [深度学习] | [Self-Attention with Relative Position Representations](https://arxiv.org/pdf/1803.02155.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/268649069) | 对Transformer里面用到的位置编码进行讨论，对自注意力进行改造，从而使用相对位置编码代替硬位置编码 | Mihaylova et al,2018

+ [深度学习] | [Group Normalization](https://arxiv.org/pdf/1803.08494.pdf) | Group Normalization是将输入的通道分成较小的子组，并根据其均值和方差归一化这些值 | Yuxin Wu et al,2018

+ [语音系统-语音识别-预训练] | [Syllable-Based Sequence-to-Sequence Speech Recognition with the Transformer in Mandarin Chinese](https://arxiv.org/pdf/1804.10752.pdf) | 使用Transformer应用在普通话语音识别，数据集是HKUST datasets  |  Shiyu et al,2018
  
+ [模型-Subword] | [Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates](https://arxiv.org/pdf/1804.10959.pdf) | unigram在给定词表及对应概率值下，直接以最大化句子的likelihood为目标来直接构建整个词表 | Kudo et al,2018

+ [对话系统-对话状态跟踪] | [Global-Locally Self-Attentive Dialogue State Tracker](https://arxiv.org/pdf/1805.09655.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/266982344) | 全局-局部自注意力状态跟踪 | Zhong et al,2018
  
+ [深度学习] | [How Does Batch Normalization Help Optimization?](https://arxiv.org/pdf/1805.11604.pdf) | 讨论Batch Normalization是如何帮助优化器工作的，主要结论是BN层能够让损失函数更加平滑 | Shibani et al,2018
  
+ [模型-对话系统] | [Multi-Cast Attention Networks for Retrieval-based Question Answering and Response Prediction](https://arxiv.org/pdf/1806.00778.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/349369847) | 一种用于通用序列对建模的整体架构，结合多种注意力机制进行特征增强 | Yi Tay et al,2018

+ [对话系统-数据增强] | [Sequence-to-Sequence Data Augmentation for Dialogue Language Understanding](https://arxiv.org/pdf/1807.01554.pdf) | 使用seq2seq生成模型对语义文本进行数据增强，核心步骤为Delexicalisation->Diversity rank->generation->surface realisation | Yutai Hou et al,2018
  
+ [模型] | [Sliced Recurrent Neural Networks](https://arxiv.org/ftp/arxiv/papers/1807/1807.02291.pdf) | 切片RNN网络，尝试突破RNN时序限制的模型 | Zeping Yu et al,2018
  
+ [模型-文本相似度/匹配/分类-GNN-图算法] | [Graph Convolutional Networks for Text Classification](https://arxiv.org/pdf/1809.05679.pdf) | 将GCN应用于文本分类中，在不引入预训练模型的情况下，该方法的表现已经很优异了。该方法将每个独立的单词以及文档作为节点，即graph中包含单词级别和文档级别两类节点。初始化单词one-hot（不使用训练向量）。对于边，则包含（文档-单词）、（单词-单词）两类边，其中（文档-单词）使用tf-idf进行度量，（单词-单词）使用PMI指数。本文的模型结构的缺点在于，只考虑到共现度方面的信息，因此语义方面很低（作者原意就是不使用预训练embedding），而且可能会受到长尾问题的影响，因此可以使用注意力来辅助提升 | Liang Yao et al, 2018
  
+ [语音系统-语音合成] | [Neural Speech Synthesis with Transformer Network](https://arxiv.org/pdf/1809.08895.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/332316226) | 本文受Transformer启发，使用多头自注意力机制取代Tacotron2中的RNN结构和原始注意力机制。 | Naihan et al,2018

+ [预训练-语言模型] | [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/269997771) | 顶顶大名的BERT，单独抽离Transformer的Encoder架构，并提出NSP、MLM预训练方式，也是基于此，是的BERT拥有强大的表征能力，并用于下游相关任务 | Devlin et al,2018

+ [深度学习] | [RelGAN: Relational Generative Adversarial Networks For Text Generation](https://openreview.net/pdf?id=rJedV3R5tm) | [阅读笔记](https://zhuanlan.zhihu.com/p/87605995) | 提出了新型的生成器和判别器结构，使得直接用Gumbel Softmax训练出的文本GAN大幅度超过了以往的各种文本GAN模型。主要由三个模块组成，分别是：在生成器上，利用relational memory，使得具有更强表达能力和在长文本上更好的模型能力；在离散数据上，训练GAN利用Gumbel-Softmax Relaxation模型，使得模型简化，替代强化学习启发式算法；在判别器上利用多层词向量表示，使得生成器往更具多样性方面更新 Weili Nie et al, 2019

+ [异常检测-综述] | [Deep Learning for Anomaly Detection: A Review](https://arxiv.org/pdf/2007.02500.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/419161328) | 本篇综述将异常检测归纳到三个框架（deep learning generic feature extraction、learning representations of normality、end-to-end anomaly score learning），共十一种类别中，并对每个类别的目标、流程、优缺点等进行了详细的阐述。最后给出了代表性的算法和数据集，并分析了当下和未来的发展方向，是一篇非常值得一读的异常检测综述 | Guansong Pang et al, 2019

+ [机器学习] | [Covariate Shift: A Review and Analysis on Classifiers](https://ieeexplore.ieee.org/abstract/document/8978471) | [阅读笔记](https://zhuanlan.zhihu.com/p/339719861) | 通过几种分类算法，在四种不同的数据集下验证几种方法处理Covariate Shift问题后的性能分析 | Geeta et al, 2019

+ [深度学习] | [Language Models as Knowledge Bases?](https://aclanthology.org/D19-1250.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/94470840) | 一篇讨论性的文章，主要提出的观点是认为像BERT等类似的预训练语言模型是另一种形式的knowledge database，能够保存大量的知识信息。作者通过效仿MLM的方式，将已有的NLP数据集进行整合，并挖空成完形填空的提问形式（以推理实体关系），文中实验了两种单向语言模型和两种双向语言模型，结果显示预训练模型的确包含了知识库中的信息。ps：这种观点也不一定对的，也有文章反对说BERT等模型只是由于对实体名称（表面形式）进行推理，如果过滤掉一些容易猜测的事实，那么模型精度将会急剧下降 | Fabio Petroni et al, 2019

+ [深度学习-预训练] | [What does BERT learn about the structure of language?](https://hal.inria.fr/hal-02131630/document) | [阅读笔记](https://zhuanlan.zhihu.com/p/74515580) | 本文主要是通过一些实验来补充验证BERT的不同层学习到的信息（具体没啥新结论，只是补充验证而已）。BERT的底层学习到的主要是token的表层信息，中层学习到的是语言学特征信息（句法结构之类的），顶层学习到的是语义特征信息。文中还进一步探索了BERT能够学习到组合结构的特征，使用了Tensor Product Decomposition Networks（TPDN）来设计实验，从自注意力机制的权重中推导出对应的依赖树，印证了BERT的组合建模方式和传统的句法分析相似 | Ganesh Jawahar et al,2019

+ [模型] | [Pay Less Attention With Lightweight And Dynamic Convolutions](https://arxiv.org/pdf/1901.10430.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/396143249) | 论文研究Lightweight、Dynamic Convolutions，卷积结构同样能够达到和Self-Attention媲美的效果 | Felix Wu et al,2019

+ [蒸馏-预训练-语言模型] | [Distilling Task-Specific Knowledge from BERT into Simple Neural Networks](https://arxiv.org/pdf/1903.12136.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/351319938) | 对BERT进行蒸馏，训练一个TextCNN模型，相比于直接使用BERT，TextCNN虽然有一定的损失，但是参数量和速度都大大提升。本文在知识蒸馏的方式上没有特别的创新，核心点在于（1）添加了额外的逻辑回归的目标，在标注数据下，hard label的交叉熵+teacher 模型的logits的MSE；在无标注数据下，teacher模型的softlabel的交叉熵+teacher模型的logits的MSE（2）数据增强，提出了多种方法；随机mask一个token；pos tag替换；n-gram sampling；mask_prob，执行mask增强，mask_prob << pos_prob，执行pos替换，最后执行n-gram sampling | Raphael Tang et al,2019
  
+ [深度学习] | [On the Convergence of Adam and Beyond](https://arxiv.org/pdf/1904.09237.pdf) | [原英文版阅读笔记](https://www.fast.ai/2018/07/02/adam-weight-decay/) | [阅读笔记](https://zhuanlan.zhihu.com/p/39543160) | Amsgrad，ICLR2018的最佳论文，主要是算法证明Adam在收敛性上存在的缺陷，并设计了理论实验，证明了这一点，同时提出了很简单的优化方法（实际的算法实现中，这个优化方法在相当多的实验中效果并不好）。Adam的收敛性缺陷在于，学习率通常是恒定的或降低的，所以随着训练过程的进行，二阶动量会随之减少，所以具体做法是增加一个变量来记录最大值，使用这个二阶动量的最大值替换原来的二阶动量进行计算，即v = max(avg_squared, max_squared) | Sashank J. Reddi et al, 2019
  
+ [预训练-语言模型-文本相似度/匹配/分类] | [Poly-encoders: Transformer Architectures and Pre-training Strategies for Fast and Accurate Multi-sentence Scoring](https://arxiv.org/pdf/1905.01969v2.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/119444637) | Poly-encoder主要的出发点就是想要保持Bi-encoder的推理速度的同时，兼顾Cross-encoder精准匹配的潜力。思想上比较好理解，Bi-encoder的主要问题在于它要求encoder将query的所有信息都塞进一个固定的比较general的向量中，而Cross-encoder为了得到更加均衡的语义表示，需要将句子对关联起来进行推理计算，导致在检索时速度极慢。因此Poly-encoder的方案就是每个query产生m个不同的vec，利用这m个vec动态的和candidate vec计算，得到最终的final_vec（作为query的最终表示），用final_vec和candidate vec进行计算得到分数 | Samuel Humeau et al,2019
  
+ [预训练-语言模型-文本相似度/匹配/分类] | [How to Fine-Tune BERT for Text Classification?](https://arxiv.org/pdf/1905.05583.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/358516009) | BERT在Text Classification上的一些微调实验 | Xipeng Qiu et al,2019

+ [预训练-对话系统] | [Pretraining Methods for Dialog Context Representation Learning](https://arxiv.org/pdf/1906.00414.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/240742891) | 作者列举了四种针对对话上下文表示的预训练方法，其中两种是作者新提出的 | Shikib et al,2019

+ [深度学习] | [Scheduled Sampling for Transformers](https://arxiv.org/pdf/1906.07651.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/267146739) | 在Transformer应用Scheduled Sampling | Mihaylova et al,2019

+ [预训练-语言模型] | [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/pdf/1906.08237.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/361737484) | XLNet--自回归语言模型的复兴，30多项任务超越BERT | Zhilin Yang et al,2019

+ [机器学习] | [Monte Carlo Gradient Estimation in Machine Learning](https://arxiv.org/pdf/1906.10652.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/110588068) | 本文是一篇关于Monte Carlo gradient estimation的survey，本文主要总结的内容是：随机梯度估计方法的相关背景知识，包括蒙特卡洛采样和随机优化；几种经典应用，包括变分推断、强化学习中的Policy gradient、敏感性分析、实验设计；两类经典的梯度估计算法 | Shakir Mohamed et al,2019

+ [预训练-语言模型] | [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/pdf/1907.11692.pdf) | 论文发现原BERT的预训练并不充分，因此作者提出了四点Bert的改进：1）：使用更大的batch在更大的数据集上对Bert进行深度训练；2）：不在使用NSP(Next Sentence Prediction)任务；3）：使用更长的序列进行训练；4）：动态改变训练数据的MASK模式；其中动态MASK就是在每次数据输入的时候进行MASK，而不是在数据预处理的时候就预先MASK好，这种方式相当于不重复看数据，使模型学习到更多的pattern | Yinhan Liu et al,2019

+ [模型-文本相似度/匹配/分类] | [Simple and Effective Text Matching with Richer Alignment Features](https://arxiv.org/pdf/1908.00300.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/84703949) | 本文模型主打的是参数少，推理速度快（实际复现也确实很快，效果也不错）。模型的结果不复杂，采用对称结构，使用Encoder、Alignment、Fusion三个模块组成的block（模型是多block结构）进行Representation，其核心应该是对于网络中三个向量的使用，residual vectors, embedding vectors 和 encoded vectors。全文的模型结构不复杂，效果不错，值得一试的模型 | Runqi Yang et al,2019

+ [预训练-语言模型-文本相似度/匹配/分类] | [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/pdf/1908.10084.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/113133510) | 和之前提出的InferSent结构如出一辙，妥妥的双塔结构，只是换成了Bert来进行特征表示。模型结构没有什么创新点，但是这个结构用起来效果挺好，速度也快，很适合工业界使用。论文中在针对句向量表示计算策略分别使用了CLS向量策略、平均池化策略、最大值池化策略三个策略进行实验，实验结果中平均池化策略最优 | Nils Reimers et al,2019

+ [对话系统-数据增强] | [Data Augmentation with Atomic Templates for Spoken Language Understanding](https://arxiv.org/pdf/1908.10770.pdf) | 使用Atomic Templates（act-slot-value）进行对话数据增强，使用seq2seq生成模型进行语句生成 | Zijian Zhao et al,2019
  
+ [预训练-语言模型] | [NEZHA: Neural Contextualized Representation For Chinese Language Understanding](https://arxiv.org/pdf/1909.00204.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/100044919) | 具体来说没有啥特别的创新点吧，在位置编码部分改成了相对位置编码。其他的比如WWM、混合精度训练、优化器自适应学习率，都是屡见不鲜的东西，整体效果而言也没有令人惊艳 | Junqiu Wei et al,2019

+ [预训练-语言模型] | [CTRL: A Conditional Transformer Language Model For Controllable Generation](https://arxiv.org/pdf/1909.05858.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/405493225) | CTRL语言模型，提供Control Code进行定向文本生成，相较于GPT可对文本风格进行控制 | Keskar et al,2019

+ [语音系统] | [A Comparative Study on Transformer vs RNN in Speech Applications](https://arxiv.org/pdf/1909.06317.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/309390439) | Transformer应用在语音领域上与RNN对比的论文，并在ESPnet上面开源了模型代码 | Nanxin et al,2019

+ [蒸馏-预训练-语言模型] | [TinyBERT: Distilling BERT for Natural Language Understanding](https://arxiv.org/pdf/1909.10351.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/84827596) | 本文提出的TinyBERT模型大小只有BERT的13.3%，推理速度是BERT的9倍，效果下降了2-3个点左右。具体细节分为基础蒸馏：（1）Embedding蒸馏，先用权重矩阵转换一下T模型的Embedding，然后在计算两个模型的Embedding的MSE loss（2）attention层蒸馏，计算S模型和T模型的单个头Attention的MSE loss；（3）hidden层蒸馏，同Embedding的蒸馏方式，先用权重矩阵转换一下；（4）Prediction层蒸馏，计算T模型输出的logits和S模型输出 logits的交叉熵，加一个temperature控制平滑。在训练的时候，分成了两段式学习框架，包含通用蒸馏和特定于任务的蒸馏，就是分别在通用语料和在下游任务语料上分别蒸馏 | Xiaoqi Jiao et al,2019

+ [预训练-语言模型] | [ALBERT: A Lite BERT For Self-superpised Learning Of Language Representations](https://arxiv.org/pdf/1909.11942.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/84273154) | Albert大大缩小了模型参数量，并且加快了训练速度，增加了模型效果。其主要对BERT做了3点改进，第一：把embedding size(E)和hidden size(H)分开可以更高效地利用参数，因为理论上存储了context信息的H要远大于E。第二：跨层参数共享，就是不管12层还是24层都只用一个transformer。第三：使用Inter-sentence coherence loss，即SOP(sentence order prediction) | Zhenzhong Lan et al,2019

+ [蒸馏-预训练-语言模型] | [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/pdf/1910.01108.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/348244612) | 本文提出的DistilBERT，相比BERT减少40%大小，保持了97%的精度，快了60%，结果还是很棒的。DistilBERT实在预训练阶段进行蒸馏：（1）S模型结构方面，和原始BERT保持一致，token-type和pooler去掉了，然后block layer数量只有T模型的一般（两层蒸一层）；（2）在损失设计方面，分为原语言模型的loss（MLM）+蒸馏loss（就是T和S输出的logits的交叉熵）+cos距离loss（T和S在block layer间的hidden对齐）。一些小tricks方面，S模型的初始化选自T模型每两层中的一层；Batch size竟可能的大；引用RoBERTa的优化策略，动态mask | Victor SANH et al,2019

+ [对话系统-对话意图识别-数据增强] | [A Closer Look At Feature Space Data Augmentation For Few-Shot Intent Classification](https://arxiv.org/pdf/1910.04176.pdf) | 针对SLU的Intent分类任务，对其文本数据进行数据增强并比较效果，其中Linear+Transfer learning效果最佳 | Varun Kumar et al,2019

+ [异常检测-半监督] | [Deep Weakly-supervised Anomaly Detection](https://arxiv.org/pdf/1910.13601.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/431687085) | 本文提出的PReNet或PRO模型结构上不复杂，通过双塔结构的全连接层（也可以是其他的特征表示层）融合特征，最后通过全连接层缩放维度输出回归分数。本文主要在于对标签数据的组合训练策略，即成对关系预测任务。通过对带标签的异常样本和未带标签的样本进行组合，构成三个类型：两个样本都是已知的异常样本（标签为较大的分值8）、两个样本都是未标记的样本（可能是正常样本，也可能是未知的异常样本，标签为较小的分值0）、两个样本中一个是已知异常样本另一个是未标记样本（标签为中等的分值4）。通过这样做，预测值可以被定义为这些样本对的异常分数，并使用MAE进行训练。预测时，分别从标记异常数据和未标记数据中随机取等量样本，和预测样本进行计算分数 | Guansong Pang et al, 2019

+ [预训练-语言模型] | [CogLTX: Applying BERT to Long Texts](https://proceedings.neurips.cc/paper/2020/file/96671501524948bc3937b4b30d0e57b9-Paper.pdf) | 本文主要介绍如何优雅且有效的的使用BERT处理长文本。一般用BERT处理长文本的方式有截断法、Pooling法、压缩法，本文介绍的就是压缩法的一种（三种效果最好的）。从直观的角度来讲，长文本中的核心语义可以由某个短文本替换（相当于长句总结），因此需要找到这个短文本。具体的做法就是（1）首先使用动态规划算法将长文本划分长文本块；（2）然后使用一个叫做MemRecall的模块对这些块进行打分（本质上是concat），从而选出分数最高的子句组成短文本；（3）然后再用这个短文本用于后续的NLP任务。总结来讲就是COGLTX相当于使用了了两个bert，MemRecall中bert就是负责打分，另一个bert执行原本的NLP任务 | Ming Ding et al,2020
  
+ [数据集] | [Improving Dialog Evaluation with a Multi-reference Adversarial Dataset and Large Scale Pretraining](https://scholar.google.com/scholar_url?url=https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl_a_00347/1923874/tacl_a_00347.pdf&hl=zh-CN&sa=T&oi=gsb-gga&ct=res&cd=0&d=13355199831609160829&ei=hXzkYNupCsyO6rQPkrG1wAo&scisig=AAGBfm39FeIrjR-BGf074wiUqDueImjYeA) | [数据集地址](https://github.com/iitmnlp/Dialogue-Evaluation-with-BERT) | DailyDialog数据集的升级版，11K的多轮对话上下文，每个上下文包括五个标准的参考回复、五个不相关的回复、五个随机挑选的回复 | Ananya B. Sai et al, 2020
  
+ [模型-预训练] | [Reformer: The Efficient Transformer](https://arxiv.org/pdf/2001.04451.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/411882151) | 使用LSH Attention、Reversible layers、Chunking FFN layers，降低Transformer计算复杂度和内存空间消耗 | Nikita Kitaev et al,2020

+ [Prompt-预训练-语言模型-文本相似度/匹配/分类] | [Exploiting Cloze Questions for Few Shot Text Classification and Natural Language Inference](https://arxiv.org/pdf/2001.07676.pdf) | 比较早研究Prompt的工作之一，PET使用了基于手工设计模板的Prompt进行训练，对无标签数据使用了简单的prompt ensemble，即将多种prompt集成在一起计算预测结果，按不同pattern的模型acc对应权重对所有的预测进行归一化，作为soft label蒸馏一个最终模型。PET在计算loss的时候主要计算目标词的cross entropy（MLM loss作为附加，Loss=(1-a)*L_CE+a*L_MLM），而忽略了词表中其他备选词，这种方式在后续的工作当中是被认为不妥的，还是使用原生的MLM loss更好。论文还在PET的基础上，提出了迭代式的PET训练，即iPET。其实就是进行多代交叉的蒸馏，随机选取每一代的模型为无标签数据进行标记，并基于此进一步训练下一代的模型，最终和PET一样，用不同模型标注的无标签数据进行预测，蒸馏一个统一的模型。PET这种手动设计Prompt的方式本身难度较大，而且是基于人为经验的，这种方式使得模型比较依赖prompt，导致稍加改动就影响模型性能，因此后续工作也朝着auto prompt方向发展 | Timo Schick et al,2020

+ [深度学习] | [Consistency of a Recurrent Language Model With Respect to Incomplete Decoding](https://arxiv.org/pdf/2002.02492.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/349675973) | 讨论Seq2Seq模型解码停不下来的原因 | Sean Welleck et al,2020

+ [深度学习] | [GLU Variants Improve Transformer](https://arxiv.org/pdf/2002.05202.pdf) | 本文借助门控线性单元(Gated Linear Unit,GLU)对模型的FeedForward层进行了修改，同时在训练的时候去掉了Dropout，并增加了解码器输出端的Embedding（这些改动增加了模型参数，但效果更佳）。文中主要对比了Bilinear、relu、gelu、swish激活函数下，使用GLU的效果，其中gelu和swish表现最佳。总得来说，实验证明了GLU的有效性，可以应用在模型里试试 | Noam Shazeer et al,2020

+ [数据集] | [CrossWOZ: A Large-Scale Chinese Cross-Domain Task-Oriented Dialogue Dataset](https://arxiv.org/pdf/2002.11893.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/259861746) | 第一个大规模的中文跨域任务导向对话数据集 | Qi Zhu et al,2020

+ [综述-对话系统-任务型对话] | [Recent Advances and Challenges in Task-oriented Dialog Systems](https://arxiv.org/pdf/2003.07490.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/260194067) | 面向任务型对话系统的最新研究和方向 | Zhang et al,2020

+ [深度学习] | [PowerNorm: Rethinking Batch Normalization in Transformers](https://arxiv.org/pdf/2003.07845.pdf) | 对于Transformer中BN表现不好的原因做了一定的empirical和theoretical的分析 | Sheng Shen et al,2020

+ [综述-预训练] | [Pre-trained Models for Natural Language Processing: A Survey](https://arxiv.org/pdf/2003.08271.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/352152573) | 超详细的NLP预训练语言模型总结清单 | Xipeng Qiu et al,2020

+ [预训练-语言模型] | [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/pdf/2003.10555.pdf) | [阅读笔记](https://kexue.fm/archives/7846#how_to_cite) | BERT使用MLM进行训练，而ELECTRA考虑到BERT的MLM模型随机选择一部分Token进行Mask的这个操作过于简单了，想要增加一下它的难度。所以它借鉴了GAN的思想，用普通的方式训练一个MLM模型（生成器），然后根据MLM模型对输入句子进行采样替换，将处理后的句子输入到另外一个模型（判别器）中，判断句子哪些部分是被替换过的，哪些部分是被没被替换的。生成器和判别器是同步训练的，因此随着生成器的训练，判断难度会慢慢增加，直观想象有利于模型学到更有价值的内容。最后只保留判别器的Encoder来用，生成器一般就不要了。由于这种渐进式的模式使得训练过程会更有针对性，所以ELECTRA的主要亮点是训练效率更高了 | Kevin Clark et al,2020

+ [数据集] | [MuTual: A Dataset for Multi-Turn Dialogue Reasoning](https://arxiv.org/pdf/2004.04494.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/282843192) | MuTual 数据集，用于针对性地评测模型在多轮对话中的推理能力 |  L Cui et al,2020

+ [对话系统-检索式对话系统] | [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/pdf/2004.04906.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/403589222) | DPR一种高效的开放域问答检索技术，应用了BERT进行编码 | Karpukhin et al,2020
  
+ [预训练-语言模型-对话系统-任务型对话] | [TOD-BERT: Pre-trained Natural Language Understanding for Task-Oriented Dialogue](https://arxiv.org/pdf/2004.06871.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/377845426) | 任务导向型对话的预训练自然语言理解模型 | Chien-Sheng Wu et al,2020

+ [深度学习] | [Shortcut Learning in Deep Neural Networks](https://arxiv.org/pdf/2004.07780.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/421175552) | 对Shortcut Learning问题进行比较详细的解释和剖析，虽然最后没有给出实际的解决方案（Shortcut Learning问题本身就没有一个体系化的策略，需要根据实际任务而定），不过提供了几种解决的视角 | Robert Geirhos et al,2020

+ [预训练-语言模型-文本相似度/匹配/分类] | [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/pdf/2004.12832.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/376475610) | 和前面的Poly-encoder出发点都是一样的，为了就是在获得BERT representation能力的同时，提高文本计算的效率。按照本文的说法，就是尽可能离线计算好Embedding，在通过Late Interaction的方式，弥补因为query和doc分离计算导致的效果损失。本文具体的模型结构是使用原生的BERT，对query和doc进行Embedding，不同之处是为了区分query和doc，分别在输入的seq的起始位置加上[Q]和[D]。Bert是编码器，CNN做维度变换，用来对BERT的隐层输出进行降维处理，Normalize是为后面计算余弦相似度做l2正则化处理，对于doc加个标点符号的mask | Omar Khattab et al,2020

+ [综述-文本相似度/匹配/分类] | [Evolution of Semantic Similarity - A Survey](https://arxiv.org/pdf/2004.13820.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/138636605) | 一篇语义相似度的综述，整体文章从数据集开始，将技术体系分为：基于知识的语义相似性方法、基于语料的语义相似性方法、基于深度神经网络的方法、基于混合模型方法四类进行分析 | Dhivya Chandrasekaran et al,2020

+ [模型-预训练-语言模型] | [Synthesizer: Rethinking Self-Attention for Transformer Models](https://arxiv.org/pdf/2005.00743.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/380602965) | 在Transformer架构下，对Self-Attention计算的探索研究，看完会对Self-Attention有个新认识 | Yi Tay et al,2020
  
+ [综述-文本相似度/匹配/分类] | [Measurement of Text Similarity: A Survey](https://scholar.google.com/scholar_url?url=https://www.mdpi.com/2078-2489/11/9/421/pdf&hl=zh-CN&sa=T&oi=gsb-gga&ct=res&cd=0&d=6927655514883966546&ei=Ftg9Yqm4D4TQmAGcuKOgCw&scisig=AAGBfm0m8ZuluCOz6UpEoMRcxqYN9oQl8A) | 语义相似度的综述，大体上从独立度量到模型计算的模型概述的比较广，但不是很全，不过从了解相似度计算来讲，还是值得一看的 | Jiapeng Wang et al,2020

+ [深度学习] | [Beyond Accuracy: Behavioral Testing of NLP Models with CheckList](https://arxiv.org/pdf/2005.04118.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/421175552) | ACL2020的Best Paper，基于NLP领域提出了测试体系来指导我们了解 NLP 模型的能力，也能够指导我们去理解问题、解决问题。不同于现代 NLP 模型常常仅关注特定的任务，CheckList 希望去评估一个模型的多方面能力，这些能力有的是模型通用的，有的则是面向特定的任务或领域 | Marco Tulio Ribeiro et al,2020

+ [预训练-语言模型] | [DeBERTa: Decoding-Enhanced Bert With Disentangled Attention](https://arxiv.org/pdf/2006.03654.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/348522530) | DeBERTa的一大亮点在于改动位置编码的介入时机，在论文中叫作Disentangled Attention。具体做法是将原本和输入embedding混合相加的pos embedding（relative）单独拎出来，然后再用位置编码和content 编码计算attention，进而增加了“位置-内容” 和 “内容-位置” 注意力的分散Disentangled Attention。然后一些其他的改动比如：1) | 因为我们在精调时一般会在 BERT 的输出后接一个特定任务的 Decoder，但是在预训练时却并没有这个 Decoder，所以本文在预训练时用一个两层的 Transformer decoder 和一个 SoftMax 作为 Decoder；2) | 为了弥补一下只有相对位置的损失，因此在decoder前加入一层绝对位置embedding；3) | bert的训练策略中，mask有10%的情况是不做任何替换，而DeBeta将不做替换改成了换位该位置词绝对位置的pos embeding | Pengcheng He et al,2020

+ [对话系统-阅读理解模型-检索式对话系统] | [Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering](https://arxiv.org/pdf/2007.01282.pdf) | Fusion-in-Decoder生成式阅读理解模型 | Izacard et al,2020

+ [数据集] | [MultiWOZ 2.2: A Dialogue Dataset with Additional Annotation Corrections and State Tracking Baselines](https://arxiv.org/pdf/2007.12720.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/260097352) | MultiWOZ是一个著名的面向任务的对话数据集，被广泛用作对话状态跟踪的基准，MultiWOZ 2.2是目前最新版本 | Zang et al,2020

+ [预训练-语言模型] | [Glancing Transformer for Non-Autoregressive Neural Machine Translation](https://arxiv.org/pdf/2008.07905.pdf) | [阅读笔记](https://www.linkresearcher.com/theses/5970ead3-229c-4193-9f67-f39dc16219f5) | 本文提出的GLAT是一种Non-Autoregressive翻译模型（摆脱BeamSearch），主要着重于并行化Decoder以及提高翻译质量，实际的效果速度快且在一些翻译方向上（英德）达到了SOTA。模型的核心结构沿用Transformer结构，参考预训练语言模型的MLM的做法，提出一种叫作GLM（Glancing LM）的方案，即使用两遍Decoder（同一个Decoder），第一遍的Decoder中，不加任何干预的获得模型的自然输出，这个时候将输出与Gold output进行对比，然后随机采样（也可以尝试其他的）目标词的词嵌入替换模型输出对应的hidden，然后再次喂入Decoder得到最终输出（注意，这里采样的词数量是根据训练情况好坏反比的，模型输出效果越好，采样的目标词越少，最终模型收敛到一次并行推理）。原理就是在第一次并行推理比较难学习到词与词之间的依赖关系，因此在第二次并行推理时，适当的引入目标词进行修正，进行增强训练 | Lihua Qian et al,2020

+ [异常检测-模型-机器学习-无监督] | [COPOD: Copula-Based Outlier Detection](https://arxiv.org/pdf/2009.09463.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/338189299) | 本文主要是基于copula统计概率函数，提出了COPOD的异常检测方法，COPOD使用了非参数（non-parametric）的方法，通过经验累积分布（Empirical CDF）来得到empirical copula，在这之后我们就可以简单的通过empirical copula来估算所有维度上的联合分布的尾端概率。因此COPOD不需要进行样本间的距离计算，从而节省运行开销且速度快，同时，该方法不需要调参，直接使用即可 | Zheng Li et al, 2020

+ [预训练-语言模型-关系抽取] | [A Frustratingly Easy Approach for Entity and Relation Extraction](https://arxiv.org/pdf/2010.12812.pdf) | [阅读笔记](http://www.sohu.com/a/430031845_129720) | 提出了一种非常简单的方法，该方法可以学习基于深度预训练语言模型构建的两个编码器，这两个模型分别被称为实体模型和关系模型（实体模型和关系模型的语境表示本质上捕获了不同的信息，因此共享其表示会损害性能）。同时，为了加快模型推断速度，该研究提出了一种新颖而有效的近似方法，该方法可实现 8-16 倍的推断加速，而准确率只有很小的降低 | Zexuan Zhong et al,2020

+ [预训练-语言模型-Prompt] | [AutoPrompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts](https://arxiv.org/pdf/2010.15980.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/366771566) | 本篇论文提出一种基于梯度的prompt搜索方案，方法比较直观，将通过梯度找出trigger word和mask拼接在文本中，形成一个语义上不通顺，但是对模型而言却具有合理prompt的样本，并且将label预测转换为masked token的预测。方法的核心在于选取trigger word，说白了就是选定一个已确定token数量的template，比如论文中{sentence}[T][T][T][T][T][P]，其中T就代表trigger word，P代表label，在这个例子中，准备使用五个token作为prompt，做法就是将这五个token标识为mask token，然后通过MLM的方式预测出token，然后选前k个最大化输入与梯度乘积的token，选出的token候选一次加入到prompt并评估预测的概率。在预测prompt token之外，还拿了mask token的hidden states过一个线性层预测label，并加上原本label位置mask token的loss进行训练。AutoPrompt的方法总的来说简单粗暴，不过带来的也是可解释性差，具体效果一般 | Taylor Shin et al,2020

+ [预训练-语言模型] | [Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://arxiv.org/pdf/2012.07436.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/351321328) | 一种效果远超Transformer的长序列预测模型，针对LSTF问题上的研究改进 | Haoyi Zhou et al,2020
  
+ [综述-可解释性] | [A Survey on Neural Network Interpretability](https://arxiv.org/pdf/2012.14261.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/341153242) | 关于神经网络可解释性的一篇综述，整理的挺不错的，不过就是相关领域前沿探索不足 | Yu Zhang et al,2020

+ [深度学习] | [A Theoretical Analysis of the Repetition Problem in Text Generation](https://arxiv.org/pdf/2012.14660.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/349675973) | 讨论Seq2Seq模型解码重复生成的原因 | Zihao Fu et al,2020

+ [预训练-语言模型-Prompt] | [Making Pre-trained Language Models Better Few-shot Learners](https://arxiv.org/pdf/2012.15723.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/386470305) | 文中提出的LM-BFF是一套简单的技术组合，用于仅在少量训练样本上对预训练的LM进行微调，其中包括：（1）基于Prompt的微调以及自动生成prompt的新方法；（2）一种动态的、有选择的在上下文中引入demonstration的方法。这里稍微介绍一下背景概念，Prompt方法主要分两种不同的研究方向：（1）基于prompt的fine-tuning（被认为是更适合小LM的few-shot learner）；（2）而对于大模型，fine-tuning比较困难，因此是希望固定他们的参数，通过不同的prompt将它们应用在不同的任务上。对于第二个研究方向，prompt分为Discrete Prompts和Soft Prompts，可以简单的认为discrete是选随机token，而soft则是直接用随机向量替换Embedding。然后还有除了Prompt之外，还有demonstration（in-context learning， 一种新的meta-learning方式），prompt和demonstration都是GPT-3很成功的设计，demonstration是多sample+input text作为模型输入，其中也有很多优化的方法 | Tianyu Gao et al,2020

+ [对话系统-预训练-检索式对话系统] | [Distilling Knowledge From Reader To Retriever For Question Answering](https://openreview.net/pdf?id=NTEz-6wysdb) | [阅读笔记](https://zhuanlan.zhihu.com/p/372694270) | 一种模型训练模型的开放域问答方法 | Izacard et al,2021

+ [预训练-语言模型-Prompt] | [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/pdf/2101.00190.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/386073664) | 本篇论文核心是针对离散的Prompt难以优化的问题，提出了参数化的prompt，仅微调prompt，freeze住LM。在小样本任务重，这种方法极大的减小的模型的参数，减少了过拟合的风险，这种参数化的prompt在小样本场景中，能够优于fine-tune的方法。这篇文章的做法和P-tuning差不多，都是设计了非自然语言的模板，只不过Prefix-tuning主要关心的是NLG的应用，而P-tuning更加关心NLU的应用 | Xiang Lisa Li et al,2021

+ [综述-向量召回] | [A Comprehensive Survey and Experimental Comparison of Graph-Based Approximate Nearest Neighbor Search](https://arxiv.org/pdf/2101.12631.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/415320221) | 论文是一篇关于graph-base的向量召回综述，聚焦实现了效率和精度最优权衡的近邻图索引，综述了 13 种具有代表性相关算法，包括NSW、HNSW等在内的优秀算法，并提出一个统一评估的pipeline | Mengzhao Wang et al,2021
  
+ [预训练-评估] | [LogME: Practical Assessment of Pre-trained Models for Transfer Learning](https://arxiv.org/pdf/2102.11005.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/358844524) | 一种通用且快速的评估选择适合下游任务的预训练模型的打分方法，logME | Kaichao You et al,2021

+ [Prompt-预训练-语言模型] | [GPT Understands, Too](https://arxiv.org/pdf/2103.10385.pdf) | [阅读笔记](https://kexue.fm/archives/8295) | 在本文之前的Prompt思路呢，要么是通过人工设计Prompt（如PET），要么是探索通过自动化搜索Prompt进行（如AutoPrompt、LM-BFF等），思路都限于搜索token来组成Prompt template（Discrete Prompt Search），而本文提出的P-tuning思路是不用关心template由哪些token word组成，对于模型而言，只需要token embedding，直观点说就是在template中，除了目标词正常以Mask token出现，prompt token则是[unused*]（也就是从未见过的token来构成模板，这里的token会过一层LSTM进行编码），其中token数目是一个超参数可以调整，这种方式极大的提升了template的搜索空间（连续）。小样本的时候固定模型权重，只优化[unused*]的Embedding，这样即使样本少也能学到prompt template，不容易过拟合。标注数据足够的话就直接放开所有权重一同训练微调就行 | Xiao Liu et al,2021

+ [Prompt-预训练-语言模型] | [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/pdf/2104.08691.pdf) | [阅读笔记](https://blog.csdn.net/c9Yv2cf9I06K2A9E/article/details/121112298) | 本文的方法和p-tuning相似，是固定LM，只训练prefix，这篇文章主要是验证了全量数据情况下，仅微调prompt相关的参数，能够达到fine-tune的效果（p-tuning的prompt token人为的选用[unused*]，而本文对prompt token的初始化分两种：置零和采用词表的一些预训练token embedding）。论文的最终结论有：1）：在一般模型大小的情况下，prompt token越多，效果越好（超过20增益减小），但是在超大模型的情况下，单个prompt token也能达到前面20个token以上的效果；2）：随机初始化、词表采样、用label标签初始化，其中label的方式效果最好；3）：LM Adaptation steps 越多，效果越好；4）：同时训练多个prompt进行ensemble，效果优于单一prompt | Brian Lester et al,2021

+ [预训练-语言模型-文本相似度/匹配/分类] | [SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/pdf/2104.08821.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/368353121) | 基于Sentence-Bert，引入对比学习的思想，在无监督与有监督语义相似度计算任务达到SOTA。主要围绕对比学习质量指标Alignment和Uniformity来进行优化，对于Unsupervised，核心是使用dropout mask生成正样本，负样本是in-batch negatives。而Supervised则是NLI中entailment关系样例对。负例：a) in-batch negatives b)NLI中关系为contradiction的样例对 | Tianyu Gao et al,2021

+ [预训练-语言模型] | [Are Pre-trained Convolutions Better than Pre-trained Transformers?](https://arxiv.org/pdf/2105.03322.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/380195756) | 将Transformer的Attention换成了卷积，尝试预训练模型新方式 | Yi Tay et al,2021

+ [综述-对话系统] | [Recent Advances in Deep Learning Based Dialogue Systems: A Systematic Survey](https://arxiv.org/pdf/2105.04387.pdf) | 对话系统综述：新进展新前沿 | JinJie Ni et al,2021

+ [对话系统-评估] | [Towards Quantifiable Dialogue Coherence Evaluation](https://arxiv.org/pdf/2106.00507.pdf) | QuantiDCE，一种实现可量化的对话连贯性评估指标模型 | Zheng Ye et al,2021

+ [对话系统-对话策略学习] | [Retrieve & Memorize: Dialog Policy Learning with Multi-Action Memory](https://arxiv.org/pdf/2106.02317.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/415170940) | 联合检索和记忆块的多action的Dialog Policy Learning模型，在action生成和response生成上效果SOTA | Yunhao Li et al,2021
  
+ [对话系统] | [Increasing Faithfulness in Knowledge-Grounded Dialogue with Controllable Features](https://arxiv.org/pdf/2107.06963.pdf) | 通过可控特征来增加知识对话系统的学习 | Rashkin et al,2021
  
+ [综述-Prompt-预训练] | [Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing](https://arxiv.org/pdf/2107.13586.pdf) | [阅读笔记1](https://zhuanlan.zhihu.com/p/409541189) / [阅读笔记2](https://zhuanlan.zhihu.com/p/461825791) | 关于Prompt-based learning的一篇综述，Prompt（提示/题词）和之前的MLM有些相似，通过定义template的方式，基于语言模型的特性直接估计出文本的概率，从而生成答案。相较于传统的语言模型依赖于针对特定下游任务的fine-tune，Prompt更加关注模型的迁移能力（它的目标就是希望对不同下游任务建立一个统一的范例），除了便捷和泛化能力之外，这样做的一个明显优势就是不同任务之间的数据可以共享，减少标注数据，随着数据累积，新的任务可以达到zero-shot learning的目的 | Pengfei Liu et al,2021

+ [文本相似度/匹配/分类-Prompt-预训练-语言模型] | [Noisy Channel Language Model Prompting for Few-Shot Text Classification](https://arxiv.org/pdf/2108.04106.pdf) | 本篇论文以实验探索为主，含有大量的实验对比，主要出发点就是在few-shot问题中，探讨控制训练参数对于direct model和channel model效果的影响，最终的论文的结论是Noisy Channel model明显优于direct model。论文中的direct model主要是指一般的P(c|x)，其中x是输入，c是label，而direct++ model则是基于direct，强化文本间的差异，引入空文本，即P(c|x)/P(c|null)，而channel model则是指使用贝叶斯公式重新参数化direct，P(c|x)=P(x|c)P(c)/P(x)，其中P(c)就是label数分之一，即P(1/C)，而P(x)独立于c，所以最终只需要计算P(x|c)。那么最后用形象一点的例子来解释direct和channel的差异就是，direct=x->c，channel=c->x。论文中对参数的控制采用了all finetuning、head tuning、transformation tuning和Prompt tuning（这里可以认为是soft prompt，即只需在输入序列中放入一些随机向量，与词汇表中的特定word embedding无关，并进行调整，同时固定预训练模型的其他部分）。在direct和channel的方法间，channel明显优于direct。在direct model的参数控制实验中，head tuning是最优的，但是当channel model配合soft prompt时，效果是最好的 | Sewon Min et al,2021

+ [对话系统-预训练] | [General-Purpose Question-Answering with MACAW](https://arxiv.org/pdf/2109.02593.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/425427299) | 生成式多角度问答模型，参数量只有GPT-3的十六分之一，作者主打的亮点是通过整合7种不同的任务范式（问题生成、答案生成...），使得模型能够通过不同角度学习到QA相关的内容，目的就是得到一个强大的QA版本预训练模型。输入的话就是通过"$s1$;$s2$.."的slot方式进行文本拼接，得到的解码输出也是如此输出的，模型内核还是基于transformer的改造 | Oyvind Tafjord et al,2021

+ [预训练-语言模型-Prompt] | [PPT: Pre-trained Prompt Tuning for Few-shot Learning](https://arxiv.org/pdf/2109.04332.pdf) | [阅读笔记](https://blog.csdn.net/c9Yv2cf9I06K2A9E/article/details/121112298) | 之前的工作都是在finetune阶段去使用prompt，这篇文章第一次提出了prompt pretraining的过程。一开始是因为观察了prompt tuning中的大模型尽管在全量数据下能够媲美finetune，但是在少样本情况下并不好，作者认为是因为在大模型上soft prompt对初始化很敏感，所以设计了一系列预训练的prompt task来给soft prompt提供一个很好的初始化。论文的结论是，通过 prompt 的预训练，在少样本情况下，大模型的 prompt tuning 同样能媲美 fine-tuning 的效果 | Yuxian Gu et al,2021

+ [对话系统-任务型对话-预训练] | [Constraint based Knowledge Base Distillation in End-to-End Task Oriented Dialogs](https://arxiv.org/pdf/2109.07396.pdf) | 基于KB的End2End的Task-Oriented的对话系统，使用pairwise相似度过滤相关信息来获得KB中的n元结构，就这一点上倒没有什么新奇，只不过相对于之前的方式修改的entity格式。不过在避免检索到部分entity相似但并不是目标的record的情况，作者加入了辅助的损失函数用于embedding constraint，这种做法确实减少了相同entity之间的相似性，从而提高record的可靠性，值得借鉴。基于现有的F1指标的缺点，提出multiset entity F1 | Dinesh Raghu et al,2021
  
+ [综述] | [Paradigm Shift in Natural Language Processing](https://arxiv.org/pdf/2109.12575.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/415867930) | 总结归纳NLP中任务范式并分析的综述，论文给出了七种范式的定义，针对此分析一些具体任务（范式迁移）的例子，并指出四种可能大一统的NLP任务范式：LM，matching，MRC，Seq2Seq（LM减少工程量，MRC具有更高的可解释性，seq2seq在处理复杂任务时具有更高的灵活性），但是没有把Prompt纳进去（狗头） | Tianxiang Sun et al,2021

+ [综述-数据增强] | [Data Augmentation Approaches in Natural Language Processing: A Survey](https://arxiv.org/pdf/2110.01852.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/420295576) | 哈工大的工作，对15中NLP数据增强方法进行了总结和对比，有详细的优缺点说明，还有一些使用技巧，实用性非常强，需要的时候可以详细的参考原文以及相关的文献的应用细节。几个开源工具：Easy DA、UNsupervised DA、nlpaug、eda_nlp_for_Chinese | Bohan Li et al,2021

+ [预训练-语言模型-Prompt] | [SPoT: Better Frozen Model Adaptation through Soft Prompt Transfer](https://arxiv.org/pdf/2110.07904.pdf) | [阅读笔记](https://blog.csdn.net/c9Yv2cf9I06K2A9E/article/details/121112298) | 之前的工作证明了prompt 的初始化很重要，而在PPT（Pre-trained Prompt）那篇论文中提出的预训练方法能够给 prompt 提供一个很好的初始化，但是有没有其他预训练的方式，比如不用设计预训练任务的，因此本文提出了一种prompt transfer（SPoT）的方法，即学习一个或者多个源任务的  prompt 来初始化目标任务的 prompt，这种方式能够使得 prompt tuning 在不同模型尺寸（包括小模型）上都能媲美甚至优于 fine-tuning（注意，无法超过 multi-task fine-tuning 的效果）。论文的结论是在全量数据 + 仅微调 prompt 的情况下，SPoT 能够在多个模型尺寸（包括小模型）下媲美和优于 model tuning 的效果，并能在使用超大模型情况下媲美强基线 Multi-task Tuning | Tu Vu et al,2021

+ [异常检测-模型-机器学习-无监督] | [ECOD: Unsupervised Outlier Detection Using Empirical Cumulative Distribution Functions](https://arxiv.org/pdf/2201.00382.pdf) | 本文作者和2020年提出的COPOD算法同作者，本文提出的ECOD算法是COPOD算法的扩展版本。ECOD算法使用经验累计分布函数的无监督离群值检测，通过类似集成的方法（结合同一样本不同维度的离群分，假设各维度特征相互独立），计算每个样本的离群分，分值越高是异常值的可能性越大。另外，在计算各特征维度的左尾和右尾ECDF，并得到对应离群分后，通过skewness（偏度）来矫正集群分得到最终结果 | Zheng Li et al, 2022

+ [预训练-语言模型-Prompt] | [Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?](https://arxiv.org/pdf/2202.12837.pdf) | [阅读笔记](https://mp.weixin.qq.com/s/qdCuPWsNg_lOxUkap1dQ9Q) | 本文主要探讨分析Prompt范式下，预训练语言模型是如何学习并work的。主要的结论是在in-context learning 学习中，学习并不是输入与标注之间的关联，而是通过展示数据形式，来激活与训练模型的能力。此外附带两个结论：在meta learning环境下，in-context leanring的这个特点更为明显；因为标签不重要，所以可以用无标注领域内数据做in-context zero shot learning | Sewon Min et al,2022

+ [预训练-语言模型-文本相似度/匹配/分类] | [How Different are Pre-trained Transformers for Text Ranking?](https://arxiv.org/pdf/2204.07233.pdf) | [阅读笔记](https://mp.weixin.qq.com/s/JaP7FjQeHyHURj1qh3rKQg) | 本文主要对BERT（Cross-Encoder，即软匹配）和传统稀疏排序方法的BM25（精确匹配）进行query-doc排序结果的比较分析，尝试搞清楚Cross-Encoder和BM25的区别，弄清CE的运作原理。论文得到的结论就是：（1）精确匹配是一个重要的基础排序策略，而CE的软匹配能力是BM25不具备的；（2）对于高度相关的doc排序，CE和BM25各自的相关性定义有着很大的不同，且BM25明显低估了许多文档的相关性；（3）CE的潜力在于，它可以召回BM25非常不看好而实际却是高度相关的doc；（4）CE通过考虑上下文信息客服了对term匹配的依赖，能够找到“不可能相关”的结果，即语义泛化能力。从整个实验中也可以明显的看出CE和BM25各自都有着自己的优势，CE并不能完全替代BM25，不管是在召回还是在排序阶段，这两者依旧是相辅相成的关系（别忘了个事实，CE方法上百亿的参数，BM25相比之下“弱小”很多） | David Rau et al,2022

+ [异常检测-综述-自监督] | [Self-Supervised Anomaly Detection: A Survey and Outlook](https://arxiv.org/pdf/2205.05173.pdf) | 一篇关于异常检测的综述论文，主要是围绕自监督形式的异常检测方法进行介绍，论文中大部分方法的切入视角是图像，搞文本或者多模态的话也可以看看，说不定有所启发。全文主要的内容大体可以分为：（1）对目前自监督形式的异常检测方法和其应用的场景进行介绍和讨论；（2）根据异常检测算法所针对的数据样本的不同，提出对这些算法进行划分，有利于根据不同场景进行算法的选择；（3）最后对未来的发展进行了讨论 | Hadi Hojjati et al, 2022

+ [异常检测-评估] | [ADBench: Anomaly Detection Benchmark](https://arxiv.org/pdf/2206.09426.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/565458918) | 一篇很全面的针对Tabular Data的异常检测方法的实验论文，通过设计Benchmark对30种算法（包括许多传统机器学习算法也加入了实验）进行实验分析。其主要贡献就是设计了Benchmark，设计切入的角度分为三个：（1）从完全无监督异常检测到完全监督的异常检测，标签的数量有多重要？（2）对于不同种类的异常，如何分析算法的优劣？（3）对于数据质量中面临的问题，比如噪音、重复、错误等，哪些算法更加鲁棒？ | Songqiao Han et al, 2022


# Blog Article | 文章

+ [用ALBERT和ELECTRA之前，请确认你真的了解它们](https://kexue.fm/archives/7846/comment-page-1) | 文章对ALBERT和ELECTRA的优缺点进行了思考，并于BERT进行的比较分析，得到两点结论：（1）如果不到xlarge版，那么没必要用ALBERT，同一速度的ALBERT效果比BERT差，同一效果的ALBERT速度比BERT慢；（2）ELECTRA的预训练速度是加快了，但从目前的实验来看，它相比同级别的BERT在下游任务上的效果并没有突出优势，可以试用，但是效果变差了也不用太失望 | 苏剑林, 2020

