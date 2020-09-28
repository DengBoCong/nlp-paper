标题：MultiWOZ 2.2 : A Dialogue Dataset with Additional Annotation Corrections and State Tracking Baselines\
原文链接：[Link](https://arxiv.org/pdf/2007.12720.pdf)\
Github：[NLP相关Paper笔记和实现](DengBoCong/nlp-paper)\
说明：阅读论文时进行相关思想、结构、优缺点，内容进行提炼和记录，论文和相关引用会标明出处，引用之处如有侵权，烦请告知删除。\
转载请注明：DengBoCong

最近在搜集一些对话数据集，众所周知，生成对话数据集是一件费钱又费时的工作，所以一般只有大机构才能做出高质量且庞大的数据集，所以看到好的数据集，那还不赶紧收藏一波。

+ [代码链接](budzianowski/multiwoz)
+ [数据集链接](http://dialogue.mi.eng.cam.ac.uk/index.php/corpus/)
+ [MultiWOZ论文地址](https://arxiv.org/pdf/1810.00278.pdf)
+ [MultiWOZ 2.1论文地址](https://arxiv.org/pdf/1907.01669.pdf)

# Abstract
MultiWOZ是一个著名的面向任务的对话数据集，其中包含10,000多个跨越8个域的带注释对话，而被广泛用作对话状态跟踪的基准。但是，最近的工作报告说，对话状态注释中存在大量噪音。MultiWOZ 2.1中识别并修复了许多错误的注释和用户话语，从而改进了该数据集的版本。本篇论文工作介绍了MultiWOZ 2.2，它是该数据集的又一个改进版本。首先，我们在MultiWOZ 2.1之上的17.3％话语中识别并修复对话状态注释错误。其次，我们通过不允许带有大量可能值的槽（例如，餐厅名称，预订时间）来重新定义数据集。此外，我们为这些插槽引入了插槽跨域注释，以在最近的模型中将它们标准化，该模型以前使用自定义字符串匹配启发法来生成它们。我们还会在更正后的数据集上对一些最新的对话状态跟踪模型进行基准测试，以方便将来进行比较。最后，我们讨论了对话数据收集的最佳做法，可以帮助避免注释错误。
# Introduction
最近，数据驱动技术已针对不同的对话系统模块实现了最先进的性能，但是，由于训练上述模块需要广泛的注释，因此收集高质量的注释对话数据集仍然是研究人员的一个挑战。很多公共数据集，如DSTC2、WOZ、SimulatedDialogue、MultiWOZ、TaskMaster、SGD等等，对促进这一领域的研究非常有用。在这些数据集中，MultiWOZ是用于对话状态跟踪的最广泛使用的基准。它包含超过10,000个对话，涉及8个领域，分别是：餐厅，酒店，景点，出租车，火车，医院，公共汽车和警察。MultiWOZ 2.2做出了如下三点贡献：
+ 我们修复MultiWOZ 2.1中的注释错误，不一致和本体问题，并发布其改进版本。
+ 我们为用户和系统话语添加了插槽跨度注释，以在未来的模型中对其进行标准化。我们还为每个用户的语句注解活跃用户的意图和请求槽。
+ 我们在更正后的数据集中对一些最新的对话状态跟踪模型进行基准测试，以便于将来的工作进行比较。

# Annotation Errors
Wizard-of-Oz（人机交互）：在此设置中，两个人群工作人员配对在一起，一个充当用户，另一个充当对话代理。每个对话由一组指定用户目标的唯一指令驱动，这些指令与扮演用户角色的人群共享。在每个用户转过身后，扮演对话代理（向导）角色的人群工人会注释更新后的对话状态。更新状态后，该工具会显示将对话状态与向导匹配的一组实体，然后向导使用该实体来生成响应并将其发送给用户。

以下两类主要错误，但在MultiWOZ 2.1中未进行纠正：
+ Hallucinated Values：幻觉值以对话状态存在，而未在对话历史记录中指定，我们观察到了四种不同类型的此类错误，它们在下图中显示并在下面进行了描述。
   + 过早的加入，代理在以后的讲话中已经提到了这些值。
   + 对话中甚至在未来的讲话中都根本没有提到这些值。
   + 由于印刷错误，在对话历史记录中找不到这些值。
   + 隐式时间处理，这特别涉及以时间为值的插槽

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200927233324153.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)

+ Inconsistent State Updates：我们还遇到了MultiWOZ 2.1中的注释，这些注释在语义上是正确的，但是没有遵循一致的注释准则。对话状态中出现不一致的原因有三个：
   + 多种来源，可以通过各种来源在对话状态中引入插槽值。
   + 值的歧义，比如“18:00”和“6 pm”
   + 跟踪策略不一致

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200927234238301.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)

# Ontology Issues
为了解决不完整的问题，MultiWOZ 2.1通过列出整个数据集中对话状态中存在的所有值来重建了本体，但是仍然存在一些未解决的问题。首先，对于某些插槽，列出了共享相同语义的多个值。其次，我们观察到本体中有多个插槽值，这些值无法与数据库中的任何实体相关联。

# Correction Procedure
为了避免上述问题，我们主张在数据收集之前定义本体，这不仅可以作为注释者的指南，而且还可以防止数据集中的注释不一致以及印刷和注释错误导致的本体破坏。本节描述了我们对新本体的定义，我们将其称为策略，然后是对状态和动作注释的更正。最后，我们还显示了修改的统计信息。
+ 策略定义：策略将不同的插槽分为两类-非分类和分类。具有大量可能值的动态插槽被归为“非分类”，与本体不同，架构不提供此类插槽的预定义值列表，他们的值是从对话历史中提取的。相反有限的插槽被归为“分类”，与本体类似，策略列出了此类插槽的所有可能值。此外，在注释期间，必须从模式中定义的预定义候选列表中选择对话状态和用户或系统操作中这些插槽的值，这有助于实现注释的完整性和一致性。如下图：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200927235925309.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
+ 分类插槽：分类插槽所有可能值的列表是从MultiWOZ 2.1随附的相应数据库中构建的。
+ 非分类插槽：从对话历史记录中提取非分类插槽的值，我们使用一种自定义的字符串匹配方法，该方法考虑了可能的拼写错误和替代表达式来定位语义上与注释相似的所有值。如果有多个匹配项，我们选择最近提及的值并注释其跨度。下图是2.1和2.2的区别

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200928000339675.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
+ 用户和系统动作：用户和系统动作注释提供了相应话语的语义表示。为了保持对话行为和domain之间的联系，这里用到了本文二作之前的一个做法，将同一domain下的对话对话行为组合起来放到了frames里。
+ 数据：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200928145728211.png#pic_center)
# Additional annotations
除了跨域注释外，我们还为每个用户回合添加了活动的用户意图和请求槽。预测活动用户的意图和请求槽是两个新的子任务，可用于评估模型性能并促进对话状态跟踪。主动意图或API的预测对于支持数百个API的大规模对话系统的效率也至关重要。
+ Active：active intent指定了用户话语中表达的所有意图。
+ Requested Slots：Requested slots指定了用户想系统请求的槽位，也就是所谓的提问槽。

# Dialogue State Tracking Benchmarks
最新数据驱动对话系统状态跟踪模型主要采用两种方法：span-based和candidate-based，本文baseline包括：SGDbaseline、TRADE、DS-DST。
+ 各个模型在这三个数据及上的joint acc
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200928150723774.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
+ 在categorical和non-categorical两类slot上单独计算joint acc
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020092815080387.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
# Discussion
Wizard-of-Oz范式虽然是个强有力的收集自然语言对话的技术，但是他也存在很大的噪音。作者在这一节就提出了几点，来最小化标注错误。

+ 在标注数据前，应该先定义一个本体或者schema。对于categorical的槽位，这个schema需要定义好明确的slot，每个slot都有一个固定的可能的value的集合。然后标注接口，需要强制性的保证这些槽值的正确性。对于non-categorical的槽位，标注接口要限制标注的value必须是出现在对话历史中的值。
+ 在标注任务之后，可以进行简单的检查，以识别错误的标注。

#  Conclusion
MultiWOZ 2.1是MultiWOZ 2.0数据集的改进版本，被广泛用作对话状态跟踪的基准。我们找出在MultiWOZ 2.1中未解决的注释错误，不一致和与本体相关的问题，并发布更正的版本– MultiWOZ 2.2。我们添加了新的策略，标准化的插槽值，更正的注释错误和标准化的跨域注释。此外，我们为每个用户回合注释了活动意图和请求槽位，除了修复现有操作外，并添加了丢失的用户和系统操作。我们在新数据集上对一些最新模型进行了基准测试：实验结果表明，MultiWOZ 2.1和MultiWOZ 2.2的模型性能相似。我们希望清理后的数据集有助于在模型之间进行更公平的比较，并促进该领域的研究