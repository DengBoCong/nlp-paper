Paper-Recurrence | Still work
========================

由于考虑到部分模型及思路想法的复现代码量比较大，集中放在一个仓库中不合适，所以这里将一些代码量较大的项目分离到单独的仓库中，并通过引用指向。

# Notice | 特别说明
**特别说明**：此处模型代码为纯模型结构的代码，且都经过本人训练、测试、评价使用无误的代码，在代码关键处我添加了必要的注释，可以放心使用。

目前我的想法是直接复现论文时，只上传模型结构的代码，这样有利于缩小项目体积（后续逐步构建完善）。当然，如果有对部分复现代码感兴趣的伙伴，想要提出交流或者想要我实现完整过程等，欢迎提Issue。

# Data | 数据
1. [smudict](https://github.com/DengBoCong/nlp-paper/blob/master/paper-code/data/cmudict-0.7b)
2. [thchs30](https://github.com/DengBoCong/nlp-paper/blob/master/paper-code/data/lexicon.txt)
3. [ubuntu corpus](http://dataset.cs.mcgill.ca/ubuntu-corpus-1.0/)
4. TrecQA：TrecQA是QA最广泛评估和长期作为标准的数据集之一，用于回答基于事实的问题的任务
5. QatarLiving：答案通常是主观的而不是事实的，在这个数据集中，每个问题有十个答案，标记为“正向”和“负向”
6. Tweet Reply Prediction：数据集中包含对知名品牌的Tweet-Response对

# Code | 模型代码

1. [Byte Pair Encoding（BPE）](https://github.com/DengBoCong/paper/blob/master/paper-code/bpe.py)：论文代码
2. [Batch Normalization](https://github.com/DengBoCong/paper/blob/master/paper-code/batch_normalization.py)
3. [Cosine Normalization](https://github.com/DengBoCong/paper/blob/master/paper-code/conv2d_cosnorm.py)
4. [Group Normalization](https://github.com/DengBoCong/paper/blob/master/paper-code/group_normalization.py)：论文代码

## Tensorflow2.3
### Tools
1. [en_text_to_phoneme](https://github.com/DengBoCong/nlp-paper/blob/master/paper-code/tensorflow_src/tools/en_text_to_phoneme.py)
### Models
1. [Transformer](https://github.com/DengBoCong/nlp-paper/blob/master/paper-code/tensorflow_src/models/transformer.py)
2. [Scheduled Sampling for Transformer](https://github.com/DengBoCong/nlp-paper/blob/master/paper-code/tensorflow_src/models/transformer.py)
3. [GPT2](https://github.com/DengBoCong/paper/blob/master/paper-code/tensorflow_src/models/gpt2.py)
4. [GPT2-TF2.3完整仓库](https://github.com/DengBoCong/GPT2-TF2.3)：使用GPT2以及TensorFlow2.3实现闲聊，后续更新PyTorch。
5. [Sequential Matching Network](https://github.com/DengBoCong/nlp-paper/blob/master/paper-code/tensorflow_src/models/smn.py)
6. [Seq-to-Seq base](https://github.com/DengBoCong/nlp-paper/blob/master/paper-code/tensorflow_src/models/seq2seq.py)
7. [Neural Belief Tracker](https://github.com/DengBoCong/nlp-paper/blob/master/paper-code/tensorflow_src/models/nbt.py)
8. [An End-to-End Trainable Neural Network Model with
Belief Tracking for Task-Oriented Dialog](https://github.com/DengBoCong/nlp-paper/blob/master/paper-code/tensorflow_src/models/task)

## Pytorch1.7.0
[Seq-to-Seq base](https://github.com/DengBoCong/nlp-paper/blob/master/paper-code/pytorch_src/seq2seq)：包含完整数据处理、训练、对话、模型保存恢复等代码