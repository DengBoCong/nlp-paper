# 目录
+ [运行说明](#运行说明)
+ [模型效果](#模型效果)


# 运行说明
+ 运行入口：
   + task_chatter.py为seq2seq的执行入口文件：指令需要附带运行参数
+ 执行的指令格式：
   + task：python task_chatter.py -t/--type [执行模式]
+ 执行类别：pre_treat(默认)/train/chat
+ 执行指令示例：
   + python task_chatter.py
   + python task_chatter.py -t pre_treat
+ pre_treat模式为文本预处理模式，如果在没有分词结果集的情况下，需要先运行pre_treat模式
+ train模式为训练模式
+ chat模式为对话模式。chat模式下运行时，输入exit即退出对话。

+ 正常执行顺序为pre_treat->train->chat

# 模型效果
待完善