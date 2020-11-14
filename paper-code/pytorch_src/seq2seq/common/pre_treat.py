import os
import json
import jieba
import numpy as np


def preprocess_raw_data(raw_data, tokenized_data):
    """
    用来对原始文本进行预处理的方法，主要是将原
    始文本进行分词后，保存在一个新的文本中，供后继使用
    Args:
        raw_data: 原始数据路径
        tokenized_data: 生成token数据保存路径
    Returns:
    """

    # 首先判断原数据集是否存在，不存在则退出
    if not os.path.exists(raw_data):
        print("数据集不存在，请添加数据集")
        exit(0)

    pairs = []
    max_len = 0
    min_len = 10000
    sentence_len = []

    # 对每一轮对话上下文进行配对，形成一问一答两个部分，如果遇到
    # 下一轮对话，直接跳过
    with open(raw_data, encoding="utf-8") as file:
        one_pair = []
        count = 0
        for line in file:
            line = line.strip('\n').replace('/', '')
            # line = re.sub(r"[%s]+" % punctuation, "", line)
            # 因为原始数据集中，是一轮一轮的对话排列的，所以需要注意的是
            # 在一轮对话结束之后，最后一句不能作为问句，需要跳到下一轮进行处理
            if line == '':
                one_pair = []
                continue
            elif len(one_pair) == 1:
                one_pair.append(line)
                pairs.append(one_pair)
                one_pair = [line]
                count += 1
                if count % 10000 == 0:
                    print('已处理：', count, '个问答对')
            else:
                one_pair.append(line)

            length = len(line)
            max_len = max(max_len, length)
            min_len = min(min_len, length)
            sentence_len.append(length)

    print("数据读取完毕，正在处理中...")

    # 将处理之后存在内存中的数据写入到新文本中
    with open(tokenized_data, 'w', encoding="utf-8") as file:
        for i in range(len(pairs)):
            file.write(" ".join(jieba.cut(pairs[i][0])) + "\t" + " ".join(jieba.cut(pairs[i][1])) + "\n")
            if i % 10000 == 0:
                print(len(range(len(pairs))), '处理进度：', i)

    print("数据处理完毕，数据信息统计：语句最大长度：{}，语句最短长度{}，语句平均长度{:.3f}".format(max_len, min_len, np.mean(sentence_len)))


def preprocess_raw_lccc_data(raw_data, tokenized_data):
    """
    用于处理LCCC数据集的方法，将LCCC数据集处理成问答对的形式
    Args:
        raw_data: 原始数据路径
        tokenized_data: 生成token数据保存路径
    Returns:
    """
    if not os.path.exists(raw_data):
        print('数据集不存在，请添加数据集!')
        exit(0)

    pairs = []
    count = 0
    max_len = 0
    min_len = 10000
    sentence_len = []

    with open(raw_data, 'r', encoding="utf-8") as file:
        raw_data = json.load(file)
        for data in raw_data:
            max_len = max(max_len, len(data[0]))
            min_len = min(min_len, len(data[0]))
            sentence_len.append(len(data[0]))
            for i in range(len(data) - 1):
                max_len = max(max_len, len(data[i + 1]))
                min_len = min(min_len, len(data[i + 1]))
                sentence_len.append(len(data[i + 1]))
                pairs.append([data[i], data[i + 1]])
            count += 1
            if count % 10000 == 0:
                print("已读取：{}轮对话数据".format(count))

    print('读取完毕，处理中...')
    count = 0
    with open(tokenized_data, 'w', encoding="utf-8") as file:
        for pair in pairs:
            file.write(pair[0] + "\t" + pair[1] + "\n")
            count += 1
            if count % 10000 == 0:
                print("数据处理进度：{}".format(count))

    print("数据处理完毕，数据信息统计：语句最大长度：{}，语句最短长度{}，语句平均长度{:.3f}".format(max_len, min_len, np.mean(sentence_len)))
