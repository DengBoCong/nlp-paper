import os
import json
import jieba
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchnlp.encoders.text import pad_tensor, stack_and_pad_tensors, StaticTokenizerEncoder


def preprocess_sentence(start_sign, end_sign, sentence):
    """
    用于给句子首尾添加start和end
    Args:
        start_sign: 开始标记
        end_sign: 结束标记
        sentence: 处理的语句
    Returns:
        合成之后的句子
    """
    sentence = start_sign + ' ' + sentence + ' ' + end_sign
    return sentence


def preprocess_request(sentence, start_sign, end_sign, token, max_length):
    sentence = " ".join(jieba.cut(sentence))
    sentence = preprocess_sentence(start_sign, end_sign, sentence)
    inputs = [token.get(i, 3) for i in sentence.split(' ')]
    inputs = torch.tensor(inputs)
    inputs = [pad_tensor(tensor=inputs[:max_length], length=max_length, padding_index=0)]
    inputs = stack_and_pad_tensors(inputs)[0]
    dec_input = torch.unsqueeze(torch.tensor([token[start_sign]]), 0)

    return inputs, dec_input


def read_tokenized_data(path, start_sign, end_sign, num_examples):
    """
    用于将分词文本读入内存，并整理成问答对，返回的是整理好的文本问答对以及权重
    Args:
        path: 分词文本路径
        start_sign: 开始标记
        end_sign: 结束标记
        num_examples: 读取的数据量大小
    Returns:
        zip(*word_pairs): 整理好的问答对
        diag_weight: 样本权重
    """
    if not os.path.exists(path):
        print('不存在已经分词好的文件，请先执行pre_treat模式')
        exit(0)

    with open(path, 'r', encoding="utf-8") as file:
        lines = file.read().strip().split('\n')
        diag_weight = []
        word_pairs = []

        # 这里如果num_examples为0的话，则读取全部文本数据，不为0则读取指定数量数据
        if num_examples != 0:
            lines = lines[:num_examples]

        for line in lines:
            # 文本数据中的问答对权重通过在问答对尾部添加“<|>”配置
            temp = line.split("<|>")
            word_pairs.append([preprocess_sentence(start_sign, end_sign, sentence) for sentence in temp[0].split('\t')])
            # 如果没有配置对应问答对权重，则默认为1.
            if len(temp) == 1:
                diag_weight.append(1.)
            else:
                diag_weight.append(float(temp[1]))

    return zip(*word_pairs), diag_weight


def load_data(dict_fn, data_fn, batch_size, start_sign, end_sign, checkpoint_dir, max_length, max_train_data_size=0):
    """
    数据加载方法，主要将分词好的数据进行整理，过程中保存字典文件，方便后续其他功能
    使用，方法返回处理好的dataset，steps_per_epoch，checkpoint_prefix
    Args:
        dict_fn: 将训练数据的字典保存，用于以后使用，路径
        data_fn: 分词好的训练数据路径
        batch_size: batch大小
        start_sign: 开始标记
        end_sign: 结束标记
        checkpoint_dir: 检查点保存路径
        max_length: 最大句子长度
        max_train_data_size: 最大训练数据大小
    Returns:
        dataset: PyTorch的DataLoader
        steps_per_epoch: 每轮的步数
        checkpoint_prefix: 保存检查点的前缀
    """
    print("训练数据读取中...")
    (input_lang, target_lang), diag_weight = read_tokenized_data(data_fn, start_sign, end_sign, max_train_data_size)
    diag_weight = torch.tensor(diag_weight, dtype=torch.float32)
    # 合并input，target用于生成统一的字典
    lang = np.hstack((input_lang, target_lang))
    print("读取完成，正在格式化训练数据...")
    tokenizer = StaticTokenizerEncoder(sample=lang, tokenize=lambda x: x.split())
    # 将文本序列转换文token id之后，并进行填充
    input_data = [pad_tensor(tensor=tokenizer.encode(example)[:max_length], length=max_length, padding_index=0) for
                  example in input_lang]
    target_data = [pad_tensor(tensor=tokenizer.encode(example)[:max_length], length=max_length, padding_index=0) for
                   example in target_lang]
    input_tensor = stack_and_pad_tensors(input_data)[0]
    target_tensor = stack_and_pad_tensors(target_data)[0]

    print("格式化完成，正在整理训练数据并保存字典")
    word_index = {}
    vocab_list = tokenizer.vocab
    for i in range(tokenizer.vocab_size):
        word_index[vocab_list[i]] = i
        word_index[i] = vocab_list[i]

    with open(dict_fn, 'w', encoding='utf-8') as file:
        file.write(json.dumps(word_index, indent=4, ensure_ascii=False))
    print("数据字典保存完成！")

    dataset = PairDataset(input_tensor, target_tensor, diag_weight)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    steps_per_epoch = len(input_tensor) // batch_size

    return loader, steps_per_epoch


def load_token_dict(dict_fn):
    """
    加载字典方法
    :return:input_token, target_token
    """
    if not os.path.exists(dict_fn):
        print("不存在字典文件，请先执行train模式并生成字典文件")
        exit(0)

    with open(dict_fn, 'r', encoding='utf-8') as file:
        token = json.load(file)

    return token


class PairDataset(Dataset):
    """
    专门用于问答对形式的数据集构建的dataset，用于配合DataLoader使用
    """

    def __init__(self, input, target, diag_weight):
        self.input_tensor = input
        self.target_tensor = target
        self.diag_weight = diag_weight

    def __getitem__(self, item):
        return self.input_tensor[item], self.target_tensor[item], self.diag_weight[item]

    def __len__(self):
        return len(self.input_tensor)


def sequences_to_texts(sequences, token_dict):
    """
    将序列转换成text
    """
    result = []
    for text in sequences:
        temp = ''
        for token in text:
            temp = temp + ' ' + token_dict[str(token)]
        result.append(temp)
    return result
