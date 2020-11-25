import io
import os
import json
import random
import numpy as np
import tensorflow as tf
from pathlib import Path
from common.kb import load_kb
from collections import defaultdict
from nltk import wordpunct_tokenize
import config.get_config as _config
from nltk.tokenize import RegexpTokenizer


def preprocess_sentence(start_sign, end_sign, w):
    """
    用于给句子首尾添加start和end
    :param w:
    :return: 合成之后的句子
    """
    w = start_sign + ' ' + w + ' ' + end_sign
    return w


def create_dataset(path, num_examples, start_sign, end_sign):
    """
    用于将分词文本读入内存，并整理成问答对
    :param path:
    :param num_examples:
    :return: 整理好的问答对
    """
    is_exist = Path(path)
    if not is_exist.exists():
        file = open(path, 'w', encoding='utf-8')
        file.write('吃饭 了 吗' + '\t' + '吃 了')
        file.close()
    size = os.path.getsize(path)
    lines = io.open(path, encoding='utf-8').read().strip().split('\n')
    if num_examples == 0:
        word_pairs = [[preprocess_sentence(start_sign, end_sign, w) for w in l.split('\t')] for l in lines]
    else:
        word_pairs = [[preprocess_sentence(start_sign, end_sign, w) for w in l.split('\t')] for l in
                      lines[:num_examples]]

    return zip(*word_pairs)


def max_length(tensor):
    """
    :param tensor:
    :return: 列表中最大的长度
    """
    return max(len(t) for t in tensor)


def read_data(path, num_examples, start_sign, end_sign):
    """
    读取数据，将input和target进行分词后返回
    :param path: Tokenizer文本路径
    :param num_examples: 最大序列长度
    :return: input_tensor, target_tensor, lang_tokenizer
    """
    input_lang, target_lang = create_dataset(path, num_examples, start_sign, end_sign)
    input_tensor, target_tensor, lang_tokenizer = tokenize(input_lang, target_lang)
    return input_tensor, target_tensor, lang_tokenizer


def tokenize(input_lang, target_lang):
    """
    分词方法，使用Keras API中的Tokenizer进行分词操作
    :param input_lang: 输入
    :param target_lang: 目标
    :return: input_tensor, target_tensor, lang_tokenizer
    """
    lang = np.hstack((input_lang, target_lang))
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token=3)
    lang_tokenizer.fit_on_texts(lang)
    input_tensor = lang_tokenizer.texts_to_sequences(input_lang)
    target_tensor = lang_tokenizer.texts_to_sequences(target_lang)
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor, maxlen=_config.max_length,
                                                                 padding='post')
    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor, maxlen=_config.max_length,
                                                                  padding='post')

    return input_tensor, target_tensor, lang_tokenizer


def load_dataset(dict_fn, data_fn, start_sign, end_sign, max_train_data_size=0):
    """
    数据加载方法，含四个元素的元组，包括如下：
    :return:input_tensor, input_token, target_tensor, target_token
    """
    input_tensor, target_tensor, lang_tokenizer = read_data(data_fn, max_train_data_size, start_sign, end_sign)

    with open(dict_fn, 'w', encoding='utf-8') as file:
        file.write(json.dumps(lang_tokenizer.word_index, indent=4, ensure_ascii=False))

    return input_tensor, target_tensor, lang_tokenizer


def load_token_dict(dict_fn):
    """
    加载字典方法
    :return:input_token, target_token
    """
    with open(dict_fn, 'r', encoding='utf-8') as file:
        token = json.load(file)

    return token


def pad_sequence(seqs, max_len):
    """
    填充序列，0
    :param seqs: 序列
    :return: 返回填充好的序列
    """
    padded = [seq + [0] * (max_len - len(seq)) for seq in seqs]
    return padded


def sequences_to_texts(sequences, token_dict):
    """
    将序列转换成text
    """
    inv = {}
    for key, value in token_dict.items():
        inv[value] = key

    result = []
    for text in sequences:
        temp = ''
        for token in text:
            temp = temp + ' ' + inv[token]
        result.append(temp)
    return result


def tokenize_en(sent, tokenizer):
    """
    用来针对英文句子的分词
    :param sent: 句子
    :param tokenizer: 正则表达式分词器
    :return: 分好的句子
    """
    tokens = tokenizer.tokenize(sent)
    ret = []
    for t in tokens:
        # 这里要注意，如果是槽位，要直接作为一个token放进去，例如<v.pricerange>
        if '<' not in t:
            ret.extend(wordpunct_tokenize(t))
        else:
            ret.append(t)
    return ret


def load_dialogs(diag_fn, kb, groups_fn=None):
    """
    加载数据集中的对话，按照格式整理好并返回
    :param diag_fn: 数据集文件路径
    :param kb: knowledge base的词表
    :param groups_fn: 语句槽位集合文件路径
    :return: 整理好的数据
    """
    with open(diag_fn) as file:
        dialogues = json.load(file)

    data = []
    for dialogue in dialogues:
        usr_utterances = []
        sys_utterances = []
        states = []
        kb_found = []
        sys_utterance_groups = []

        for turn in dialogue['dialogue']:
            usr_utterances.append('<sos> ' + turn['transcript'] + '<eos>')
            sys_utterances.append('<sos> ' + turn['system_transcript'] + '<eos>')
            slots = []
            search_keys = []

            for state in turn['belief_state']:
                if state['act'] == 'inform':
                    slots.append(state['slots'][0])
                    state['slots'][0][0] = state['slots'][0][0].replace(' ', '').replace('center', 'centre')
                    search_keys.append(state['slots'][0])
                elif state['act'] == 'request':
                    slots.append((state['slots'][0][1].replace(' ', '') + '_req', 'care'))
                else:
                    raise RuntimeError('illegal state : %s' % (state,))

            states.append(slots)
            ret = kb.search_multi(search_keys)
            kb_found.append(len(ret))

        # 这里就跳过第一个，因为一般系统第一个是空
        sys_utterances = sys_utterances[1:]
        usr_utterances = usr_utterances[:-1]
        kb_found = kb_found[:-1]
        states = states[:-1]

        data.append({
            'usr_utterances': usr_utterances,
            'sys_utterances': sys_utterances,
            'sys_utterance_groups': sys_utterance_groups,
            'states': states,
            'kb_found': kb_found,
        })
    return data


def load_ontology(fn):
    """
    加载对话数据集中的本体
    :param fn:本体数据集的文件路径
    :return:返回整理好的本体和本体索引
    """
    with open(fn) as file:
        data = json.load(file)

    onto = {}
    onto_idx = defaultdict(dict)
    # 这里获取用户告知系统的信息
    inform_data = data['informable']

    for key, values in inform_data.items():
        onto[key] = values + ['dontcare']
        onto_idx[key]['dontcare'] = 0
        for value in values:
            onto_idx[key][value] = len(onto_idx[key])

        key = key + '_req'
        onto[key] = values + ['dontcare']
        onto_idx[key] = {
            'dontcare': 0,
            'care': 1,
        }

    req_data = data['requestable']
    for key in req_data:
        key = key + '_req'
        onto[key] = ['dontcare']
        onto_idx[key] = {
            'dontcare': 0,
            'care': 1,
        }

    return onto, onto_idx


class DataLoader:
    """
    对话数据加载工具类
    """

    def __init__(self, dialogues, max_length, tokenizer, onto, onto_idx, max_train_data_size, kb_fonud_len=5,
                 mode='train'):
        self.dialogues = dialogues
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cur = 0
        self.onto = onto
        self.onto_idx = onto_idx
        self.kb_found_len = kb_fonud_len
        self.max_train_data_size = max_train_data_size
        self.mode = mode

    def get_vocabs(self):
        """
        获取对话数据集中的token集合，分为user和system两个token集合
        :return: user和system两个token集合
        """
        vocabs = []
        sys_vocabs = []
        for dialogue in self.dialogues:
            for s in dialogue['usr_utterances']:
                vocabs.extend(self._sent_normalize(s))
            for s in dialogue['sys_utterances']:
                sys_vocabs.extend(self._sent_normalize(s))
        return set(vocabs), set(sys_vocabs)

    def __len__(self):
        sum = 0
        if self.max_train_data_size == 0:
            for dialogue in self.dialogues:
                sum += len(dialogue['usr_utterances'])
        else:
            for i in range(self.max_train_data_size):
                sum += len(self.dialogues[i]['usr_utterances'])
        return sum

    def _sent_normalize(self, sent):
        """
        分词器
        :param sent: 语句
        :return: 语句序列
        """
        tokenizer = RegexpTokenizer(r'<[a-z][.\w]+>|[^<]+')
        return tokenize_en(sent=sent.lower(), tokenizer=tokenizer)

    def _get(self, i):
        """
        获取整理对话数据集中的第i个对话的相关数据，整理
        至对应格式，并统一将数据类型转成tf.int64
        :param i: 第i个对话数据
        :return: 整理好的对话数据
        """
        dialogue = self.dialogues[i]
        usr_utterances = [self._gen_utterance_seq(self.tokenizer, s) for s in dialogue['usr_utterances']]
        usr_utterances = tf.convert_to_tensor(pad_sequence(seqs=usr_utterances, max_len=self.max_length),
                                              dtype=tf.int64)
        states = self._gen_state_vectors(dialogue['states'])
        kb_indicator = [[0] if x == 0 else [1] for x in dialogue['kb_found']]
        sys_utterances = [self._gen_utterance_seq(self.tokenizer, s) for s in dialogue['sys_utterances']]
        sys_utterances = [tf.reshape(tf.convert_to_tensor(utt, dtype=tf.int64), [1, -1]) for utt in sys_utterances]
        sys_utterance_groups = tf.convert_to_tensor(dialogue['sys_utterance_groups'], dtype=tf.int64)

        return dialogue['usr_utterances'], dialogue['sys_utterances'], \
               dialogue['kb_found'], usr_utterances, sys_utterances, \
               states, kb_indicator, sys_utterance_groups

    def _gen_utterance_seq(self, tokenizer, utterance):
        """
        将语句转成token索引向量
        :param tokenizer: 索引字典
        :param utterance: 语句
        :return: 返回转换好的向量
        """
        utterance = self._sent_normalize(utterance)
        utterance = [tokenizer.get(x, 0) for x in utterance]
        return utterance

    def _gen_state_vectors(self, states):
        """
        将状态序列中槽位值转成Tensor序列
        :param states: 状态列表
        :return: 整理好的状态张量
        """
        state_vectors = {slot: tf.cast(tf.zeros(len(states)), dtype=tf.float32).numpy() for slot in self.onto}
        for t, states_at_time_t in enumerate(states):
            for s, v in states_at_time_t:
                if v == 'center':
                    v = 'centre'
                state_vectors[s][t] = self.onto_idx[s][v]
        return state_vectors

    def __iter__(self):
        return self

    def reset(self):
        self.cur = 0

    def next(self):
        """
        移动到下一个对话，如果运行到test数据集，直接停止
        :return: 返回对应对话的数据
        """
        ret = self._get(self.cur)
        self.cur += 1
        # 没运行完一个epoch，就直接乱进行下一个epoch
        if self.cur > self.max_train_data_size and not self.max_train_data_size == 0:
            self.cur = 0
        if self.cur == len(self.dialogues):
            if self.mode == 'test':
                raise StopIteration()
            random.shuffle(self.dialogues)
            self.cur = 0

        return ret


def load_data(dialogues_train, max_length, kb_fn, ontology_fn, tokenizer, max_train_data_size, kb_indicator_len):
    """
    加载对原始数据、本体数据、database数据处理好的数据集
    :param dialogues_train: 原始对话数据路径
    :param kb_fn: database数据路径
    :param ontology_fn: 本体数据路径
    :param tokenizer: token
    :param kb_indicator_len: kb指针长度
    """
    kb = load_kb(kb_fn, 'name')

    dialogue_data = load_dialogs(dialogues_train, kb)
    onto, onto_idx = load_ontology(ontology_fn)
    kb_found_len = kb_indicator_len - 2

    return DataLoader(dialogues=dialogue_data, max_length=max_length, tokenizer=tokenizer, onto=onto, onto_idx=onto_idx,
                      max_train_data_size=max_train_data_size, kb_fonud_len=kb_found_len)
