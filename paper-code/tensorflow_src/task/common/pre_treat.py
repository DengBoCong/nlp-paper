import re
import os
import json
from collections import defaultdict
from common.data_utils import tokenize_en
from nltk.tokenize import RegexpTokenizer


class Delexicalizer:
    """
    去词化器
    """

    def __init__(self, info_slots, semi_dict, values, replaces):
        self.info_slots = info_slots  # 所有informable槽位信息
        self.semi_dict = semi_dict  # 语句中槽位同义词替换字典
        self.values = values  # 数据库中所有requestable槽位信息
        self.replaces = replaces

        self.inv_info_slots = self._inverse_dict(self.info_slots, '%s')  # informable槽值对字典
        self.inv_values = self._inverse_dict(self.values, '<v.%s> ',
                                             func=lambda x: x.upper())  # requestable槽值对字典，槽位已同义化
        self.inv_semi_dict = self._inverse_dict(self.semi_dict, '%s')

        self.inv_semi_dict = {k: "<v.%s> " % self.inv_info_slots[v].upper()
        if v in self.inv_info_slots else "<s.%s> " % v.upper() for k, v in self.inv_semi_dict.items()}

        self.num_matcher = re.compile(r' \d{1,2}([., ])')
        self.post_matcher = re.compile(
            r'( [.]?c\.b[.]?[ ]?\d[ ]?[,]?[ ]?\d[.]?[ ]?[a-z][\.]?[ ]?[a-z][\.]?)|( cb\d\d[a-z]{2})')
        self.phone_matcher = re.compile(r'[ (](#?0)?(\d{10}|\d{4}[ ]\d{5,6}|\d{3}-\d{3}-\d{4})[ ).,]')
        self.street_matcher = re.compile(
            r' (([a-z]+)?\d{1,3}([ ]?-[ ]?\d+)? )?[a-z]+ (street|road|avenue)(, (city [a-z]+))?')

    def _inverse_dict(self, d, fmt="%s ", func=str):
        """
        将字典中key和value转换工具
        """
        inv = {}
        for k, vs in d.items():
            for v in vs:
                inv[v.lower()] = fmt % (func(k))
        return inv

    def delex(self, sent):
        """
        将句子去词化
        """
        sent = ' ' + sent.lower()
        sent = self.post_matcher.sub(' <v.POSTCODE> ', sent)
        sent = " , ".join(sent.split(","))

        # for r, v in self.replaces:
        #     sent = sent.replace(" " + r + " ", " " + v + " ")

        sent = sent.replace('  ', ' ')

        sent = self.phone_matcher.sub(' <v.PHONE> ', sent)
        for v in sorted(self.inv_values.keys(), key=len, reverse=True):
            sent = sent.replace(v, self.inv_values[v])

        sent = self.street_matcher.sub(' <v.ADDRESS> ', sent)
        for v in sorted(self.inv_semi_dict.keys(), key=len, reverse=True):
            sent = sent.replace(v, self.inv_semi_dict[v])

        sent = self.num_matcher.sub(' <COUNT> ', sent)

        sent = sent.replace('  ', ' ')

        return sent.strip()


def create_delexicaliser(semi_dict_fn, kb_fn, onto_fn, req_slots=["address", "phone", "postcode", "name"]):
    """
    去词化器创建工具
    """
    semi_dict = defaultdict(list)
    values = defaultdict(list)

    with open(kb_fn) as file:
        kb = json.load(file)

    with open(semi_dict_fn) as file:
        semi_dict = json.load(file)

    with open(onto_fn) as file:
        onto_data = json.load(file)

    for entry in kb:
        for slot in req_slots:
            if slot in entry:
                values[slot].append(entry[slot])

    # slots = ["area", "food", "pricerange", "address", "phone", "postcode", "name"]
    return Delexicalizer(onto_data['informable'], semi_dict, values, '')


def convert_delex(diag_fn, delex_fn, output_fn):
    """
    系统回复槽位生成，将结果保存在一个文件中
    """
    with open(diag_fn) as file:
        dialogues = json.load(file)

    with open(delex_fn) as file:
        delexed = file.readlines()

    delex_iter = iter(delexed)
    for diag_idx, diag in enumerate(dialogues):
        for turn_idx, turn in enumerate(diag['diaglogue']):
            dialogues[diag_idx]['diaglogue'][turn_idx]['system_transcript'] = next(delex_iter).replace("\t", "").strip()

    with open(output_fn, 'w', encoding='utf-8') as file:
        file.write(json.dumps(dialogues, indent=4, ensure_ascii=False))


def preprocess_raw_task_data(raw_data, tokenized_data, semi_dict, database, ontology):
    """
    专门针对task标注数据的client和agent对话的token数据处理
    :param raw_data:  原始对话数据路径
    :param tokenized_data: 生成token数据保存路径
    :return:
    """
    # 首先判断原数据集是否存在，不存在则退出
    if not os.path.exists(raw_data):
        print('数据集不存在，请添加数据集!')
        exit()

    pairs = []
    delex = create_delexicaliser(semi_dict, database, ontology)
    tokenizer = RegexpTokenizer(r'<[a-z][.\w]+>|[^<]+')

    with open(raw_data, encoding='utf-8') as file:
        pair_count = 0
        dialogues = json.load(file)

        for diag in dialogues:
            for turn in diag['dialogue']:
                user = tokenize_en(turn['transcript'].lower(), tokenizer)
                system = tokenize_en(delex.delex(turn['system_transcript']).lower(), tokenizer)
                pairs.append([user, system])
                pair_count += 1
                if pair_count % 1000 == 0:
                    print('已处理：', pair_count, '个问答对')

    print('读取完毕，处理中...')

    train_tokenized = open(tokenized_data, 'w', encoding='utf-8')
    for i in range(len(pairs)):
        train_tokenized.write(' '.join(pairs[i][0]) + '\t' + ' '.join(pairs[i][1]) + '\n')
        if i % 1000 == 0:
            print('处理进度：', i)

    train_tokenized.close()
