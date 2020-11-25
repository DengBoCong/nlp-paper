import json
from functools import reduce
from collections import defaultdict


def load_kb(kb_fn, primary):
    with open(kb_fn) as f:
        data = json.load(f)

    kb = KnowledgeBase(data[0].keys(), primary)

    for obj in data:
        kb.add(obj)

    return kb


class KnowledgeBase:
    """
    提供基于知识检索的API
    """

    def __init__(self, columns, primary):
        self.columns = columns
        self.primary = primary
        self.index = {k: defaultdict(list) for k in self.columns}
        self.objs = {}

    def add(self, obj):
        """
        添加一个知识对象到KB中
        """
        for key, value in obj.items():
            self.index[key][value].append(obj[self.primary])

        self.objs[obj[self.primary]] = obj

    def get(self, primary):
        """
        通过key查询知识对象
        """
        return self.objs[primary]

    def search(self, key, value):
        return set(self.index[key][value]);

    def search_multi(self, kvs):
        """
        通过key批量查询知识对象
        :params kvs: key和value的列表，使用lambda表达式进行累积操作
        """
        ret = reduce(lambda y, x: y & set(self.index[x[0]][x[1]])
        if y is not None else set(self.index[x[0]][x[1]]), kvs, None)
        return ret if ret is not None else set()
