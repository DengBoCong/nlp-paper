from typing import Any
from typing import AnyStr
from typing import NoReturn
from typing import List

def slice_neg_pos_data(data_path: AnyStr, save_path: AnyStr, if_self: bool = False) -> NoReturn:
    """ 文本匹配中句子对数据增强

    :param data_path: 原始数据集路径
    :param save_path: 数据增强瘦的数据保存路径
    :param if_self: 是否使用自身pairs
    :return:
    """
    remain = dict()
    res = dict()
    positive = list()
    negative = list()
    count = 0
    negative_set = set()

    def find(key: AnyStr) -> AnyStr:
        if key != remain[key]:
            remain[key] = find(remain[key])
        return remain[key]

    def union(key1: AnyStr, key2: AnyStr) -> NoReturn:
        remain[find(key2)] = find(key1)

    with open(data_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip().strip("\n").split("\t")
            if len(line) != 3:
                continue
            if line[2] == "1":
                if remain.get(line[0], "a") == "a":
                    remain[line[0]] = line[0]
                if remain.get(line[1], "a") == "a":
                    remain[line[1]] = line[1]
                positive.append([line[0], line[1]])
            elif line[2] == "0":
                negative.append([line[0], line[1]])
                if if_self:
                    negative_set.add(line[0])
                    negative_set.add(line[1])

        for first_query, second_query in positive:
            union(first_query, second_query)

        for first_query, second_query in positive:
            if res.get(find(first_query), "a") == "a":
                res[find(first_query)] = set()
            res[find(first_query)].add(first_query)
            res[find(first_query)].add(second_query)

    with open(save_path, "a", encoding="utf-8") as save_file:
        print("正在处理正样本")
        for key, value in res.items():
            elements = list(value)
            length = len(elements)
            for i in range(length):
                for j in range(i + 1, length):
                    save_file.write(elements[i] + "\t" + elements[j] + "\t1" + "\n")
                    save_file.write(elements[j] + "\t" + elements[i] + "\t1" + "\n")
                    count += 2
            if if_self:
                for element in elements:
                    save_file.write(element + "\t" + element + "\t1" + "\n")
                    count += 1

            if count % 1000 == 0:
                print("\r已处理 {} 条query-pairs".format(count), end="", flush=True)

        print("\n正在处理负样本")
        count = 0
        for first, second in negative:
            save_file.write(first + "\t" + second + "\t0" + "\n")
            save_file.write(second + "\t" + first + "\t0" + "\n")

            count += 2
            if count % 1000 == 0:
                print("\r已处理 {} 条query-pairs".format(count), end="", flush=True)

        if if_self:
            print("\n正在处理负样本转化正样本")
            count = 0
            for ne_element in negative_set:
                save_file.write(ne_element + "\t" + ne_element + "\t1" + "\n")

                count += 1
                if count % 1000 == 0:
                    print("\r已处理 {} 条query-pairs".format(count), end="", flush=True)