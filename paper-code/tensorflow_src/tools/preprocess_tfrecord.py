import os
import jieba
import multiprocessing as mt
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from typing import Any
from typing import AnyStr
from typing import NoReturn
from typing import List

MAX_SENTENCE_LEN = 20  # 最大句子长度

def load_tokenizer(dict_path: AnyStr) -> Tokenizer:
    """ 加载分词器工具

    :param dict_path: 字典路径
    :return: 分词器
    """
    if not os.path.exists(dict_path):
        raise FileNotFoundError("字典不存在，请检查后重试！")

    with open(dict_path, "r", encoding="utf-8") as dict_file:
        json_string = dict_file.read().strip().strip("\n")
        tokenizer = tokenizer_from_json(json_string=json_string)

    return tokenizer

def preprocess_raw_data(data_path: AnyStr, record_data_path: AnyStr, dict_path: AnyStr,
                        max_len: Any, max_data_size: Any = 0, pair_size: Any = 3) -> NoReturn:
    """ 处理原始数据，并将处理后的数据保存为TFRecord格式

    :param data_path: 原始数据路径
    :param record_data_path: 分词好的数据路径
    :param dict_path: 字典保存路径
    :param max_len: 最大序列长度
    :param max_data_size: 最大处理数据量
    :param pair_size: 数据对大小，用于剔除不符合要求数据
    :return: 无返回值
    """
    first_queries = []
    second_queries = []
    labels = []
    count = 0
    tokenizer = load_tokenizer(dict_path=dict_path)

    with open(data_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip().strip("\n").split("\t")
            if line == "" or len(line) != pair_size:
                continue

            first = " ".join(jieba.cut(line[0]))
            second = " ".join(jieba.cut(line[1]))

            if len(first.split(" ")) == 0 or len(second.split(" ")) == 0:
                continue

            first_queries.append(first)
            second_queries.append(second)
            labels.append(int(line[2]) if pair_size == 3 else 0)
            count += 1
            if count % 100 == 0:
                print("\r已读取 {} 条query-pairs".format(count), end="", flush=True)
            if count == max_data_size:
                break
    first_queries_seq = tokenizer.texts_to_sequences(first_queries)
    second_queries_seq = tokenizer.texts_to_sequences(second_queries)

    first_queries_seq = tf.keras.preprocessing.sequence.pad_sequences(first_queries_seq,
                                                                      maxlen=max_len, dtype="int32", padding="post")
    second_queries_seq = tf.keras.preprocessing.sequence.pad_sequences(second_queries_seq,
                                                                       maxlen=max_len, dtype="int32", padding="post")

    writer = tf.data.experimental.TFRecordWriter(record_data_path)
    dataset = tf.data.Dataset.from_tensor_slices((first_queries_seq, second_queries_seq, labels))

    def generator():
        for first_query, second_query, label in dataset:
            example = tf.train.Example(features=tf.train.Features(feature={
                "first": tf.train.Feature(int64_list=tf.train.Int64List(value=first_query)),
                "second": tf.train.Feature(int64_list=tf.train.Int64List(value=second_query)),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            }))
            yield example.SerializeToString()

    print("\n正在写入数据，请稍后")
    serialized_dataset = tf.data.Dataset.from_generator(generator, output_types=tf.string, output_shapes=())
    writer.write(serialized_dataset)

    print("数据预处理完毕，TFRecord数据文件已保存！")


def preprocess_raw_data_not_tokenized(data_path: AnyStr, record_data_path: AnyStr,
                                      max_len: Any, max_data_size: Any = 0, pair_size: Any = 3) -> NoReturn:
    """ 处理原始数据，并将处理后的数据保存为TFRecord格式

    :param data_path: 原始数据路径
    :param record_data_path: 分词好的数据路径
    :param max_len: 最大序列长度
    :param max_data_size: 最大处理数据量
    :param pair_size: 数据对大小，用于剔除不符合要求数据
    :return: 无返回值
    """
    first_queries = []
    second_queries = []
    labels = []
    count = 0

    with open(data_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip().strip("\n").split("\t")
            if line == "" or len(line) != pair_size:
                continue

            first = line[0].split(" ")
            second = line[1].split(" ")

            if len(first) == 0 or len(second) == 0:
                continue

            first_queries.append(first)
            second_queries.append(second)
            labels.append(int(line[2]) if pair_size == 3 else 0)
            count += 1
            if count % 100 == 0:
                print("\r已读取 {} 条query-pairs".format(count), end="", flush=True)
            if count == max_data_size:
                break

    first_queries_seq = tf.keras.preprocessing.sequence.pad_sequences(first_queries, maxlen=max_len,
                                                                      dtype="int32", padding="post")
    second_queries_seq = tf.keras.preprocessing.sequence.pad_sequences(second_queries, maxlen=max_len,
                                                                       dtype="int32", padding="post")

    writer = tf.data.experimental.TFRecordWriter(record_data_path)
    dataset = tf.data.Dataset.from_tensor_slices((first_queries_seq, second_queries_seq, labels))

    def generator():
        for first_query, second_query, label in dataset:
            example = tf.train.Example(features=tf.train.Features(feature={
                "first": tf.train.Feature(int64_list=tf.train.Int64List(value=first_query)),
                "second": tf.train.Feature(int64_list=tf.train.Int64List(value=second_query)),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            }))
            yield example.SerializeToString()

    print("\n正在写入数据，请稍后")
    serialized_dataset = tf.data.Dataset.from_generator(generator, output_types=tf.string, output_shapes=())
    writer.write(serialized_dataset)

    print("数据预处理完毕，TFRecord数据文件已保存！")


def load_dataset(record_path: AnyStr, batch_size: Any, buffer_size: Any,
                 num_parallel_reads: Any = None, data_type: AnyStr = "train",
                 reshuffle_each_iteration: Any = True, drop_remainder: Any = True) -> tf.data.Dataset:
    """ 获取Dataset

    :param record_path:
    :param batch_size: batch大小
    :param buffer_size: 缓冲大小
    :param num_parallel_reads: 读取线程数
    :param data_type: 加载数据类型，train/valid
    :param reshuffle_each_iteration: 是否每个epoch打乱
    :param drop_remainder: 是否去除余数
    :return: 加载的Dataset
    """
    if not os.path.exists(record_path):
        raise FileNotFoundError("TFRecord文件不存在，请检查后重试")

    dataset = tf.data.TFRecordDataset(filenames=record_path, num_parallel_reads=num_parallel_reads)
    dataset = dataset.map(map_func=_parse_dataset_item, num_parallel_calls=mt.cpu_count())
    if data_type == "train":
        dataset = dataset.shuffle(
            buffer_size=buffer_size, reshuffle_each_iteration=reshuffle_each_iteration
        ).prefetch(tf.data.experimental.AUTOTUNE)

    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    return dataset


def _parse_dataset_item(example: tf.train.Example.FromString) -> tf.io.parse_single_example:
    """ 用于Dataset中的TFRecord序列化字符串恢复

    :param example: 序列化字符串
    :return: 恢复后的数据
    """
    features = {
        "first": tf.io.FixedLenFeature([MAX_SENTENCE_LEN], tf.int64,
                                       default_value=tf.zeros([MAX_SENTENCE_LEN], dtype=tf.int64)),
        "second": tf.io.FixedLenFeature([MAX_SENTENCE_LEN], tf.int64,
                                        default_value=tf.zeros([MAX_SENTENCE_LEN], dtype=tf.int64)),
        "label": tf.io.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }
    example = tf.io.parse_single_example(serialized=example, features=features)
    return example["first"], example["second"], example["label"]