import os
import copy
import torch
from optparse import OptionParser


class CmdParser(OptionParser):
    def error(self, msg):
        print('Error!提示信息如下：')
        self.print_help()
        self.exit(0)

    def exit(self, status=0, msg=None):
        exit(status)


class BeamSearch(object):
    """
    BeamSearch使用说明：
    1.首先需要将问句编码成token向量并对齐，然后调用init_input方法进行初始化
    2.对模型要求能够进行批量输入
    3.BeamSearch使用实例已经集成到Chatter中，如果不进行自定义调用，
    可以将聊天器继承Chatter，在满足上述两点的基础之上设计_create_predictions方法，并调用BeamSearch
    """

    def __init__(self, beam_size, max_length, worst_score):
        """
        初始化BeamSearch的序列容器
        """
        self.remain_beam_size = beam_size  # 保存原始beam大小，用于重置
        self.max_length = max_length - 1
        self.remain_worst_score = worst_score  # 保留原始worst_score，用于重置

    def __len__(self):
        """
        已存在BeamSearch的序列容器的大小
        """
        return len(self.container)

    def init_all_inner_variables(self, inputs, dec_input):
        """
        用来初始化输入
        Args:
            inputs: encoder的输入
            dec_input: decoder的输入
        Returns:
        """
        self.container = []  # 保存中间状态序列的容器，元素格式为(score, sequence)类型为(float, [])
        self.container.append((1, dec_input))
        self.inputs = inputs
        self.dec_inputs = dec_input
        self.beam_size = self.remain_beam_size  # 新一轮中，将beam_size重置为原beam大小
        self.worst_score = self.remain_worst_score  # 新一轮中，worst_score重置
        self.result = []  # 用来保存已经遇到结束符的序列

    def expand_beam_size_inputs(self):
        """
        用来动态的更新模型的inputs和dec_inputs，以适配随着Beam Search
        结果的得出而变化的beam_size
        Args:
        Returns: 返回扩展至beam_size大小的enc_inputs和dec_inputs
        """
        # 生成多beam输入
        inputs = self.inputs
        for i in range(len(self) - 1):
            inputs = torch.cat((inputs, self.inputs), dim=0)
        requests = inputs
        # 生成多beam的decoder的输入
        temp = self.container[0][1]
        for i in range(1, len(self)):
            temp = torch.cat((temp, self.container[i][1]), dim=0)
        self.dec_inputs = copy.deepcopy(temp)
        return requests, self.dec_inputs

    def _reduce_end(self, end_sign):
        """
        当序列遇到了结束token，需要将该序列从容器中移除
        Args:
            end_sign: 结束标记
        Returns:
        """
        for idx, (s, dec) in enumerate(self.container):
            temp = dec.numpy()
            if temp[0][-1] == end_sign:
                self.result.append((self.container[idx][0], self.container[idx][1]))
                del self.container[idx]
                self.beam_size -= 1

    def add(self, predictions, end_sign):
        """
        往容器中添加预测结果，在本方法中对预测结果进行整理、排序的操作
        Args:
            predictions: 传入每个时间步的模型预测值
        Returns:
        """
        remain = copy.deepcopy(self.container)
        self.container.clear()
        predictions = predictions
        for i in range(self.dec_inputs.shape[0]):
            for _ in range(self.beam_size):
                token_index = torch.argmax(input=predictions[i], dim=0)
                # 计算分数
                score = remain[i][0] * predictions[i][token_index]
                predictions[i][token_index] = 0
                # 判断容器容量以及分数比较
                if len(self) < self.beam_size or score > self.worst_score:
                    self.container.append(
                        (score, torch.cat((remain[i][1], torch.reshape(token_index, shape=[1, -1])), dim=-1)))
                    if len(self) > self.beam_size:
                        sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.container)])
                        del self.container[sorted_scores[0][1]]
                        self.worst_score = sorted_scores[1][0]
                    else:
                        self.worst_score = min(score, self.worst_score)
        self._reduce_end(end_sign=end_sign)

    def get_result(self, top_k=1):
        """
        获取最终beam个序
        Args:
            top_k: 返回序列数量
        Returns: beam个序列
        """
        results = [element[1] for element in sorted(self.result)[-top_k:]]
        # 每轮回答之后，需要重置容器内部的相关变量值
        return results
