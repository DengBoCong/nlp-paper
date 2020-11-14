import os
import sys
import time
import copy
import torch
from common.utils import BeamSearch
import common.data_utils as data_utils


class Chatter(object):
    """"
    面向使用者的聊天器基类
    该类及其子类实现和用户间的聊天，即接收聊天请求，产生回复。
    不同模型或方法实现的聊天子类化该类。
    """

    def __init__(self, checkpoint_dir, beam_size, max_length):
        """
        聊天器初始化，用于加载模型
        """
        self.max_length = max_length
        self.checkpoint_dir = checkpoint_dir
        self.beam_search_container = BeamSearch(
            beam_size=beam_size,
            max_length=max_length,
            worst_score=0
        )

    def _init_loss_accuracy(self):
        """
        初始化损失
        """
        pass

    def _train_step(self, inp, tar, step_loss):
        """
        模型训练步方法，需要返回时间步损失
        """
        pass

    def _create_predictions(self, inputs, dec_input, t):
        """
        使用模型预测下一个Token的id
        """
        pass

    def train(self, epochs, data_fn, max_train_data_size):
        """
        对模型进行训练
        """
        print('训练开始，正在准备数据中...')
        loader, steps_per_epoch = data_utils.load_data(dict_fn=self.dict_fn,
                                                       data_fn=data_fn,
                                                       max_train_data_size=max_train_data_size,
                                                       checkpoint_dir=self.checkpoint_dir,
                                                       batch_size=self.batch_size,
                                                       start_sign=self.start_sign, end_sign=self.end_sign,
                                                       max_length=self.max_length)

        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch + 1, epochs))
            start_time = time.time()
            batch_sum = 0
            sample_sum = 0
            epoch_loss = 0

            for (batch, (inp, tar, weight)) in enumerate(loader, 0):
                inp = inp.permute(1, 0)
                tar = tar.permute(1, 0)
                if inp.size()[1] != self.batch_size:
                    break
                self.optimizer.zero_grad()
                predictions = self._train_step(inp, tar, weight)
                predictions = torch.reshape(predictions[1:], shape=[-1, predictions.shape[-1]])
                tar = torch.reshape(tar[1:], shape=[-1])
                loss = self.criterion(predictions, tar)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                batch_sum = batch_sum + inp.size()[1]
                sample_sum = steps_per_epoch * inp.size()[1]
                print('\r', '{}/{} [==================================]'.format(batch_sum, sample_sum), end='',
                      flush=True)

            step_time = (time.time() - start_time)
            sys.stdout.write(' - {:.4f}s/step - loss: {:.4f}\n'
                             .format(step_time, epoch_loss / sample_sum))
            sys.stdout.flush()
            save_epoch = self.epoch_iterator + epoch + 1
            torch.save({
                'epoch': save_epoch,
                'encoder_state_dict': self.encoder.state_dict(),
                'decoder_state_dict': self.decoder.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': epoch_loss / sample_sum
            }, self.checkpoint_dir + '\\seq2seq-{}.pth'.format(save_epoch))
            with open(self.checkpoint_dir + '\\checkpoint.txt', 'w', encoding='utf-8') as file:
                file.write(str(save_epoch))

        print('训练结束')

    def respond(self, req):
        # 对req进行初步处理
        inputs, dec_input = data_utils.preprocess_request(sentence=req, start_sign=self.start_sign,
                                                          end_sign=self.end_sign, token=self.token,
                                                          max_length=self.max_length)

        self.beam_search_container.init_all_inner_variables(inputs=inputs, dec_input=dec_input)
        inputs, dec_input = self.beam_search_container.expand_beam_size_inputs()
        for t in range(self.max_length):
            predictions = self._create_predictions(inputs, dec_input)
            self.beam_search_container.add(predictions=predictions, end_sign=self.token.get(self.end_sign))
            if self.beam_search_container.beam_size == 0:
                break

            inputs, dec_input = self.beam_search_container.expand_beam_size_inputs()
            dec_input = torch.unsqueeze(dec_input[:, -1], dim=-1)
        beam_search_result = self.beam_search_container.get_result(top_k=1)
        result = ''
        # 从容器中抽取序列，生成最终结果
        for i in range(len(beam_search_result)):
            temp = beam_search_result[i].numpy()
            text = data_utils.sequences_to_texts(temp, self.token)
            text[0] = text[0].replace(self.start_sign, '').replace(self.end_sign, '').replace(' ', '')
            result = '<' + text[0] + '>' + result
        return result
