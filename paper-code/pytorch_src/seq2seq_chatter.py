import os
import sys
import torch
import random
import torch.nn as nn
import torch.optim as optim
sys.path.append(sys.path[0][:-12])
import common.data_utils as data_utils
from common.utils import CmdParser
from model.chatter import Chatter
import model.seq2seq as seq2seq
import config.get_config as get_config
from common.pre_treat import preprocess_raw_lccc_data


class Seq2SeqChatter(Chatter):
    """
    Seq2Seq模型的聊天类
    """

    def __init__(self, execute_type: str, batch_size: int, embedding_dim: int, units: int, dropout: float,
                 checkpoint_dir: str, beam_size: int, vocab_size: int, dict_fn: str, max_length: int,
                 start_sign: str, end_sign: str):
        """
        Seq2Seq聊天器初始化，用于加载模型
        """
        super().__init__(checkpoint_dir, beam_size, max_length)
        self.dict_fn = dict_fn
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.epoch_iterator = 0
        self.start_sign = start_sign
        self.end_sign = end_sign
        self.encoder = seq2seq.Encoder(vocab_size, embedding_dim, units, units, dropout)
        attention = seq2seq.BahdanauAttention(units)
        self.decoder = seq2seq.Decoder(vocab_size, embedding_dim, units, units, dropout, attention)
        self.optimizer = optim.Adam([{'params': self.encoder.parameters(), 'lr': 1e-3},
                                     {'params': self.decoder.parameters(), 'lr': 1e-3}])
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        if execute_type == "chat":
            print('正在从“{}”处加载字典...'.format(dict_fn))
            self.token = data_utils.load_token_dict(dict_fn=dict_fn)
        print('正在检查是否存在检查点...')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_path = checkpoint_dir + '\\checkpoint.txt'

        if not os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'w', encoding='utf-8') as file:
                print("没有检查到检查点，已创建checkpoint记录文本")
            if execute_type == "train":
                print('正在train模式...')
            else:
                print('请先执行train模式，再进入chat模式')
                exit(0)
        else:
            with open(checkpoint_path, 'r', encoding='utf-8') as file:
                lines = file.read().strip().split('\n')
                version = lines[0]
                if version is not "":
                    version = int(version)
                    checkpoint = torch.load(checkpoint_dir + '\\seq2seq-{}.pth'.format(version))
                    self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
                    self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    self.epoch_iterator = checkpoint['epoch']
                    print("检测到检查点，已成功加载检查点")

    def _train_step(self, inp: torch.Tensor, tar: torch.Tensor, weight: torch.Tensor,
                    teacher_forcing_ratio: float = 0.5):
        enc_outputs, enc_state = self.encoder(inp)
        dec_state = enc_state
        dec_input = tar[0, :]
        outputs = torch.zeros(self.max_length, self.batch_size, self.vocab_size)

        for t in range(1, self.max_length):
            predictions, dec_hidden = self.decoder(dec_input, dec_state, enc_outputs)
            outputs[t] = predictions
            teacher_force = random.random() < teacher_forcing_ratio
            top_first = torch.argmax(predictions, dim=-1)
            dec_input = (tar[t] if teacher_force else top_first)

        return outputs

    def _create_predictions(self, inputs, dec_input):
        with torch.no_grad():
            inputs = inputs.permute(1, 0)  # [40, 1]
            dec_input = dec_input.permute(1, 0)  # [1, 1]
            enc_out, enc_hidden = self.encoder(inputs)  # [40, 1, 2048]  [1, 1024]
            dec_hidden = enc_hidden
            dec_input = dec_input[0, :]  # [1,1]
            predictions, _ = self.decoder(dec_input, dec_hidden, enc_out)
        return predictions


def get_chatter(execute_type, batch_size, embedding_dim, units, dropout,
                checkpoint_dir, beam_size, vocab_size, dict_fn, max_length, start_sign, end_sign):
    """
    初始化要使用的聊天器
    Args:
        execute_type: 程序执行的模式类别
        batch_size: 批次大小
        embedding_dim: 词嵌入特征维度
        units: GRU特征单元数
        dropout: 采样率
        checkpoint_dir: 检查点保存路径
        beam_size: beam_size大小
        vocab_size: 词汇量大小
        dict_fn: 字典保存位置
        max_length: 单个句子最大长度
        start_sign: 开始标记
        end_sign: 结束标记
    Returns:
        chatter: 实例化的Seq2SeqChatter聊天器
    """
    chatter = Seq2SeqChatter(execute_type=execute_type,
                             batch_size=batch_size,
                             embedding_dim=embedding_dim,
                             units=units,
                             dropout=dropout,
                             checkpoint_dir=checkpoint_dir,
                             beam_size=beam_size,
                             vocab_size=vocab_size,
                             dict_fn=dict_fn,
                             max_length=max_length,
                             start_sign=start_sign,
                             end_sign=end_sign)
    return chatter


def main():
    parser = CmdParser(version='%seq2seq chatbot V1.0')
    parser.add_option("-t", "--type", action="store", type="string",
                      dest="type", default="pre_treat",
                      help="execute type, pre_treat/train/chat")
    (options, args) = parser.parse_args()

    if options.type == 'train':
        chatter = get_chatter(execute_type=options.type, batch_size=get_config.BATCH_SIZE,
                              embedding_dim=get_config.seq2seq_embedding_dim, units=get_config.seq2seq_units,
                              dropout=get_config.seq2seq_dropout, checkpoint_dir=get_config.seq2seq_checkpoint,
                              beam_size=get_config.beam_size, vocab_size=get_config.seq2seq_vocab_size,
                              dict_fn=get_config.seq2seq_dict_fn, max_length=get_config.seq2seq_max_length,
                              start_sign=get_config.start_sign, end_sign=get_config.end_sign)
        chatter.train(epochs=get_config.epochs, data_fn=get_config.lccc_tokenized_data,
                      max_train_data_size=get_config.seq2seq_max_train_data_size)
    elif options.type == 'chat':
        print('待完善')
        chatter = get_chatter(execute_type=options.type, batch_size=get_config.BATCH_SIZE,
                              embedding_dim=get_config.seq2seq_embedding_dim, units=get_config.seq2seq_units,
                              dropout=get_config.seq2seq_dropout, checkpoint_dir=get_config.seq2seq_checkpoint,
                              beam_size=get_config.beam_size, vocab_size=get_config.seq2seq_vocab_size,
                              dict_fn=get_config.seq2seq_dict_fn, max_length=get_config.seq2seq_max_length,
                              start_sign=get_config.start_sign, end_sign=get_config.end_sign)
        print("Agent: 你好！结束聊天请输入ESC。")
        while True:
            req = input("User: ")
            if req == "ESC":
                print("Agent: 再见！")
                exit(0)
            response = chatter.respond(req=req)
            print("Agent: ", response)
    elif options.type == 'pre_treat':
        preprocess_raw_lccc_data(raw_data=get_config.lccc_data,
                                 tokenized_data=get_config.lccc_tokenized_data)
    else:
        parser.error(msg='')


if __name__ == "__main__":
    """
    Seq2Seq入口：指令需要附带运行参数
    cmd：python seq2seq2_chatter.py -t/--type [执行模式]
    执行类别：pre_treat/train/chat

    chat模式下运行时，输入ESC即退出对话
    """
    main()
