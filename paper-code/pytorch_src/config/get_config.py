import os
import json

seq2seq_config = os.path.dirname(__file__) + r'\config.json'
path = os.path.dirname(__file__)[:-18]


def get_config_json(config_file='main.json'):
    with open(config_file, 'r') as file:
        return json.load(file)


def config(config_file=seq2seq_config):
    return get_config_json(config_file=config_file)


conf = {}

conf = config()

# 公共配置
BATCH_SIZE = conf['batch_size']
BUFFER_SIZE = conf['buffer_size']
data = path + conf['tokenized_data']  # 训练数据位置
resource_data = path + conf['resource_data']  # 原始数据位置
tokenized_data = path + conf['tokenized_data']  # 预处理之后数据位置
beam_size = conf['beam_size']  # beam_search大小
epochs = conf['epochs']  # 训练轮次
start_sign = conf['start_sign']
end_sign = conf['end_sign']
unk_sign = conf['unk_sign']
lccc_data = path + conf['lccc_data']
lccc_tokenized_data = path + conf['lccc_tokenized_data']
douban_tokenized_data = path + conf['douban_tokenized_data']
ubuntu_tokenized_data = path + conf['ubuntu_tokenized_data']
ubuntu_valid_data = path + conf['ubuntu_valid_data']
candidate_database = path + conf['candidate_database']

# seq2seq模型相关配置
seq2seq_units = conf['seq2seq']['units']
seq2seq_checkpoint = path + conf['seq2seq_checkpoint']  # 训练结果保存位置
seq2seq_dict_fn = path + conf['seq2seq_dict_fn']  # 字典保存位置
seq2seq_vocab_size = conf['seq2seq']['vocab_size']
seq2seq_embedding_dim = conf['seq2seq']['embedding_dim']
seq2seq_max_train_data_size = conf['seq2seq']['max_train_data_size']
seq2seq_max_length = conf['seq2seq']['max_length']  # 最大文本长度
seq2seq_dropout = conf['seq2seq']['dropout']

# transformer模型相关配置
transformer_vocab_size = conf['transformer']['vocab_size']
transformer_checkpoint = path + conf['transformer_checkpoint']  # 训练结果保存位置
transformer_num_layers = conf['transformer']['num_layers']
transformer_d_model = conf['transformer']['d_model']
transformer_num_heads = conf['transformer']['num_heads']
transformer_units = conf['transformer']['units']
transformer_dropout = conf['transformer']['dropout']
transformer_dict_fn = path + conf['transformer_dict_fn']  # 字典保存位置
transformer_max_train_data_size = conf['transformer']['max_train_data_size']
transformer_max_length = conf['transformer']['max_length']  # 最大文本长度

# smn模型相关配置
smn_checkpoint = path + conf['smn_checkpoint']
smn_dict_fn = path + conf['smn_dict_fn']
smn_embedding_dim = conf['smn']['embedding_dim']
smn_max_sentence = conf['smn']['max_sentence']
smn_max_utterance = conf['smn']['max_utterance']
smn_units = conf['smn']['units']
smn_vocab_size = conf['smn']['vocab_size']
smn_max_train_data_size = conf['smn']['max_train_data_size']
smn_learning_rate = conf['smn']['learning_rate']
smn_max_valid_data_size = conf['smn']['max_valid_data_size']
smn_max_database_size = conf['smn']['max_database_size']
