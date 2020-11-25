import os
import json

seq2seq_config = os.path.dirname(__file__) + r'\model_config.json'
path = os.path.dirname(__file__)[:-6]


def get_config_json(config_file='main.json'):
    with open(config_file, 'r') as file:
        return json.load(file)


def config(config_file=seq2seq_config):
    return get_config_json(config_file=config_file)


conf = {}

conf = config()

# task模型相关配置
epochs = conf['epochs']
vocab_size = conf['vocab_size']
beam_size = conf['beam_size']
embedding_dim = conf['embedding_dim']
max_length = conf['max_length']
units = conf['units']
task_train_data = path + conf['train_data']
sent_groups = path + conf['sent_groups']  # 含插槽的句子组合
database = path + conf['database']
ontology = path + conf['ontology']
semi_dict = path + conf['semi_dict']
dialogues_train = path + conf['dialogues_train']
dict_fn = path + conf['dict_fn']
dialogues_tokenized = path + conf['tokenized_data']
kb_indicator_len = conf['kb_indicator_len']
max_train_data_size = conf['max_train_data_size']
