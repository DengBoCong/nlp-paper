import os
import sys
import time
import tensorflow as tf
import model.model as task
from common.kb import load_kb

sys.path.append(sys.path[0][:-10])
from model.chatter import Chatter
import common.data_utils as _data
from common.common import CmdParser
import config.get_config as _config
from common.pre_treat import preprocess_raw_task_data


class TaskChatter(Chatter):
    """
    Task模型的聊天器
    """

    def __init__(self, checkpoint_dir, beam_size):
        super().__init__(checkpoint_dir, beam_size)
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.optimizer = tf.keras.optimizers.RMSprop()
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')

    def _init_loss_accuracy(self):
        print('待完善')

    def _train_step(self, inp, tar, step_loss):
        print('待完善')

    def _create_predictions(self, inputs, dec_input, t):
        print('待完善')

    def train(self, dict_fn, data_fn, start_sign, end_sign, max_train_data_size):
        _, _, lang_tokenizer = _data.load_dataset(dict_fn=dict_fn, data_fn=data_fn, start_sign=start_sign,
                                                  end_sign=end_sign, max_train_data_size=max_train_data_size)
        data_load = _data.load_data(_config.dialogues_train, _config.max_length, _config.database, _config.ontology,
                                    lang_tokenizer.word_index, _config.max_train_data_size, _config.kb_indicator_len)

        model = task.task(_config.units, data_load.onto,
                          _config.vocab_size, _config.embedding_dim, _config.max_length)

        checkpoint = tf.train.Checkpoint(model=model, optimizer=self.optimizer)
        ckpt = tf.io.gfile.listdir(self.checkpoint_dir)
        if ckpt:
            checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir)).expect_partial()

        sample_sum = len(data_load)
        for epoch in range(_config.epochs):
            print('Epoch {}/{}'.format(epoch + 1, _config.epochs))
            start_time = time.time()

            batch_sum = 0

            while (True):
                _, _, _, usr_utts, _, state_gt, kb_indicator, _ = data_load.next()
                if data_load.cur == 0:
                    break
                kb_indicator = tf.convert_to_tensor(kb_indicator)
                with tf.GradientTape() as tape:
                    state_preds = model(inputs=[usr_utts, kb_indicator])
                    loss = 0
                    for key in state_preds:
                        loss += tf.keras.losses.SparseCategoricalCrossentropy(
                            from_logits=True)(state_gt[key], state_preds[key])
                gradients = tape.gradient(loss, model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                self.train_loss(loss)
                kb = load_kb(_config.database, "name")

                batch_sum = batch_sum + len(usr_utts)
                print('\r', '{}/{} [==================================]'.format(batch_sum, sample_sum), end='',
                      flush=True)
            step_time = (time.time() - start_time)
            sys.stdout.write(' - {:.4f}s/step - loss: {:.4f}\n'
                             .format(step_time, self.train_loss.result()))
            sys.stdout.flush()
            checkpoint.save(file_prefix=self.checkpoint_prefix)
        print('训练结束')


def main():
    parser = CmdParser(version='%task chatbot V1.0')
    parser.add_option("-t", "--type", action="store", type="string",
                      dest="type", default="pre_treat",
                      help="execute type, pre_treat/train/chat")
    (options, args) = parser.parse_args()

    chatter = TaskChatter(checkpoint_dir=_config.task_train_data, beam_size=_config.beam_size)

    if options.type == 'train':
        chatter.train(dict_fn=_config.dict_fn,
                      data_fn=_config.dialogues_tokenized,
                      start_sign='<sos>',
                      end_sign='<eos>',
                      max_train_data_size=0)
    elif options.type == 'chat':
        print('Agent: 你好！结束聊天请输入ESC。')
        while True:
            req = input('User: ')
            if req == 'ESC':
                print('Agent: 再见！')
                exit(0)
            # response = chatter.respond(req)
            response = '待完善'
            print('Agent: ', response)
    elif options.type == 'pre_treat':
        preprocess_raw_task_data(raw_data=_config.dialogues_train,
                                 tokenized_data=_config.dialogues_tokenized,
                                 semi_dict=_config.semi_dict,
                                 database=_config.database,
                                 ontology=_config.ontology)
    else:
        parser.error(msg='')


if __name__ == "__main__":
    """
    TaskModel入口：指令需要附带运行参数
    cmd：python task_chatter.py -t/--type [执行模式]
    执行类别：pre_treat/train/chat

    chat模式下运行时，输入exit即退出对话
    """
    main()
