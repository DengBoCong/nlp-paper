import tensorflow as tf
import model.tracker as tracker
import warnings

warnings.filterwarnings("ignore")


def get_slots_tracker(onto, units):
    """
    根据inform和request的槽位的个数，生成对应的tracker
    :param onto: 处理过的本体数据集
    :param state_tracker_hidden_size: 处理过的本体数据集
    """
    slot_trackers = {}
    slot_len_sum = 0

    for slot in onto:
        if len(onto[slot]) > 2:
            slot_trackers[slot] = tracker.inform_slot_tracker(
                units=units, n_choices=len(onto[slot]), name="inform_slot_tracker_{}".format(slot))
            slot_len_sum += len(onto[slot])
        else:
            slot_trackers[slot] = tracker.request_slot_tracker(
                units=units, name="request_slot_tracker_{}".format(slot))
            slot_len_sum += 2

    return slot_trackers, slot_len_sum


def task_encoder(units, vocab_size, embedding_dim, name="task_encoder"):
    """
    task的encoder，使用双向LSTM对用户语句进行编码，输出序列和合并后的隐藏层
    """
    inputs = tf.keras.Input(shape=(None,), name='task_encoder_inputs')
    embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)(inputs)
    output, forward_state, backward_state = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(units=units, return_sequences=True,
                            return_state=True, dropout=0.9), merge_mode='concat')(embedding)

    state = tf.concat([forward_state, backward_state], -1)
    return tf.keras.Model(inputs=inputs, outputs=[output, state], name=name)


def task(units, onto, vocab_size, embedding_dim, max_sentence_len, name="task_model"):
    """
    Task-Orient模型，使用函数式API实现，将encoder和decoder封装
    :param vocab_size:token大小
    """
    usr_utts = tf.keras.Input(shape=max_sentence_len, name="task_model_inputs")
    kb_indicator = tf.keras.Input(shape=1)
    _, encoder_hidden = task_encoder(units=units, vocab_size=vocab_size, embedding_dim=embedding_dim)(
        usr_utts)
    inputs = tf.concat([encoder_hidden, kb_indicator], -1)
    _, state = tracker.state_tracker(units=units, vocab_size=vocab_size, embedding_dim=embedding_dim)(
        inputs)
    slot_trackers, slot_len_sum = get_slots_tracker(onto=onto, units=units)
    state_pred = {slot: slot_trackers[slot](state) for slot in onto}

    return tf.keras.Model(inputs=[usr_utts, kb_indicator], outputs=state_pred, name=name)
