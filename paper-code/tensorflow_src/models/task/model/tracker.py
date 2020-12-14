import tensorflow as tf


def inform_slot_tracker(units, n_choices, name="inform_slot_tracker"):
    """
    informable插槽跟踪器，informable插槽是用户告知系统的信息，用
    来约束对话的一些条件，系统为了完成任务必须满足这些条件
    用来获得时间t的状态的槽值分布，比如price=cheap
    输入为状态跟踪器的输入'state_t'，输出为槽值分布'P(v_s_t| state_t)'
    """
    inputs = tf.keras.Input(shape=(units,), name="inform_slot_tracker_inputs")
    outputs = tf.keras.layers.Dense(units=n_choices)(inputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)


def request_slot_tracker(units, name="request_slot_tracker"):
    """
    requestable插槽跟踪器，requestable插槽是用户询问系统的信息
    用来获得时间t的状态的非分类插槽槽值分布，
    比如：
    address=1 (地址被询问)
    phone=0 (用户不关心电话号码)
    输入为状态跟踪器的输入'state_t'，输出为槽值二元分布'P(v_s_t| state_t)'
    """
    inputs = tf.keras.Input(shape=(units,), name="request_slot_tracker_inputs")
    outputs = tf.keras.layers.Dense(units=2)(inputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)


def state_tracker(units, vocab_size, embedding_dim, name="state_tracker"):
    inputs = tf.keras.Input(shape=(None,), name="state_tracker_inputs")
    embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)(inputs)
    output, state = tf.keras.layers.GRU(units=units,
                                        return_sequences=True,
                                        return_state=True,
                                        dropout=0.9)(inputs=embedding)
    return tf.keras.Model(inputs=[inputs], outputs=[output, state], name=name)
