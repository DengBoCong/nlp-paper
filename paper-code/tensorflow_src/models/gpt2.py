import tensorflow as tf


def creat_padding_mask(inputs):
    """
    对input中的padding单位进行mask
    :param inputs: 句子序列输入
    :return: 填充部分标记
    """
    mask = tf.cast(tf.math.equal(inputs, 0), dtype=tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]


def creat_look_ahead_mask(inputs):
    sequence_length = tf.shape(inputs)[1]
    look_ahead_mask = 1 - \
        tf.linalg.band_part(tf.ones((sequence_length, sequence_length)), -1, 0)
    padding_mask = creat_padding_mask(inputs)
    return tf.maximum(look_ahead_mask, padding_mask)


def positional_encoding(position, deep):
    i = tf.range(deep, dtype=tf.float32)[tf.newaxis, :]
    position = tf.range(position, dtype=tf.float32)[:, tf.newaxis]
    angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(deep, tf.float32))
    angle_rads = position * angles

    sines = tf.math.sin(angle_rads[:, 0::2])
    cosines = tf.math.cos(angle_rads[:, 1::2])
    pos_encoding = tf.concat([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[tf.newaxis, ...]
    return tf.cast(pos_encoding, tf.float32)


def split_heads(inputs, batch_size, num, deep):
    depth = deep // num
    inputs = tf.reshape(inputs, (batch_size, -1, num, depth))
    return tf.transpose(inputs, perm=[0, 2, 1, 3])


def positional_encoding_layer(position, deep):
    inputs = tf.keras.Input(shape=(None, deep))
    pos_encoding = positional_encoding(position, deep)
    outputs = inputs + pos_encoding[:, :tf.shape(inputs)[1], :]
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def encoder(vocab_size, deep, dropout):
    inputs = tf.keras.Input(shape=(None,))
    embedding = tf.keras.layers.Embedding(vocab_size, deep)(inputs)
    embedding *= tf.math.sqrt(tf.cast(deep, tf.float32))
    embedding = positional_encoding_layer(vocab_size, deep)(embedding)
    outputs = tf.keras.layers.Dropout(rate=dropout)(embedding)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def self_attention(query, key, value, mask):
    matmul = tf.matmul(query, key, transpose_b=True)
    deep = tf.cast(tf.shape(key)[-1], tf.float32)
    scaled_attention = matmul / tf.math.sqrt(deep)

    if mask is not None:
        scaled_attention += (mask * -1e9)
    attention_weight = tf.nn.softmax(scaled_attention, axis=-1)
    output = tf.matmul(attention_weight, value)
    return output


def attention(deep, num):
    query = tf.keras.Input(shape=(None, deep))
    key = tf.keras.Input(shape=(None, deep))
    value = tf.keras.Input(shape=(None, deep))
    mask = tf.keras.Input(shape=(1, None, None))
    batch_size = tf.shape(query)[0]

    query_fc = tf.keras.layers.Dense(units=deep)(query)
    key_fc = tf.keras.layers.Dense(units=deep)(key)
    value_fc = tf.keras.layers.Dense(units=deep)(value)

    query_fc = split_heads(query_fc, batch_size, num, deep)
    key_fc = split_heads(key_fc, batch_size, num, deep)
    value_fc = split_heads(value_fc, batch_size, num, deep)

    attention_state = self_attention(query_fc, key_fc, value_fc, mask)
    attention_state = tf.transpose(attention_state, perm=[0, 2, 1, 3])
    concat_attention = tf.reshape(attention_state, (batch_size, -1, deep))
    output = tf.keras.layers.Dense(units=deep)(concat_attention)

    return tf.keras.Model(inputs=[query, key, value, mask], outputs=output)


def block(units, deep, num, dropout):
    inputs = tf.keras.Input(shape=(None, deep))
    mask = tf.keras.Input(shape=(1, None, None))

    attention_state = attention(deep, num)(
        inputs=[inputs, inputs, inputs, mask])
    attention_state = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(attention_state + inputs)
    outputs = tf.keras.layers.Dense(
        units=units, activation="relu")(attention_state)
    outputs = tf.keras.layers.Dense(units=deep)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(outputs + attention_state)

    return tf.keras.Model(inputs=[inputs, mask], outputs=outputs)


def decoder(num_layers, num_heads, units, deep, dropout):
    inputs = tf.keras.Input(shape=(None, deep))
    mask = tf.keras.Input(shape=(1, None, None))

    outputs = inputs
    for i in range(num_layers):
        outputs = block(units, deep, num_heads, dropout)(
            inputs=[outputs, mask])
    return tf.keras.Model(inputs=[inputs, mask], outputs=outputs)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    优化器将 Adam 优化器与自定义的学习速率调度程序配合使用，这里直接参考了官网的实现
    因为是公式的原因，其实大同小异
    """

    def __init__(self, d_model, warmup_steps=2000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def gpt2(vocab_size, num_layers, units, deep, num_heads, dropout):
    inputs = tf.keras.Input(shape=(None,))
    outputs = encoder(vocab_size=vocab_size, deep=deep,
                      dropout=dropout)(inputs)

    mask = tf.keras.layers.Lambda(
        creat_look_ahead_mask, output_shape=(1, None, None))(inputs)

    for i in range(num_layers):
        outputs = block(units, deep, num_heads, dropout)(
            inputs=[outputs, mask])
    output = tf.keras.layers.Dense(units=vocab_size)(outputs)

    return tf.keras.Model(inputs=inputs, outputs=output)
