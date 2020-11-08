import tensorflow as tf


class PositionalEncoding(tf.keras.layers.Layer):
    """
    位置编码的简单实现，实现了位置编码的两个公式(针对奇偶位置进行的编码)
    位置编码原理自行翻阅资料，这边不做注释
    """

    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) /
                            tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[
                tf.newaxis, :], d_model=d_model
        )

        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


def create_padding_mask(input):
    """
    对input中的padding单位进行mask
    :param input:
    :return:
    """
    mask = tf.cast(tf.math.equal(input, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(input):
    seq_len = tf.shape(input)[1]
    look_ahead_mask = 1 - \
        tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(input)
    return tf.maximum(look_ahead_mask, padding_mask)


def transformer_encoder_layer(units, d_model, num_heads, dropout, name="transformer_encoder_layer"):
    """
    # Transformer的encoder层，使用函数式API
    :param units:单元大小
    :param d_model:深度
    :param num_heads:多头注意力的头部层数量
    :param dropout:dropout的权重
    :param name:
    :return:
    """
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    attention = MultiHeadAttention(d_model, num_heads, name="attention")({
        'query': inputs,
        'key': inputs,
        'value': inputs,
        'mask': padding_mask
    })

    attention = tf.keras.layers.Dropout(rate=dropout)(attention)
    attention = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(inputs + attention)

    outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(attention + outputs)

    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)


# Transformer的decoder层，使用函数式API
def transformer_decoder_layer(units, d_model, num_heads, dropout, name="transformer_decoder_layer"):
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")
    look_ahead_mask = tf.keras.Input(
        shape=(1, None, None), name="look_ahead_mask")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    attention1 = MultiHeadAttention(d_model, num_heads, name="attention_1")(inputs={
        'query': inputs,
        'key': inputs,
        'value': inputs,
        'mask': look_ahead_mask
    })
    attention1 = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(attention1 + inputs)

    attention2 = MultiHeadAttention(d_model, num_heads, name="attention_2")(inputs={
        'query': attention1,
        'key': enc_outputs,
        'value': enc_outputs,
        'mask': padding_mask
    })
    attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
    attention2 = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(attention2 + attention1)

    outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention2)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(outputs + attention2)

    return tf.keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name
    )


def encoder(vocab_size, num_layers, units, d_model,
            num_heads, dropout, name="encoder"):
    """
    transformer的encoder，使用函数式API进行编写，实现了
    模型层内部的一系列操作，num_layers决定了使用多少个
    encoder_layer层，更具Transformer架构里面的描述，可以根据
    效果进行调整，在encoder中还进行了位置编码，具体原理自行翻阅
    资料，就是实现公式的问题，这里就不多做注释了
    :param vocab_size:token大小
    :param num_layers:编码解码的数量
    :param units:单元大小
    :param d_model:深度
    :param num_heads:多头注意力的头部层数量
    :param dropout:dropout的权重
    :param name:
    :return: Model(inputs=[inputs, padding_mask], outputs=outputs)
    """
    inputs = tf.keras.Input(shape=(None,), name="inputs")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")
    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    # 这里layer使用的name是为了调试的时候答应信息方便查看，也可以不写
    for i in range(num_layers):
        outputs = transformer_encoder_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name="transformer_encoder_layer_{}".format(i),
        )([outputs, padding_mask])

    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)


def decoder(vocab_size, num_layers, units, d_model, num_heads, dropout, name="decoder"):
    """
    transformer的decoder，使用函数式API进行编写，实现了
    模型层内部的一系列操作，相关的一些变量的时候基本和上面
    的encoder差不多，这里不多说
    :param vocab_size:token大小
    :param num_layers:编码解码的层数量
    :param units:单元大小
    :param d_model:深度
    :param num_heads:多头注意力的头部层数量
    :param dropout:dropout的权重
    :param name:
    :return:
    """
    inputs = tf.keras.Input(shape=(None,), name="inputs")
    enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")
    look_ahead_mask = tf.keras.Input(
        shape=(1, None, None), name="look_ahead_mask")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):
        outputs = transformer_decoder_layer(
            units=units, d_model=d_model, num_heads=num_heads,
            dropout=dropout, name="transformer_decoder_layer_{}".format(i),
        )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

    return tf.keras.Model(inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
                          outputs=outputs, name=name)


def transformer(vocab_size, num_layers, units, d_model,
                num_heads, dropout, name="transformer"):
    """
    transformer的粗粒度的结构实现，在忽略细节的情况下，看作是
    encoder和decoder的实现，这里需要注意的是，因为是使用self_attention，
    所以在输入的时候，这里需要进行mask，防止暴露句子中带预测的信息，影响
    模型的效果
    :param vocab_size:token大小
    :param num_layers:编码解码层的数量
    :param units:单元大小
    :param d_model:深度
    :param num_heads:多头注意力的头部层数量
    :param dropout:dropout的权重
    :param name:
    :return:
    """
    inputs = tf.keras.Input(shape=(None,), name="inputs")
    dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

    # 使用了Lambda将方法包装成层，为的是满足函数式API的需要
    enc_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None),
        name="enc_padding_mask"
    )(inputs)

    look_ahead_mask = tf.keras.layers.Lambda(
        create_look_ahead_mask, output_shape=(1, None, None),
        name="look_ahead_mask"
    )(dec_inputs)

    dec_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None),
        name="dec_padding_mask"
    )(inputs)

    enc_outputs = encoder(
        vocab_size=vocab_size, num_layers=num_layers, units=units,
        d_model=d_model, num_heads=num_heads, dropout=dropout
    )(inputs=[inputs, enc_padding_mask])

    dec_outputs = decoder(
        vocab_size=vocab_size, num_layers=num_layers, units=units,
        d_model=d_model, num_heads=num_heads, dropout=dropout
    )(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

    outputs = tf.keras.layers.Dense(
        units=vocab_size, name="outputs")(dec_outputs)
    return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)


def gumbel_softmax(inputs, alpha):
    """
    按照论文中的公式，实现GumbelSoftmax，具体见论文公式
    Args:
        inputs: 输入
        alpha: 温度
    Returns:混合Gumbel噪音后，做softmax以及argmax之后的输出
    """
    uniform = tf.random.uniform(shape=tf.shape(inputs), maxval=1, minval=0)
    # 以给定输入的形状采样Gumbel噪声
    gumbel_noise = -tf.math.log(-tf.math.log(uniform))
    # 将Gumbel噪声添加到输入中，输入第三维就是分数
    gumbel_outputs = inputs + gumbel_noise
    gumbel_outputs = tf.cast(gumbel_outputs, dtype=tf.float32)
    # 在给定温度下，进行softmax并返回
    gumbel_outputs = tf.nn.softmax(alpha * gumbel_outputs)
    gumbel_outputs = tf.argmax(gumbel_outputs, axis=-1)
    return tf.cast(gumbel_outputs, dtype=tf.float32)


def embedding_mix(gumbel_inputs, inputs):
    probability = tf.random.uniform(shape=tf.shape(
        inputs), maxval=1, minval=0, dtype=tf.float32)
    return tf.where(probability < 0.3, x=gumbel_inputs, y=inputs)


def transformer_scheduled_sample(vocab_size, num_layers, units, d_model, num_heads,
                                 dropout, alpha=1.0, name="transformer_scheduled_sample"):
    """
    Transformer应用Scheduled Sample
    Args:
        vocab_size:token大小
        num_layers:编码解码层的数量
        units:单元大小
        d_model:深度
        num_heads:多头注意力的头部层数量
        dropout:dropout的权重
        name:
    Returns:
    """
    inputs = tf.keras.Input(shape=(None,), name="inputs")
    dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

    # 使用了Lambda将方法包装成层，为的是满足函数式API的需要
    enc_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None),
        name="enc_padding_mask"
    )(inputs)

    look_ahead_mask = tf.keras.layers.Lambda(
        create_look_ahead_mask, output_shape=(1, None, None),
        name="look_ahead_mask"
    )(dec_inputs)

    dec_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None),
        name="dec_padding_mask"
    )(inputs)

    enc_outputs = encoder(
        vocab_size=vocab_size, num_layers=num_layers, units=units,
        d_model=d_model, num_heads=num_heads, dropout=dropout
    )(inputs=[inputs, enc_padding_mask])

    transformer_decoder = decoder(
        vocab_size=vocab_size, num_layers=num_layers, units=units,
        d_model=d_model, num_heads=num_heads, dropout=dropout
    )

    dec_first_outputs = transformer_decoder(
        inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

    # dec_outputs的几种方式
    # 1. dec_outputs = tf.argmax(dec_outputs, axis=-1)  # 使用这个方式的话，就是直接返回最大的概率用来作为decoder的inputs
    # 2. tf.layers.Sparsemax(axis=-1)(dec_outputs) # 使用Sparsemax的方法，具体公式参考论文
    # 3. tf.math.top_k() # 混合top-k嵌入，使用得分最高的5个词汇词嵌入的加权平均值。
    # 4. 使用GumbelSoftmax的方法，具体公式参考论文，下面就用GumbelSoftmax方法
    # 这里使用论文的第四种方法：GumbelSoftmax
    gumbel_outputs = gumbel_softmax(dec_first_outputs, alpha=alpha)
    dec_first_outputs = embedding_mix(gumbel_outputs, dec_inputs)

    dec_second_outputs = transformer_decoder(
        inputs=[dec_first_outputs, enc_outputs, look_ahead_mask, dec_padding_mask])
    outputs = tf.keras.layers.Dense(
        units=vocab_size, name="outputs")(dec_second_outputs)
    return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)


def accuracy(real, pred):
    real = tf.reshape(real, shape=(-1, 40 - 1))
    return tf.keras.metrics.sparse_categorical_accuracy(real, pred)
