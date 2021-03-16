import numpy as np
import tensorflow as tf
from typing import *

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model: Any, warmup_steps: Any = 4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step: Any):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {
            "d_model": self.d_model,
            "warmup_steps": self.warmup_steps
        }


def loss_func_mask(real: tf.Tensor, pred: tf.Tensor, weights: tf.Tensor = None):
    """ 屏蔽填充的SparseCategoricalCrossentropy损失

    真实标签real中有0填充部分，这部分不记入预测损失

    :param weights: 样本权重
    :param real: 真实标签张量
    :param pred: logits张量
    :return: 损失平均值
    """
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
    mask = tf.math.logical_not(tf.math.equal(real, 0))  # 填充位为0，掩蔽

    loss_ = loss_object(real, pred, sample_weight=weights)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

def triangular_causal_mask(B, L) -> tf.Tensor:
    mask_shape = [B, 1, L, L]
    mask = tf.equal(x=tf.linalg.band_part(input=tf.ones(shape=mask_shape, dtype=tf.bool),
                                          num_lower=-1, num_upper=0), y=tf.constant(False))

    return mask


def prob_mask(B, H, L, index, scores) -> tf.Tensor:
    mask = tf.equal(x=tf.linalg.band_part(index=tf.ones(L, scores.shape[-1], dtype=tf.bool),
                                          num_lower=-1, num_upper=0), y=tf.constant(False))
    mask_ex = mask[None, None, :].expand(B, H, L, scores.shape[-1])
    indicator = mask_ex[np.arange(B)[:, None, None], np.arange(H)[None, :, None], index, :]
    return indicator.reshape(scores.shape)


def positional_embedding(position: Any, d_model: Any, d_type: tf.dtypes.DType = tf.float32) -> tf.Tensor:
    """ PE(pos,2i) = sin(pos/10000^(2i/d_model)) | PE(pos,2i+1) = cos(pos/10000^(2i/d_model))

    :param position: 字符总数
    :param d_model: 词嵌入大小
    :param d_type: 运算精度
    :return:
    """
    angle_rates = 1 / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
    angle_rads = np.arange(position)[:, np.newaxis] * angle_rates

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=d_type)


def data_embedding(embedding_dim: Any, d_type: tf.dtypes.DType = tf.float32,
                   position: Any = 5000, name: AnyStr = "data_embedding") -> tf.keras.Model:
    """ Data Embedding

    :param embedding_dim: 特征维度
    :param d_type: 运行精度
    :param position: 位置总数
    :param name: 名称
    :return: Data Embedding
    """
    inputs = tf.keras.Input(shape=(None, embedding_dim), dtype=d_type, name="{}_inputs".format(name))
    month_inputs = tf.keras.Input(shape=(None,), dtype=d_type, name="{}_month_inputs".format(name))

    # token_embedding = tf.keras.layers.Conv1D(filters=embedding_dim, kernel_size=3, padding="same")(inputs)

    pos_inputs = inputs * tf.math.sqrt(x=tf.cast(x=embedding_dim, dtype=d_type), name="{}_sqrt".format(name))
    pos_encoding = positional_embedding(position=position, d_model=embedding_dim, d_type=d_type)
    pos_embeddings = pos_inputs + pos_encoding[:, :tf.shape(pos_inputs)[1], :]

    month_embedding = tf.keras.layers.Embedding(input_dim=13, output_dim=embedding_dim)(month_inputs)

    embeddings = pos_embeddings + month_embedding  # token_embedding +

    return tf.keras.Model(inputs=[inputs, month_inputs], outputs=embeddings, name=name)


def scaled_dot_product_attention(queries: Any, keys: Any, values: Any,
                                 mask: Any = None, d_type: tf.dtypes.DType = tf.float32):
    """ 点积注意力

    :param queries: 请求的形状 == (..., seq_len_q, depth)
    :param keys: 主键的形状 == (..., seq_len_k, depth)
    :param values: 数值的形状 == (..., seq_len_v, depth_v)
    :param mask: 填充遮罩
    :param d_type: 运算精度
    :return: 上下文向量和attention权重
    """

    matmul_qk = tf.matmul(queries, keys, transpose_b=True)
    dk = tf.cast(x=tf.shape(keys)[-1], dtype=d_type)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(logits=scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, values)

    return output


def prob_query_key(query: tf.Tensor, key: tf.Tensor, sample_key: Any, n_top: Any) -> Tuple:
    """

    :param query: 查询
    :param key: 键
    :param sample_key: 采样个数
    :param n_top: top数量
    :return:
    """
    B, H, L, E = key.shape
    _, _, S, _ = query.shape
    key_expand = tf.broadcast_to(input=tf.expand_dims(input=key, axis=-3), shape=(B, H, S, L, E))
    index_sample = np.random.randint(low=0, high=L, size=(S, sample_key))
    key_sample = key_expand[:, :, tf.expand_dims(input=np.arange(S), axis=1), index_sample, :]
    query_key_sample = tf.matmul(tf.expand_dims(inputs=query, axis=-2),
                                 tf.squeeze(input=tf.transpose(key_sample, perm=[-2, -1])))

    M = np.max(a=query_key_sample, axis=-1)[0] - tf.divide(x=tf.reduce_sum(input_tensor=query_key_sample, axis=-1), y=L)
    M_top = tf.math.top_k(input=M, k=n_top, sorted=False)[1]

    query_reduce = query[np.arange(B)[:, None, None], np.arange(H)[None, :, None], M_top, :]
    query_key = tf.matmul(query_reduce, tf.transpose(key, perm=[-2, -1]))

    return query_key, M_top


def get_initial_context(value: Any, L_query: Any) -> tf.Tensor:
    value_sum = tf.reduce_sum(input_tensor=value, axis=-2)
    context = tf.broadcast_to(input=tf.expand_dims(input=value_sum, axis=-2),
                              shape=(value.shape[0], value.shape[1], L_query, value_sum.shape[-1]))

    return context


def update_context(context_in, value, scores, index) -> tf.Tensor:
    attention = tf.nn.softmax(logits=scores, axis=-1)
    context_in[np.arange(value.shape[0])[:, None, None], np.arange(value.shape[1])[None, :, None], index,
    :] = tf.matmul(attention, value)

    return context_in


def prob_attention(batch_size: Any, num_heads: Any, depth: Any, factor: Any = 5, mask: Any = None,
                   d_type: tf.dtypes.DType = tf.float32, scale: tf.Tensor = None,
                   name: AnyStr = "scaled_dot_produce_attention") -> tf.keras.Model:
    queries = tf.keras.Input(shape=(None, num_heads, depth), dtype=d_type, name="{}_queries".format(name))
    keys = tf.keras.Input(shape=(None, num_heads, depth), dtype=d_type, name="{}_keys".format(name))
    values = tf.keras.Input(shape=(None, num_heads, depth), dtype=d_type, name="{}_values".format(name))

    query = tf.reshape(queries, shape=(batch_size, num_heads, queries.shape[1], -1))
    key = tf.reshape(keys, shape=(batch_size, num_heads, keys.shape[1], -1))
    value = tf.reshape(values, shape=(batch_size, num_heads, values.shape[1], -1))

    U = factor * np.ceil(np.log(keys.shape[1])).astype("int").item()
    u = factor * np.ceil(np.log(queries.shape[1])).astype("int").item()

    scores_top, index = prob_query_key(query, key, u, U)
    scale_vote = scale or 1. / tf.math.sqrt(queries.shape[3])
    if scale_vote is not None:
        scores_top = scores_top * scale_vote

    context = get_initial_context(value, queries.shape[1])
    context = update_context(context, value, scores_top, index)

    return tf.keras.Model(inputs=[queries, keys, values], outputs=context, name=name)


def attention_layer(batch_size: Any, d_model: Any, num_heads: Any, attention: tf.keras.Model = None,
                    d_type: tf.dtypes.DType = tf.float32, name: AnyStr = "attention_layer") -> tf.keras.Model:
    """ Attention Layer

    :param batch_size: batch大小
    :param d_model: 特征维大小
    :param num_heads: 头注意力数量
    :param attention: 使用的self-attention类型
    :param d_type: 运算精度
    :param name: 名称
    :return: Attention Layer
    """
    queries = tf.keras.Input(shape=(None, d_model), dtype=d_type, name="{}_queries".format(name))
    keys = tf.keras.Input(shape=(None, d_model), dtype=d_type, name="{}_keys".format(name))
    values = tf.keras.Input(shape=(None, d_model), dtype=d_type, name="{}_values".format(name))

    assert d_model % num_heads == 0
    depth = d_model // num_heads

    query = tf.keras.layers.Dense(units=d_model)(queries)
    key = tf.keras.layers.Dense(units=d_model)(keys)
    value = tf.keras.layers.Dense(units=d_model)(values)

    query = tf.transpose(tf.reshape(query, (batch_size, -1, num_heads, depth)), perm=[0, 2, 1, 3])
    key = tf.transpose(tf.reshape(key, (batch_size, -1, num_heads, depth)), perm=[0, 2, 1, 3])
    value = tf.transpose(tf.reshape(value, (batch_size, -1, num_heads, depth)), perm=[0, 2, 1, 3])

    if attention is None:
        context = scaled_dot_product_attention(queries=query, keys=key, values=value, d_type=d_type)
    else:
        context = attention(inputs=[query, key, value])
    context = tf.transpose(context, perm=[0, 2, 1, 3])
    concat_context = tf.reshape(context, (batch_size, -1, d_model))

    outputs = tf.keras.layers.Dense(d_model)(concat_context)

    return tf.keras.Model(inputs=[queries, keys, values], outputs=outputs, name=name)


def conv_layer(d_model: Any, d_type: tf.dtypes.DType = tf.float32, name: AnyStr = "conv_layer") -> tf.keras.Model:
    """

    :param d_model:
    :param d_type:
    :param name:
    :return:
    """

    inputs = tf.keras.Input(shape=(12, d_model), dtype=d_type, name="{}_inputs")
    outputs = tf.keras.layers.Conv1D(filters=d_model, kernel_size=3, padding="same", activation="relu")(inputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)


def encoder_layer(batch_size: Any, d_model: Any, num_heads: Any, dropout: Any,
                  d_type: tf.dtypes.DType = tf.float32, name="encoder_layer") -> tf.keras.Model:
    """

    :param batch_size:
    :param d_model:
    :param num_heads:
    :param dropout:
    :param d_type:
    :param name:
    :return:
    """
    inputs = tf.keras.Input(shape=(12, d_model), dtype=d_type, name="{}_inputs".format(name))

    # attention = scaled_dot_product_attention(num_heads=num_heads, depth=d_model // num_heads,
    #                                          d_type=d_type, mask=padding_mask)
    attention_output = attention_layer(batch_size=batch_size, d_model=d_model, num_heads=num_heads,
                                       d_type=d_type)(inputs=[inputs, inputs, inputs])

    attention_output = tf.keras.layers.Dropout(rate=dropout, dtype=d_type,
                                               name="{}_attention_dropout".format(name))(attention_output)
    attention_output = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, dtype=d_type, name="{}_attention_layer_norm".format(name))(inputs + attention_output)

    conv_output = tf.keras.layers.Conv1D(filters=4 * d_model, kernel_size=1,
                                         strides=1, activation="gelu", padding="same")(attention_output)
    conv_output = tf.keras.layers.Dropout(rate=dropout, dtype=d_type,
                                          name="{}_outputs_dropout".format(name))(conv_output)
    conv_output = tf.keras.layers.Conv1D(filters=d_model, kernel_size=1, strides=1, padding="same")(conv_output)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype=d_type,
                                                 name="{}_outputs_layer_norm".format(name))(conv_output)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)


def encoder(num_layers: int, batch_size: int, embedding_dim: int, num_heads: int,
            dropout: float, d_type: tf.dtypes.DType = tf.float32, name: str = "encoder") -> tf.keras.Model:
    """transformer的encoder

    :param num_layers: 编码解码的数量
    :param batch_size: batch大小
    :param embedding_dim: 词嵌入维度
    :param num_heads: 多头注意力的头部层数量
    :param dropout: dropout的权重
    :param d_type: 运算精度
    :param name: 名称
    :return: Transformer的Encoder
    """
    inputs = tf.keras.Input(shape=(12, embedding_dim), name="{}_inputs".format(name), dtype=d_type)

    outputs = tf.keras.layers.Dropout(rate=dropout, dtype=d_type, name="{}_dropout".format(name))(inputs)

    for i in range(num_layers):
        enc_outputs = encoder_layer(batch_size=batch_size, d_model=embedding_dim, num_heads=num_heads, dropout=dropout,
                                    d_type=d_type, name="{}_enc_layer_{}".format(name, i))(outputs)
        outputs = conv_layer(d_model=embedding_dim, d_type=d_type, name="{}_conv_layer_{}".format(name, i))(enc_outputs)

    outputs = encoder_layer(batch_size=batch_size, d_model=embedding_dim, num_heads=num_heads, dropout=dropout,
                            d_type=d_type, name="{}_enc_layer_{}".format(name, num_layers))(outputs)

    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype=d_type,
                                                 name="{}_outputs_layer_norm".format(name))(outputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)


def decoder_layer(batch_size: Any, d_model: int, num_heads: int, dropout: float,
                  d_type: tf.dtypes.DType = tf.float32, name: str = "decoder_layer") -> tf.keras.Model:
    """Transformer的decoder层

    :param batch_size: batch大小
    :param units: 词汇量大小
    :param d_model: 深度，词嵌入维度
    :param num_heads: 注意力头数
    :param dropout: dropout的权重
    :param d_type: 运算精度
    :param name: 名称
    :return: Transformer的Decoder内部层
    """
    inputs = tf.keras.Input(shape=(30, d_model), dtype=d_type, name="{}_inputs".format(name))
    enc_outputs = tf.keras.Input(shape=(12, d_model), dtype=d_type, name="{}_encoder_outputs".format(name))
    # look_ahead_mask = tf.keras.Input(shape=(1, None, None), dtype=d_type, name="{}_look_ahead_mask".format(name))

    # self_attention = scaled_dot_product_attention(num_heads=num_heads, depth=d_model // num_heads,
    #                                               d_type=d_type, mask=look_ahead_mask)
    self_attention_output = attention_layer(
        batch_size=batch_size, d_model=d_model, num_heads=num_heads,
        d_type=d_type, name="{}_attention_layer_1".format(name)
    )(inputs=[inputs, inputs, inputs])

    self_attention_output = tf.keras.layers.Dropout(rate=dropout, dtype=d_type,
                                                    name="{}_attention_dropout1".format(name))(self_attention_output)
    self_attention_output = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, dtype=d_type, name="{}_attention_layer_norm1".format(name))(inputs + self_attention_output)

    # cross_attention = scaled_dot_product_attention(num_heads=num_heads, depth=d_model // num_heads,
    #                                                d_type=d_type, mask=padding_mask)
    cross_attention_output = attention_layer(
        batch_size=batch_size, d_model=d_model, num_heads=num_heads,
        d_type=d_type, name="{}_attention_layer_2".format(name)
    )(inputs=[self_attention_output, enc_outputs, enc_outputs])

    cross_attention_output = tf.keras.layers.Dropout(
        rate=dropout, dtype=d_type, name="{}_attention_dropout2".format(name))(cross_attention_output)
    cross_attention_output = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, dtype=d_type,
        name="{}_attention_layer_norm2".format(name)
    )(self_attention_output + cross_attention_output)

    outputs = tf.keras.layers.Conv1D(filters=4 * d_model, kernel_size=1,
                                     strides=1, activation="gelu")(cross_attention_output)
    outputs = tf.keras.layers.Dropout(rate=dropout, dtype=d_type,
                                      name="{}_outputs_dropout".format(name))(outputs)
    outputs = tf.keras.layers.Conv1D(filters=d_model, kernel_size=1, strides=1)(outputs)

    outputs = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, dtype=d_type, name="{}_outputs_layer_norm".format(name))(cross_attention_output + outputs)

    return tf.keras.Model(inputs=[inputs, enc_outputs], outputs=outputs, name=name)


def decoder(batch_size: Any, num_layers: int, embedding_dim: int, num_heads: int,
            dropout: float, d_type: tf.dtypes.DType = tf.float32, name: str = "decoder") -> tf.keras.Model:
    """transformer的decoder

    :param batch_size: batch大小
    :param num_layers: 编码解码的数量
    :param embedding_dim: 词嵌入维度
    :param num_heads: 多头注意力的头部层数量
    :param dropout: dropout的权重
    :param d_type: 运算精度
    :param name: 名称
    :return: Transformer的Decoder
    """
    inputs = tf.keras.Input(shape=(30, embedding_dim), dtype=d_type, name="{}_inputs".format(name))
    enc_outputs = tf.keras.Input(shape=(12, embedding_dim), dtype=d_type, name="{}_encoder_outputs".format(name))

    outputs = tf.keras.layers.Dropout(rate=dropout, dtype=d_type, name="{}_dropout".format(name))(inputs)

    for i in range(num_layers):
        outputs = decoder_layer(
            batch_size=batch_size, d_model=embedding_dim, num_heads=num_heads, dropout=dropout,
            d_type=d_type, name="decoder_layer_{}".format(i))(inputs=[outputs, enc_outputs])

    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype=d_type,
                                                 name="{}_outputs_layer_norm".format(name))(outputs)

    return tf.keras.Model(inputs=[inputs, enc_outputs], outputs=outputs, name=name)


def informer(embedding_dim: Any, enc_num_layers: Any, dec_num_layers: Any, batch_size: Any, num_heads: Any, dropout: Any,
             d_type: tf.dtypes.DType = tf.float32, name: AnyStr = "informer") -> tf.keras.Model:
    enc_inputs = tf.keras.Input(shape=(24, 72, 4), dtype=d_type, name="{}_enc_inputs".format(name))
    dec_inputs = tf.keras.Input(shape=(24, 72, 4), dtype=d_type, name="{}_dec_inputs".format(name))
    enc_month_inputs = tf.keras.Input(shape=(12,), dtype=d_type, name="{}_enc_month_inputs".format(name))
    dec_month_inputs = tf.keras.Input(shape=(30,), dtype=d_type, name="{}_dec_month_inputs".format(name))

    enc_feature = tf.keras.layers.Conv2D(filters=1, kernel_size=3)(enc_inputs)
    enc_feature = tf.keras.layers.Flatten()(enc_feature)
    enc_feature = tf.expand_dims(input=tf.keras.layers.Dense(units=embedding_dim)(enc_feature), axis=0)
    dec_feature = tf.keras.layers.Conv2D(filters=1, kernel_size=3)(dec_inputs)
    dec_feature = tf.keras.layers.Flatten()(dec_feature)
    dec_feature = tf.expand_dims(input=tf.keras.layers.Dense(units=embedding_dim)(dec_feature), axis=0)

    enc_embeddings = data_embedding(embedding_dim=embedding_dim, d_type=d_type,
                                    name="data_embedding_1")(inputs=[enc_feature, enc_month_inputs])
    enc_outputs = encoder(num_layers=enc_num_layers, batch_size=batch_size, embedding_dim=embedding_dim,
                          num_heads=num_heads, dropout=dropout, d_type=d_type)(enc_embeddings)
    dec_embeddings = data_embedding(embedding_dim=embedding_dim, d_type=d_type,
                                    name="data_embedding_2")(inputs=[dec_feature, dec_month_inputs])
    dec_outputs = decoder(batch_size=batch_size, num_layers=dec_num_layers, embedding_dim=embedding_dim,
                          num_heads=num_heads, dropout=dropout, d_type=d_type)(inputs=[dec_embeddings, enc_outputs])

    outputs = tf.keras.layers.Dense(units=1)(dec_outputs)

    return tf.keras.Model(inputs=[enc_inputs, dec_inputs, enc_month_inputs, dec_month_inputs],
                          outputs=outputs, name=name)
