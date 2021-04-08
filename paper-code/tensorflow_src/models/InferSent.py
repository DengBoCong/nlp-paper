from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def infer_sent(vocab_size: int, num_layers: int, units: int, embedding_dim: int, num_heads: int,
               dropout: float, d_type: tf.dtypes.DType = tf.float32, name: str = "model") -> tf.keras.Model:
    """短文本匹配模型

    :param vocab_size: token大小
    :param num_layers: 编码解码的数量
    :param units: 单元大小
    :param embedding_dim: 词嵌入维度
    :param num_heads: 多头注意力的头部层数量
    :param dropout: dropout的权重
    :param d_type: 运算精度
    :param name: 名称
    :return:
    """
    premise_inputs = tf.keras.Input(shape=(None,), name="{}_premise_inputs".format(name), dtype=d_type)
    hypothesis_inputs = tf.keras.Input(shape=(None,), name="{}_hypothesis_inputs".format(name), dtype=d_type)

    premise_padding_mask = tf.keras.layers.Lambda(create_padding_mask, output_shape=(1, 1, None),
                                                  name="{}_pre_padding_mask".format(name))(premise_inputs)
    hypothesis_padding_mask = tf.keras.layers.Lambda(create_padding_mask, output_shape=(1, 1, None),
                                                     name="{}_hyp_padding_mask".format(name))(hypothesis_inputs)

    premise_embeddings = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                                                   dtype=d_type, name="{}_pre_embeddings".format(name))(premise_inputs)
    hypothesis_embeddings = tf.keras.layers.Embedding(
        input_dim=vocab_size, output_dim=embedding_dim,
        dtype=d_type, name="{}_hyp_embeddings".format(name)
    )(premise_inputs)
    # initializer = tf.random_normal_initializer(0.0, 0.1)

    u, v = extract_net(
        vocab_size=vocab_size, num_layers=num_layers, units=units,
        embedding_dim=embedding_dim, num_heads=num_heads, dropout=dropout, d_type=d_type
    )(inputs=[premise_embeddings, hypothesis_embeddings, premise_padding_mask, hypothesis_padding_mask])

    outputs = feature_net(embedding_dim=embedding_dim, dropout=dropout, d_type=d_type)(inputs=[u, v])
    outputs = tf.nn.softmax(logits=outputs, axis=-1)

    return tf.keras.Model(inputs=[premise_inputs, hypothesis_inputs], outputs=outputs, name=name)


def feature_net(embedding_dim: int, dropout: float, d_type: tf.dtypes.DType = tf.float32, name: str = "model"):
    """ 特征处理层

    :param embedding_dim: 词嵌入维度
    :param dropout: dropout的权重
    :param d_type: 运算精度
    :param name: 名称
    :return:
    """

    premise_inputs = tf.keras.Input(shape=(None, 2 * embedding_dim), name="{}_pre_inputs".format(name), dtype=d_type)
    hypothesis_inputs = tf.keras.Input(shape=(None, 2 * embedding_dim), name="{}_hyp_inputs".format(name), dtype=d_type)

    u = tf.reduce_max(premise_inputs, axis=1)
    v = tf.reduce_max(hypothesis_inputs, axis=1)

    diff = tf.abs(tf.subtract(u, v))
    mul = tf.multiply(u, v)

    features = tf.concat([u, v, diff, mul], axis=-1)

    features_drop = tf.keras.layers.Dropout(rate=dropout, dtype=d_type)(features)
    features_outputs = tf.keras.layers.Dense(units=embedding_dim, activation="relu")(features_drop)

    outputs_drop = tf.keras.layers.Dropout(rate=dropout, dtype=d_type)(features_outputs)
    outputs = tf.keras.layers.Dense(units=2, activation="relu")(outputs_drop)

    return tf.keras.Model(inputs=[premise_inputs, hypothesis_inputs], outputs=outputs, name=name)


def extract_net(vocab_size: int, num_layers: int, units: int, embedding_dim: int, num_heads: int,
                dropout: float, d_type: tf.dtypes.DType = tf.float32, name: str = "extract_net") -> tf.keras.Model:
    """ 特征抽取层

    :param vocab_size: token大小
    :param num_layers: 编码解码的数量
    :param units: 单元大小
    :param embedding_dim: 词嵌入维度
    :param num_heads: 多头注意力的头部层数量
    :param dropout: dropout的权重
    :param d_type: 运算精度
    :param name: 名称
    :return:
    """
    premise_embeddings = tf.keras.Input(shape=(None, embedding_dim), name="{}_pre_inputs".format(name), dtype=d_type)
    hypothesis_embeddings = tf.keras.Input(shape=(None, embedding_dim), name="{}_hyp_inputs".format(name), dtype=d_type)
    premise_padding_mask = tf.keras.Input(shape=(1, 1, None), name="{}_pre_padding_mask".format(name), dtype=d_type)
    hypothesis_padding_mask = tf.keras.Input(shape=(1, 1, None), name="{}_hyp_padding_mask".format(name), dtype=d_type)

    u_premise = encoder(
        vocab_size=vocab_size, num_layers=num_layers, units=units, embedding_dim=embedding_dim,
        num_heads=num_heads, dropout=dropout, d_type=d_type, name="premise_encoder"
    )(inputs=[premise_embeddings, premise_padding_mask])
    v_hypothesis = encoder(
        vocab_size=vocab_size, num_layers=num_layers, units=units, embedding_dim=embedding_dim,
        num_heads=num_heads, dropout=dropout, d_type=d_type, name="hypothesis_encoder"
    )(inputs=[hypothesis_embeddings, hypothesis_padding_mask])

    u_premise_lstm = bi_lstm_block(hidden_size=embedding_dim // 2, embedding_dim=embedding_dim,
                                   d_type=d_type, name="{}_pre_bi_lstm".format(name))(premise_embeddings)
    v_hypothesis_lstm = bi_lstm_block(hidden_size=embedding_dim // 2, embedding_dim=embedding_dim,
                                      d_type=d_type, name="{}_hyp_bi_lstm".format(name))(hypothesis_embeddings)

    u_outputs = tf.concat([u_premise, u_premise_lstm], axis=-1)
    v_outputs = tf.concat([v_hypothesis, v_hypothesis_lstm], axis=-1)

    return tf.keras.Model(inputs=[premise_embeddings, hypothesis_embeddings, premise_padding_mask,
                                  hypothesis_padding_mask], outputs=[u_outputs, v_outputs], name=name)


def bi_lstm_block(hidden_size: int, embedding_dim: int,
                  d_type: tf.dtypes.DType = tf.float32, name: str = "bi_lstm") -> tf.keras.Model:
    """ 双向LSTM

    :param hidden_size: 单元大小
    :param embedding_dim: 词嵌入维度
    :param d_type: 运算精度
    :param name: 名称
    :return:
    """
    inputs = tf.keras.Input(shape=(None, embedding_dim), name="{}_inputs".format(name), dtype=d_type)

    lstm = tf.keras.layers.LSTM(
        units=hidden_size, return_sequences=True, return_state=True,
        recurrent_initializer="glorot_uniform", dtype=d_type, name="{}_lstm_cell".format(name)
    )
    bi_lstm = tf.keras.layers.Bidirectional(layer=lstm, dtype=d_type, name="{}_bi_lstm".format(name))

    outputs = bi_lstm(inputs)[0]

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)


def encoder(vocab_size: int, num_layers: int, units: int, embedding_dim: int, num_heads: int,
            dropout: float, d_type: tf.dtypes.DType = tf.float32, name: str = "encoder") -> tf.keras.Model:
    """ 文本句子编码

    :param vocab_size: token大小
    :param num_layers: 编码解码的数量
    :param units: 单元大小
    :param embedding_dim: 词嵌入维度
    :param num_heads: 多头注意力的头部层数量
    :param dropout: dropout的权重
    :param d_type: 运算精度
    :param name: 名称
    :return:
    """
    inputs = tf.keras.Input(shape=(None, embedding_dim), name="{}_inputs".format(name), dtype=d_type)
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="{}_padding_mask".format(name), dtype=d_type)

    embeddings = inputs * tf.math.sqrt(x=tf.cast(x=embedding_dim, dtype=d_type), name="{}_sqrt".format(name))
    pos_encoding = positional_encoding(position=vocab_size, d_model=embedding_dim, d_type=d_type)
    embeddings = embeddings + pos_encoding[:, :tf.shape(embeddings)[1], :]

    outputs = tf.keras.layers.Dropout(rate=dropout, dtype=d_type, name="{}_dropout".format(name))(embeddings)

    for i in range(num_layers):
        outputs = encoder_layer(units=units, d_model=embedding_dim, num_heads=num_heads, dropout=dropout,
                                d_type=d_type, name="{}_layer_{}".format(name, i))([outputs, padding_mask])

    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)


def encoder_layer(units: int, d_model: int, num_heads: int, dropout: float,
                  d_type: tf.dtypes.DType = tf.float32, name: str = "encoder_layer") -> tf.keras.Model:
    """
    :param units: 词汇量大小
    :param d_model: 深度，词嵌入维度
    :param num_heads: 注意力头数
    :param dropout: dropout的权重
    :param d_type: 运算精度
    :param name: 名称
    :return:
    """
    inputs = tf.keras.Input(shape=(None, d_model), dtype=d_type, name="{}_inputs".format(name))
    padding_mask = tf.keras.Input(shape=(1, 1, None), dtype=d_type, name="{}_padding_mask".format(name))

    attention, _ = MultiHeadAttention(d_model, num_heads)(q=inputs, k=inputs, v=inputs, mask=padding_mask)
    attention = tf.keras.layers.Dropout(rate=dropout, dtype=d_type, name="{}_attention_dropout".format(name))(attention)
    attention = tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype=d_type,
                                                   name="{}_attention_layer_norm".format(name))(inputs + attention)

    outputs = tf.keras.layers.Dense(units=units, activation="relu",
                                    dtype=d_type, name="{}_dense_act".format(name))(attention)
    outputs = tf.keras.layers.Dense(units=d_model, dtype=d_type, name="{}_dense".format(name))(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout, dtype=d_type, name="{}_outputs_dropout".format(name))(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype=d_type,
                                                 name="{}_outputs_layer_norm".format(name))(attention + outputs)

    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)

def scaled_dot_product_attention(q, k, v, mask=None):
    """计算注意力权重。
    q, k, v 必须具有匹配的前置维度。
    k, v 必须有匹配的倒数第二个维度，例如：seq_len_k = seq_len_v。
    虽然 mask 根据其类型（填充或前瞻）有不同的形状，
    但是 mask 必须能进行广播转换以便求和。

    参数:
      q: 请求的形状 == (..., seq_len_q, depth)
      k: 主键的形状 == (..., seq_len_k, depth)
      v: 数值的形状 == (..., seq_len_v, depth_v)
      mask: Float 张量，其形状能转换成
            (..., seq_len_q, seq_len_k)。默认为None。

    返回值:
      输出，注意力权重
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # 缩放 matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # 将 mask 加入到缩放的张量上。
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax 在最后一个轴（seq_len_k）上归一化，因此分数相加等于1。
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """分拆最后一个维度到 (num_heads, depth).
        转置结果使得形状为 (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_q, d_model)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

def _get_angles(pos: tf.Tensor, i: tf.Tensor, d_model: tf.Tensor) -> Tuple:
    """pos/10000^(2i/d_model)

    :param pos: 字符总的数量按顺序递增
    :param i: 词嵌入大小按顺序递增
    :param d_model: 词嵌入大小
    :return: shape=(pos.shape[0], d_model)
    """
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position: int, d_model: int, d_type: tf.dtypes.DType = tf.float32) -> Tuple:
    """PE(pos,2i) = sin(pos/10000^(2i/d_model)) | PE(pos,2i+1) = cos(pos/10000^(2i/d_model))

    :param position: 字符总数
    :param d_model: 词嵌入大小
    :param d_type: 运算精度
    """
    angle_rads = _get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=d_type)

def create_padding_mask(seq: tf.Tensor) -> Tuple:
    """ 用于创建输入序列的扩充部分的mask

    :param seq: 输入序列
    :return: mask
    """
    seq = tf.cast(x=tf.math.equal(seq, 0), dtype=tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)