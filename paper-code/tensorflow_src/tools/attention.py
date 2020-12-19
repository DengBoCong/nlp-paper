import tensorflow as tf

# bahdanau attention
def bahdanau_attention(hidden_dim: int, units: int):
    """
    :param units: 全连接层单元数
    """
    query = tf.keras.Input(shape=(hidden_dim))
    values = tf.keras.Input(shape=(None, hidden_dim))
    V = tf.keras.layers.Dense(1)
    W1 = tf.keras.layers.Dense(units)
    W2 = tf.keras.layers.Dense(units)
    # query其实就是decoder的前一个状态，decoder的第一个状态就是上
    # 面提到的encoder反向RNN的最后一层，它作为decoderRNN中的初始隐藏层状态
    # values其实就是encoder每个时间步的隐藏层状态，所以下面需要将query扩展一个时间步维度进行之后的操作
    hidden_with_time_axis = tf.expand_dims(query, 1)
    score = V(tf.nn.tanh(W1(values) + W2(hidden_with_time_axis)))
    attention_weights = tf.nn.softmax(score, axis=1)
    context_vector = attention_weights * values
    context_vector = tf.reduce_mean(context_vector, axis=1)

    return tf.keras.Model(inputs=[query, values], outputs=[context_vector, attention_weights])


# luong attention
def luong_attention_concat(hidden_dim: int, units: int) -> tf.keras.Model:
    """
    :param units: 全连接层单元数
    """
    query = tf.keras.Input(shape=(hidden_dim))
    values = tf.keras.Input(shape=(None, hidden_dim))
    W1 = tf.keras.layers.Dense(units)
    V = tf.keras.layers.Dense(1)
    # query其实就是decoder的前一个状态，decoder的第一个状态就是上
    # 面提到的encoder反向RNN的最后一层，它作为decoderRNN中的初始隐藏层状态
    # values其实就是encoder每个时间步的隐藏层状态，所以下面需要将query扩展一个时间步维度进行之后的操作
    hidden_with_time_axis = tf.expand_dims(query, 1)
    scores = V(tf.nn.tanh(W1(hidden_with_time_axis + values)))
    attention_weights = tf.nn.softmax(scores, axis=1)
    context_vector = tf.matmul(attention_weights, values)
    context_vector = tf.reduce_mean(context_vector, axis=1)

    return tf.keras.Model(inputs=[query, values], outputs=[attention_weights, context_vector])


def luong_attention_dot(query: tf.Tensor, value: tf.Tensor) -> tf.Tensor:
    """
    :param query: decoder的前一个状态
    :param value: encoder的output
    """
    hidden_with_time_axis = tf.expand_dims(query, 1)
    scores = tf.matmul(hidden_with_time_axis, value, transpose_b=True)
    attention_weights = tf.nn.softmax(scores, axis=1)
    context_vector = tf.matmul(attention_weights, value)
    context_vector = tf.reduce_mean(context_vector, axis=1)

# self-attention
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
    matmul_qk = tf.matmul(
        q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # 缩放 matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # 将 mask 加入到缩放的张量上。
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax 在最后一个轴（seq_len_k）上归一化，因此分数相加等于1。
    attention_weights = tf.nn.softmax(
        scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


# multi-head attention
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

        # (batch_size, num_heads, seq_len_q, depth)
        q = self.split_heads(q, batch_size)
        # (batch_size, num_heads, seq_len_k, depth)
        k = self.split_heads(k, batch_size)
        # (batch_size, num_heads, seq_len_v, depth)
        v = self.split_heads(v, batch_size)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)

        return output, attention_weights


# Location Sensitive Attention
class Attention(tf.keras.layers.Layer):
    def __init__(self, attention_dim, attention_filters, attention_kernel):
        super(Attention, self).__init__()
        self.attention_dim = attention_dim
        self.attention_location_n_filters = attention_filters
        self.attention_location_kernel_size = attention_kernel
        self.query_layer = tf.keras.layers.Dense(
            self.attention_dim, use_bias=False, activation="tanh")
        self.memory_layer = tf.keras.layers.Dense(
            self.attention_dim, use_bias=False, activation="tanh")
        self.V = tf.keras.layers.Dense(1, use_bias=False)
        self.location_layer = LocationLayer(self.attention_location_n_filters, self.attention_location_kernel_size,
                                            self.attention_dim)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, memory, attention_weights_cat):
        processed_query = self.query_layer(tf.expand_dims(query, axis=1))
        processed_memory = self.memory_layer(memory)

        attention_weights_cat = tf.transpose(attention_weights_cat, (0, 2, 1))
        processed_attention_weights = self.location_layer(
            attention_weights_cat)
        energies = tf.squeeze(self.V(tf.nn.tanh(
            processed_query + processed_attention_weights + processed_memory)), -1)
        return energies

    def __call__(self, attention_hidden_state, memory, attention_weights_cat):
        alignment = self.get_alignment_energies(
            attention_hidden_state, memory, attention_weights_cat)
        attention_weights = tf.nn.softmax(alignment, axis=1)
        attention_context = tf.expand_dims(attention_weights, 1)

        attention_context = tf.matmul(attention_context, memory)
        attention_context = tf.squeeze(attention_context, axis=1)
        return attention_context, attention_weights


class LocationLayer(tf.keras.layers.Layer):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim1):
        super(LocationLayer, self).__init__()
        self.location_convolution = tf.keras.layers.Conv1D(
            filters=attention_n_filters,
            kernel_size=attention_kernel_size,
            padding="same",
            use_bias=False,
            name="location_conv")
        self.location_layer = tf.keras.layers.Dense(
            units=attention_dim1, use_bias=False, activation="tanh", name="location_layer")

    def call(self, attention_weights_cat):
        processed_attention = self.location_convolution(attention_weights_cat)
        processed_attention = self.location_layer(processed_attention)
        return processed_attention
