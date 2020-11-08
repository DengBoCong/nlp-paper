import tensorflow as tf


def accumulate(units, embedding_dim, max_utterance, max_sentence):
    """
    SMN的解码器，主要是对匹配对的两个相似度矩阵进行计
    算，并返回最终的最后一层GRU的状态，用于计算分数
    Args:
        units: GRU单元数
        embedding_dim: embedding维度
        max_utterance: 每轮最大语句数
        max_sentence: 句子最大长度
    Returns: GRU的状态
    """
    utterance_inputs = tf.keras.Input(
        shape=(max_utterance, max_sentence, embedding_dim))
    response_inputs = tf.keras.Input(shape=(max_sentence, embedding_dim))
    response_gru = tf.keras.Input(shape=(max_sentence, units))
    a_matrix = tf.keras.initializers.GlorotNormal()(
        shape=(units, units), dtype=tf.float32)

    conv2d_layer = tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), padding='valid',
                                          kernel_initializer='he_normal', activation='relu')
    max_polling2d_layer = tf.keras.layers.MaxPooling2D(
        pool_size=(3, 3), strides=(3, 3), padding='valid')
    dense_layer = tf.keras.layers.Dense(
        50, activation='tanh', kernel_initializer='glorot_normal')

    # 这里需要做一些前提工作，因为我们要针对每个batch中的每个utterance进行运算，所
    # 以我们需要将batch中的utterance序列进行拆分，使得batch中的序列顺序一一匹配
    utterance_embeddings = tf.unstack(
        utterance_inputs, num=max_utterance, axis=1)
    matching_vectors = []
    for utterance_input in utterance_embeddings:
        # 求解第一个相似度矩阵，公式见论文
        matrix1 = tf.matmul(utterance_input, response_inputs, transpose_b=True)
        utterance_gru = tf.keras.layers.GRU(units, return_sequences=True,
                                            kernel_initializer='orthogonal')(utterance_input)
        matrix2 = tf.einsum("aij,jk->aik", utterance_gru, a_matrix)
        # matrix2 = tf.matmul(utterance_gru, a_matrix)
        # 求解第二个相似度矩阵
        matrix2 = tf.matmul(matrix2, response_gru, transpose_b=True)
        matrix = tf.stack([matrix1, matrix2], axis=3)

        conv_outputs = conv2d_layer(matrix)
        pooling_outputs = max_polling2d_layer(conv_outputs)
        flatten_outputs = tf.keras.layers.Flatten()(pooling_outputs)

        matching_vector = dense_layer(flatten_outputs)
        matching_vectors.append(matching_vector)

    vector = tf.stack(matching_vectors, axis=1)
    outputs = tf.keras.layers.GRU(
        units, kernel_initializer='orthogonal')(vector)

    return tf.keras.Model(inputs=[utterance_inputs, response_inputs, response_gru], outputs=outputs)


def smn(units, vocab_size, embedding_dim, max_utterance, max_sentence):
    """
    SMN的模型，在这里将输入进行accumulate之后，得
    到匹配对的向量，然后通过这些向量计算最终的分类概率
    Args:
        units: GRU单元数
        vocab_size: embedding词汇量
        embedding_dim: embedding维度
        max_utterance: 每轮最大语句数
        max_sentence: 句子最大长度
    Returns: 匹配对打分
    """
    utterances = tf.keras.Input(shape=(max_utterance, max_sentence))
    responses = tf.keras.Input(shape=(max_sentence,))

    embeddings = tf.keras.layers.Embedding(
        vocab_size, embedding_dim, name="encoder")
    utterances_embeddings = embeddings(utterances)
    responses_embeddings = embeddings(responses)

    # 这里对response进行GRU的Word级关系建模，这里用正交矩阵初始化内核权重矩阵，用于输入的线性变换。
    responses_gru = tf.keras.layers.GRU(units=units, return_sequences=True,
                                        kernel_initializer='orthogonal')(responses_embeddings)
    dec_outputs = accumulate(units=units, embedding_dim=embedding_dim, max_utterance=max_utterance,
                             max_sentence=max_sentence)(
        inputs=[utterances_embeddings, responses_embeddings, responses_gru])

    outputs = tf.keras.layers.Dense(
        2, kernel_initializer='glorot_normal')(dec_outputs)

    return tf.keras.Model(inputs=[utterances, responses], outputs=outputs)
