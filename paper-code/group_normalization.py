def GroupNorm(x, gamma, beta, G, eps=1e-5):
    # x: 举例输入shape为[N, C, H, W]
    # gamma, beta: scale和offset，shape为[1, C, 1, 1]
    # G: GN数量
    N, C, H, W = x.shape
    x = tf.reshape(x, [N, G, C // G, H, W])

    mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
    x = (x - mean) / tf.sqrt(var + eps)
    x = tf.reshape(x, [N, C, H, W])
    return x * gamma + beta