import numpy as np


def batchnorm_forward(x, gamma, beta, eps):
    N, D = x.shape
    # 第一步：计算平均
    mu = 1./N * np.sum(x, axis=0)
    # 第二步：每个训练样本减去平均
    xmu = x - mu
    # 第三步：计算分母
    sq = xmu ** 2
    # 第四步：计算方差
    var = 1./N * np.sum(sq, axis=0)
    # 第五步：加上eps保证数值稳定性，然后计算开方
    sqrtvar = np.sqrt(var + eps)
    # 第六步：倒转sqrtvar
    ivar = 1./sqrtvar
    # 第七步：计算归一化
    xhat = xmu * ivar
    # 第八步：加上两个参数
    gammax = gamma * xhat
    out = gammax + beta
    # cache储存计算反向传递所需要的一些内容
    cache = (xhat, gamma, xmu, ivar, sqrtvar, var, eps)

    return out, cache


def batchnorm_backward(dout, cache):
    # 展开存储在cache中的变量
    xhat, gamma, xmu, ivar, sqrtvar, var, eps = cache
    # 获得输入输出的维度
    N, D = dout.shape
    dbeta = np.sum(dout, axis=0)
    dgammax = dout
    dgamma = np.sum(dgammax * xhat, axis=0)
    dxhat = dgammax * gamma
    divar = np.sum(dxhat * xmu, axis=0)
    dxmu1 = dxhat * ivar
    dsqrtvar = -1./(sqrtvar ** 2) * divar
    dvar = 0.5 * 1. / np.sqrt(var + eps) * dsqrtvar
    dsq = 1. / N * np.ones((N, D)) * dvar
    dxmu2 = 2 * xmu * dsq
    dx1 = (dxmu1 + dxmu2)
    dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)
    dx2 = 1. / N * np.ones((N, D)) * dmu
    dx = dx1 + dx2

    return dx, dgamma, dbeta
