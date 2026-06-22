"""nano-mllm · P01 训练引擎：手写前向 + 反向 + 梯度检查
对应路线图 P01「深度学习核心三件套」。
运行: python nano-mllm/engine/autograd_demo.py
依赖: numpy
"""
import numpy as np


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def forward(x, W1, b1, W2, b2):
    z1 = W1 @ x + b1
    h = np.maximum(0, z1)          # ReLU
    z2 = W2 @ h + b2
    return softmax(z2), (x, z1, h)


def backward(y, yhat, cache, W2):
    x, z1, h = cache
    dz2 = yhat - y                 # softmax + CE 的简洁梯度
    dW2 = np.outer(dz2, h)
    dh = W2.T @ dz2
    dz1 = dh * (z1 > 0)            # ReLU' = (z1>0)
    dW1 = np.outer(dz1, x)
    return dW1, dW2


def loss_of(W1, b1, W2, b2, x, y):
    p, _ = forward(x, W1, b1, W2, b2)
    return -np.log(p[y.argmax()] + 1e-12)


def grad_check(W1, b1, W2, b2, x, y, eps=1e-5):
    """数值梯度 vs 解析梯度，应一致到 ~1e-7（这是 torch.autograd.gradcheck 的原理）。"""
    yhat, cache = forward(x, W1, b1, W2, b2)
    dW1, _ = backward(y, yhat, cache, W2)
    num = np.zeros_like(W1)
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            up, dn = W1.copy(), W1.copy()
            up[i, j] += eps
            dn[i, j] -= eps
            num[i, j] = (loss_of(up, b1, W2, b2, x, y) - loss_of(dn, b1, W2, b2, x, y)) / (2 * eps)
    return np.abs(num - dW1).max() / (np.abs(num).max() + 1e-12)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x, y = rng.standard_normal(4), np.eye(3)[1]
    W1, b1 = rng.standard_normal((8, 4)), np.zeros(8)
    W2, b2 = rng.standard_normal((3, 8)), np.zeros(3)

    yhat, cache = forward(x, W1, b1, W2, b2)
    dW1, dW2 = backward(y, yhat, cache, W2)
    W1 -= 0.1 * dW1   # 一步梯度下降：W ← W − lr·∂L/∂W
    W2 -= 0.1 * dW2

    rel = grad_check(W1, b1, W2, b2, x, y)
    print(f"梯度检查 max 相对误差 = {rel:.2e}  ->  {'✅ 反向写对了' if rel < 1e-6 else '❌ 检查反向'}")
