import numpy as np


def identity_function(x):
    """항등 함수로 입력값을 그대로 리턴한다."""
    return x


def step_function(x):
    """0 보다 큰 입력: 1 출력, 0 이하의 입력: 0 출력"""
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    """"""
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(x, 0)


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))


def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size


def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)
