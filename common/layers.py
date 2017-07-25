import numpy as np
import common.functions as fn


class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)  # x의 모든 원소에 대하여, 0 이하의 원소: True, 0 보다 큰 원소: False로 이루어진 bool 배열
        out = x.copy()
        out[self.mask] = 0  # 0 이하의 원소는 0으로 대체

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out =  fn.sigmoid(x)
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * self.out * (1.0 - self.out)
        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape = None
        # 가중치와 편향 매개변수의 미분
        self.dW = None
        self.db = None

    def forward(self, x):
        # 텐서 대응
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        return np.dot(self.x, self.W) + self.b

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)  # 입력 데이터 모양 변경(텐서 대응)
        return dx


class SoftmaxWithLoss:
    """학습에 사용되는 함수로 오류(정답과의 오차)의 미분 값으로 가중치 조절을 위해 사용한다."""
    def __init__(self):
        self.loss = None  # 손실 함수의 결과값
        self.y = None  # softmax 함수의 결과값(신경망의 예측값)
        self.t = None  # 실제 데이터의 정답(정답 레이블) (one-hot encoding)

    def forward(self, x, t):
        self.t = t
        self.y = fn.softmax(x)
        self.loss = fn.cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # 정답 레이블이 one-hot encoding 인 경우
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx /= batch_size

        return dx


class Dropout:
    """"""
    pass


class Convolution:
    """"""
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad


    def forward(self, input_data):
        pass