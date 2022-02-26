import numpy as np


class LWRegression:
    """
    Locally Weighted Regression
    局部加权线性回归
    """
    def __init__(self, lamb=1, theta=None, _x=None, _y=None, sigma=1.0):
        """
        :param lamb: 正则化系数
        :param theta: 权重矩阵
        :param _x: x_train
        :param _y: y_train
        :param sigma: 高斯核参数
        """
        self.lamb = lamb
        self.theta = theta
        self.x = _x
        self.y = _y
        self.sigmoid = sigma

    def fit(self, _x, _y):
        """
        数据拟合
        :param _x: x_train
        :param _y: y_train
        :return: y_predict
        """

        def mix_x(_x):
            """
            特征变量x矩阵合成为特定的矩阵
            :param _x: x_train
            :return: x_train concatenate a vector
            """
            # 加了一列1，相当于常数项
            x0 = np.ones([_x.shape[0], 1])
            _X = np.concatenate([x0, _x], axis=1)
            return _X

        self.x = mix_x(_x)
        self.y = np.array(_y)

    def predict(self, x_i):
        """预测函数(一次预测单个值版本)"""
        def mix_x(_x):
            """
            特征变量x矩阵合成为特定的矩阵
            :param _x: x_train
            :return: x_train concatenate a vector
            """
            x0 = np.ones([_x.shape[0], 1])
            _X = np.concatenate([x0, _x], axis=1)
            return _X

        def create_w(_x, _xi, sigmoid):
            """
            计算w权重
            :param _x: x_train
            :param _xi: x_test
            :param sigmoid: 参数
            :return: 权重矩阵
            """
            return np.exp(- np.square(x_i[:, np.newaxis, :] - _x).sum(axis=2) / (2 * (sigmoid ** 2)))

        x_i = mix_x(x_i)
        _w = create_w(self.x, x_i, self.sigmoid)
        self.theta = np.multiply(np.linalg.inv(
            np.dot(np.multiply(self.x.T, _w[:, np.newaxis, :]), self.x)),
            np.dot(np.multiply(self.x.T, _w[:, np.newaxis, :]), self.y).reshape([_w.shape[0], 1, -1])).sum(axis=2)
        return np.multiply(self.theta, x_i).sum(axis=1, keepdims=True)
