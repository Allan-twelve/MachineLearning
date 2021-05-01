import numpy as np


class LinearRegression:
    """
    Linear Regression
    线性回归
    """
    def __init__(self, normalization=False, lamb=1, theta=None):
        """
        :param normalization: l2正则化
        :param lamb: 正则化系数
        :param theta: 权重矩阵
        """
        self.NL = normalization
        self.lamb = lamb
        self.theta = theta

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

        _x = mix_x(_x)
        _y = np.array(_y)
        if self.NL:
            _theta_j = self.lamb * np.eye(np.shape(_x)[1])
            _theta_j[0, 0] = 0
            self.theta = np.dot(np.linalg.inv(np.dot(_x.T, _x) + self.lamb * _theta_j), np.dot(_x.T, _y))
        else:
            self.theta = np.dot(np.linalg.inv(np.dot(_x.T, _x)), np.dot(_x.T, _y))

    def predict(self, _xi):
        """预测函数"""
        def mix_x(_x):
            """
            特征变量x矩阵合成为特定的矩阵
            :param _x: x_train
            :return: x_train concatenate a vector
            """
            x0 = np.ones([_x.shape[0], 1])
            _X = np.concatenate([x0, _x], axis=1)
            return _X

        _xi = mix_x(_xi)
        return np.dot(_xi, self.theta)
