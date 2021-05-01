import numpy as np


class LinearRegression:
    """
    Linear Regression
    线性回归
    """
    def __init__(self, alpha=0.01, cycle=1000, normalization=False, lamb=0, theta=None):
        """
        :param alpha: 学习率
        :param cycle: 迭代次数
        :param normalization: l2正则化
        :param lamb: 正则化系数
        :param theta: 权重矩阵
        """
        self.alpha = alpha
        self.cycle = cycle
        self.NL = normalization
        self.lamb = lamb
        self.theta = theta

    def fit(self, _x, _y):
        """
        拟合数据
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
            # 就是加了一列1
            x0 = np.ones([_x.shape[0], 1])
            _X = np.concatenate([x0, _x], axis=1)
            return _X

        _x = mix_x(_x)
        self.theta = np.ones([_x.shape[1], 1])
        # 正则化可选
        if self.NL:
            for _i in range(int(self.cycle)):
                _theta_j = (1 - self.lamb * self.alpha / _x.shape[0]) * self.theta
                _theta_j[0, 0] = self.theta[0, 0]
                self.theta = _theta_j - (self.alpha / _x.shape[0]) * np.dot(_x.T, (np.dot(_x, self.theta) - _y))
        else:
            for _i in range(int(self.cycle)):
                self.theta = self.theta - (self.alpha / _x.shape[0]) * np.dot(_x.T, (np.dot(_x, self.theta) - _y))

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
