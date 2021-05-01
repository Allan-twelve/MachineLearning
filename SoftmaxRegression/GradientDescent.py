import numpy as np


class Softmax:
    """
    Softmax
    软分类
    """
    def __init__(self, alpha=0.1, cycle=1000, theta=None, _list=None):
        """
        :param alpha: 学习率
        :param cycle: 迭代次数
        :param theta: 权重矩阵
        :param _list: 所有的分类列表
        """
        self.alpha = alpha
        self.cycle = cycle
        self.theta = theta
        self.list = _list

    def fit(self, _x, _y):
        """
        数据拟合
        :param _x: x_train
        :param _y: y_train
        :return: theta
        """

        def mix_x(_x):
            """
            特征变量x矩阵合成为特定的矩阵
            :param _x: x_train
            :return: x_train concatenate a vector
            """
            x0 = np.ones([_x.shape[0], 1])
            _X = np.concatenate([x0, _x], axis=1)
            return _X

        def judge(_y, _i, _j):
            """示性函数，将y返回0或1"""
            if _y[_j] == _i:
                return 1
            else:
                return 0

        _x = mix_x(_x)
        _y = np.array(_y)
        self.list = list(set(_y[:, 0]))
        self.theta = np.ones([len(set(_y[:, 0])), _x.shape[1]])
        for _time in range(int(self.cycle)):
            for i in range(len(set(_y[:, 0]))):
                _i = list(set(_y[:, 0]))[i]
                _J = np.zeros([_x.shape[1], 1])
                for j in range(_x.shape[0]):
                    _J += (np.dot(_x[j].T, np.exp(np.dot(self.theta[i], _x[j].T))) /
                           np.sum(np.exp(np.dot(self.theta, _x[j].T)) - judge(_y, _i, j))).reshape(3, 1)
                self.theta[i] = self.theta[i] - (self.alpha / _x.shape[0]) * _J.T

    def predict(self, x_i):
        """预测函数"""
        def mix_x(_x):
            """将特征变量x矩阵合成为特定的矩阵"""
            x0 = np.ones([_x.shape[0], 1])
            _X = np.concatenate([x0, _x], axis=1)
            return _X

        x_i = mix_x(x_i)
        _result = np.exp(np.dot(self.theta, x_i.T)) / np.sum(np.exp(np.dot(self.theta, x_i.T)))
        return self.list[int(np.argmax(_result.T))]
