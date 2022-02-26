import numpy as np


class SoftMax:
    """
    软分类
    """

    def __init__(self, alpha=0.01, cycle=1000, theta=None, _list=None):
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

        _x = mix_x(_x)
        _y = np.array(_y).reshape(-1, 1)
        self.list = list(set(_y[:, 0]))
        self.theta = np.ones([len(set(_y[:, 0])), _x.shape[1]])
        for _time in range(int(self.cycle)):
            for i in range(len(set(_y[:, 0]))):
                _class = self.list[i]
                y_i = np.zeros([_y.shape[0], 1])
                y_i[_y == _class] = 1
                _t = np.dot((np.exp(np.dot(_x, self.theta[i].T).reshape(-1, 1)) /
                             (np.sum(np.exp(np.dot(_x, self.theta.T)), axis=1, keepdims=True) - y_i)).T, _x)
                self.theta[i] = self.theta[i] - (self.alpha / _x.shape[0]) * _t

    def predict(self, x_i, soft=True):
        """预测函数"""

        def mix_x(_x):
            """将特征变量x矩阵合成为特定的矩阵"""
            x0 = np.ones([_x.shape[0], 1])
            _X = np.concatenate([x0, _x], axis=1)
            return _X

        x_i = mix_x(x_i)
        _result = np.exp(np.dot(x_i, self.theta.T)) / np.sum(np.exp(np.dot(x_i, self.theta.T)), axis=1, keepdims=True)
        if soft:
            return np.array([self.list]).reshape(-1, 1)[np.argmax(_result, axis=1)]
        else:
            return _result
