import numpy as np


class KNN:
    """
    KNN
    K近邻
    """
    def __init__(self, k=10, _x=None, _y=None, p=2):
        """
        :param k: nearest neighbors
        :param _x: x_train
        :param _y: y_train
        :param p: Kind of distance
        """
        self.k = k
        self.x = _x
        self.y = _y
        self.p = p

    def fit(self, _x, _y):
        """拟合函数"""
        self.x = np.array(_x)
        self.y = np.array(_y)

    def predict(self, _xi):
        """预测函数"""
        # 这里的(a-b)^2转换成了a^2-2ab+b^2以降低对内存的占用
        # 然而还是挺没多大用
        _xi = np.array(_xi)
        x_1 = np.power(self.x, self.p).sum(axis=1)
        x_2 = np.power(_xi, self.p).sum(axis=1)
        _multi = np.dot(self.x, _xi.T)
        # 利用numpy广播进行运算
        _l = np.power(x_1[:, np.newaxis] - 2 * _multi + x_2, (1 / self.p))
        _arg = np.argpartition(_l, self.k, axis=1)[:, :self.k]
        _class = self.y[_arg, 0]
        y_pre = []
        for i in range(_xi.shape[0]):
            # 所谓的众数
            y_pre.append(np.argmax(np.bincount(_class[i].tolist())))
        return np.array([y_pre]).T
