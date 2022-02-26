import numpy as np


class NB:
    """
    Naive Bayes
    朴素贝叶斯
    """

    def __init__(self, x_percent=None, y_percent=None, y_class=None, length=None):
        self.x_p = x_percent
        self.y_p = y_percent
        self.y_class = y_class
        self.len = length

    def fit(self, _x, _y):
        """
        :param _x: x_train
        :param _y: y_train
        """
        _y = np.array(_y).reshape(-1, 1)
        y_percent = {}
        self.y_class = set(_y[:, 0])
        self.len = _y.shape[0]
        for i in self.y_class:
            y_percent[i] = np.sum(_y == i) / self.len
        x_percent = {}
        for i in self.y_class:
            x_percent[i] = {}
            for j in range(_x.shape[1]):
                x_percent[i][j] = {}
                _ind = np.squeeze(_y == i)
                x_i = _x[_ind]
                x_class = set(x_i[:, j])
                for k in x_class:
                    x_percent[i][j][k] = (np.sum(x_i[:, j] == k) + 1) / (len(x_i[:, j]) + len(x_class))
        self.x_p = x_percent
        self.y_p = y_percent

    def predict(self, _x):
        """
        :param _x: x_test
        :return: y_predict
        """
        def predict_i(x_i):
            """
            :param x_i: x_i
            :return: y_predict_i
            """
            x_pre = np.ones([1, len(self.y_class)])
            num = 0
            for i in self.y_class:
                x_pre[0, num] = self.y_p[i]
                for j in self.x_p[i].keys():
                    if x_i[j] in self.x_p[i][j]:
                        x_pre[0, num] *= self.x_p[i][j][x_i[j]]
                    else:
                        x_pre[0, num] *= 1 / (self.y_p[i] * self.len + len(self.x_p[i][j].keys()))
                num += 1
            return list(self.y_class)[np.argmax(x_pre)]

        return np.apply_along_axis(predict_i, axis=1, arr=_x)
