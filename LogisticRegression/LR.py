import numpy as np


class LogisticRegression:
    """
    Logistic Regression
    逻辑回归
    """
    def __init__(self, alpha=0.1, cycle=1000, normalization=False, lamb=0, theta=None, _x=None, _y=None, _list=None):
        """
        :param alpha: 学习率
        :param cycle: 迭代次数
        :param normalization: 正则化
        :param lamb: lambda
        :param theta: 权重矩阵
        :param _x: x_train
        :param _y: y_train
        :param _list: 所有分类列表
        """
        self.alpha = alpha
        self.cycle = cycle
        self.NL = normalization
        self.lamb = lamb
        self.theta = theta
        self.x = _x
        self.y = _y
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

        self.x = mix_x(_x)
        self.y = np.array(_y)
        self.list = list(set(self.y[:, 0]))
        self.theta = np.ones([self.x.shape[1], 1])
        # 正则化可选
        if self.NL:
            for _i in range(int(self.cycle)):
                _theta_j = (1 - self.lamb * self.alpha / self.x.shape[0]) * self.theta
                _theta_j[0] = self.theta[0]
                self.theta = _theta_j - (self.alpha / self.x.shape[0]) * \
                             np.dot(self.x.T, (1 / (1 + np.exp(-np.dot(self.x, self.theta))) - self.y))
        else:
            for _i in range(int(self.cycle)):
                self.theta = self.theta - (self.alpha / self.x.shape[0]) * \
                             np.dot(self.x.T, (1 / (1 + np.exp(-np.dot(self.x, self.theta))) - self.y))

    def predict(self, _xi, binarize=True):
        """
        预测函数，返回概率
        :param binarize:
        :param _xi: x_test
        :return: y_predict
        """

        def mix_x(_x):
            """
            将x矩阵合成为特定的矩阵
            :param _x: x_train
            :return: x_train concatenate a vector
            """
            x0 = np.ones([_x.shape[0], 1])
            _X = np.concatenate([x0, _x], axis=1)
            return _X

        _xi = mix_x(_xi)
        # sigmoid函数
        _result = (1 / (1 + np.exp(-np.dot(_xi, self.theta))))
        if binarize:
            _result[_result >= 0.5] = 1
            _result[_result < 0.5] = 0
        return _result

    def OneVsAll(self, _x_i):
        """
        one vs all 预测概率(多分类)
        :param _x_i: x_test
        :return: y_predict
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

        def judge(_i, _j):
            """示性函数，将y返回0或1"""
            if _j == _i:
                return 1
            else:
                return 0

        _x_i = mix_x(_x_i)
        _result = []
        x_old = self.x[:, 1:]
        y_old = self.y
        l_old = self.list
        for _i in set(self.y[:, 0]):
            self.x = x_old
            self.y = y_old
            _y_i = []
            for _j in self.y:
                _y_i.append(judge(_i, _j))
            _y_i = np.array([_y_i]).T
            self.fit(self.x, _y_i)
            _result.append(1 / (1 + np.exp(-np.dot(_x_i, self.theta))))
        _result = np.array([_result]) / np.sum(_result, axis=0)
        _result = np.squeeze(_result, axis=[0, 1]).T
        return [l_old[i] for i in np.argmax(_result, axis=1)]
