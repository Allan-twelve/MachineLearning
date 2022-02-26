import numpy as np


class PCA:
    """
    Principal Component Analysis
    主成分分析
    """
    def __init__(self, component, _mean=None, lambda_list=None, lambda_vectors=None):
        self.component = component
        self.mean = _mean
        self.lamb = lambda_list
        self.vectors = lambda_vectors

    @staticmethod
    def zero_mean(_x, axis=0):
        """
        0均值化
        :param _x: 原特征矩阵
        :param axis: 维度
        :return: 0均值化后的矩阵，各特征的均值
        """
        _mean = np.mean(_x, axis=axis)
        _res = _x - _mean
        return _res, _mean

    def fit(self, _x):
        """
        :param _x: 输入数据集
        """
        _X, self.mean = self.zero_mean(_x)
        _covArray = np.cov(_x, rowvar=False)
        self.lamb, self.vectors = np.linalg.eig(_covArray)
        self.vectors = self.vectors.T

    def fit_transform(self, _x):
        """
        :param _x: 输入数据集
        :return: 降维后的数据
        """
        _X, self.mean = self.zero_mean(_x)
        _covArray = np.cov(_x, rowvar=False)
        self.lamb, self.vectors = np.linalg.eig(_covArray)
        self.vectors = self.vectors.T

        _ind = np.argsort(self.lamb)[::-1]
        _P = self.vectors[_ind[:self.component]]
        return np.dot(_P, _X.T).T

    def transform(self, _x):
        """
        :param _x: 输入数据集
        :return: 降维后数据
        """
        if (self.lamb is None) or (self.vectors is None):
            raise Exception("Still have not been fitted")
        else:
            _X = _x - self.mean
            _ind = np.argsort(self.lamb)[::-1]
            _P = self.vectors[_ind[:self.component]]
            return np.dot(_P, _X.T).T
