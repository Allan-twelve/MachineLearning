import numpy as np


class SVM:
    """
    Support Vector Machine
    支持向量机
    """
    def __init__(self, C=1, max_iter=100, alpha=None, weight=None, b=None,
                 kernel='Linear', sigmoid=0.5, p=2, p_c=1, t=1e-3, support=None):
        """
        :param C: 松弛变量
        :param max_iter: 最大迭代数
        :param alpha: 拉格朗日乘子
        :param weight: 权重
        :param b: 截距
        :param kernel: 核
        :param sigmoid: 高斯核的方差
        :param p: 多项式指数
        :param p_c: 多项式常数
        :param t: 容错率
        :param support: 支持向量索引
        """
        self.C = C
        self.cycle = max_iter
        self.kernel = kernel
        self.sigma = sigmoid
        self.p = p
        self.p_c = p_c
        self.t = t
        self.support = support
        self.alpha = alpha
        self.weight = weight
        self.b = b

    def fit(self, _x, _y):
        """
        :param _x: x_train
        :param _y: y_train
        """
        def K(x_i, x_j, kernel=self.kernel):
            if kernel == 'Linear':
                return np.dot(x_i, x_j.T)
            elif kernel == 'Gauss':
                if x_i.ndim == 1 and x_j.ndim == 1:
                    return - np.sum(np.square(x_i - x_j)) / 2 * np.square(self.sigma)
                else:
                    return - np.sum(np.square(x_i - x_j), axis=1, keepdims=True) / 2 * np.square(self.sigma)
            elif kernel == 'Polynomial':
                if x_j.ndim == 1:
                    return np.power((np.dot(x_i, x_j.T) + self.p_c), self.p).reshape(-1, 1)
                else:
                    return np.power((np.dot(x_i, x_j.T) + self.p_c), self.p)

        # 保证y是列向量
        _y = np.array(_y).reshape(-1, 1)
        # 初始化alpha和b
        _alpha = np.zeros([_x.shape[0], 1])
        _b = 0
        # 计算所有的E值列表
        _g = np.dot(K(_x, _x).T, (_alpha * _y)) + _b
        _e = _g - _y
        for i in range(self.cycle):
            # 初始化次数
            alpha_change_times = 0
            # 遍历全部样本
            for j in range(_x.shape[0]):
                # 选出不符合KKT的a_1进行优化
                if np.abs(_alpha[j]) < self.t:
                    if _y[j] * (np.dot(K(_x, _x[j]).T, (_alpha * _y))) + _b >= 1:
                        continue
                    else:
                        ind_1 = j
                elif np.abs(_alpha[j] - self.C) < self.t:
                    if _y[j] * (np.dot(K(_x, _x[j]).T, (_alpha * _y))) + _b <= 1:
                        continue
                    else:
                        ind_1 = j
                else:
                    if np.abs(_y[j] * (np.dot(K(_x, _x[j]).T, (_alpha * _y))) + _b - 1) < self.t:
                        continue
                    else:
                        ind_1 = j
                # 选出a_1
                a_1 = _alpha[ind_1]
                x_1 = _x[ind_1]
                y_1 = _y[ind_1]
                # 计算E1值
                g_1 = np.dot(K(_x, x_1).reshape(1, -1), (_alpha * _y)) + _b
                e_1 = g_1 - y_1
                # 选出a_2
                delta_e = np.abs(_e - e_1)
                ind_2 = np.argmax(delta_e)
                a_2 = _alpha[ind_2]
                x_2 = _x[ind_2]
                y_2 = _y[ind_2]
                # 计算E2值
                g_2 = np.dot(K(_x, x_2).T, (_alpha * _y)) + _b
                e_2 = g_2 - y_2
                # 划分上下界
                if y_1 != y_2:
                    _l = max(0, a_2 - a_1)
                    _h = min(self.C, self.C + a_2 - a_1)
                else:
                    _l = max(0, a_2 + a_1 - self.C)
                    _h = min(self.C, a_2 + a_1)
                # 计算新a_2
                _eta = K(x_1, x_1) + K(x_2, x_2) - 2 * K(x_1, x_2)
                new_a_2 = a_2 + y_2 * (e_1 - e_2) / _eta
                # 进行区域切割，防止越界
                if new_a_2 > _h:
                    new_a_2 = _h
                elif new_a_2 < _l:
                    new_a_2 = _l
                else:
                    pass
                # 基本不变时跳过
                # if new_a_2 == a_2:
                #     continue
                # 计算新a_1
                new_a_1 = a_1 + y_1 * y_2 * (a_2 - new_a_2)
                # 更新alpha列表
                _alpha[ind_1] = new_a_1
                _alpha[ind_2] = new_a_2
                # 更新b
                b_1 = - e_1 - y_1 * K(x_1, x_1) * (new_a_1 - a_1) - y_2 * K(x_2, x_1) * (new_a_2 - a_2) + _b
                b_2 = - e_2 - y_1 * K(x_1, x_2) * (new_a_1 - a_1) - y_2 * K(x_2, x_2) * (new_a_2 - a_2) + _b
                if self.C > new_a_1 > 0:
                    _b = b_1
                elif self.C > new_a_2 > 0:
                    _b = b_2
                else:
                    _b = (b_1 + b_2) / 2
                # 更新E列表
                _g = np.dot(K(_x, _x), (_alpha * _y)) + _b
                _e = _g - _y
                # 改变次数+1
                alpha_change_times += 1
            # 没有改变时跳出循环
            if alpha_change_times == 0:
                break
        self.alpha = _alpha
        # 计算权重
        num = np.arange(0, _alpha.shape[0]).reshape(-1, 1)
        self.support = num[_alpha > 0]
        _weight = np.zeros([1, _x.shape[1]])
        for i in range(_alpha.shape[0]):
            if _alpha[i] > 0:
                _weight += _y[i] * _alpha[i] * _x[i]
        self.weight = _weight
        self.b = _b

    def predict(self, x_i):
        """
        :param x_i: x_test
        :return: y_predict
        """
        def K(x_i, x_j, kernel=self.kernel):
            if kernel == 'Linear':
                return np.dot(x_i, x_j.T)
            elif kernel == 'Gauss':
                if x_i.ndim == 1 and x_j.ndim == 1:
                    return - np.sum(np.square(x_i - x_j)) / 2 * np.square(self.sigma)
                else:
                    return - np.sum(np.square(x_i - x_j), axis=1, keepdims=True) / 2 * np.square(self.sigma)
            elif kernel == 'Polynomial':
                if x_j.ndim == 1:
                    return np.power((np.dot(x_i, x_j.T) + self.p_c), self.p).reshape(-1, 1)
                else:
                    return np.power((np.dot(x_i, x_j.T) + self.p_c), self.p)

        def sign(items):
            _i = items.copy()
            _i[items > 0] = 1
            _i[items < 0] = -1
            return _i

        return sign(K(self.weight, x_i) + self.b)
