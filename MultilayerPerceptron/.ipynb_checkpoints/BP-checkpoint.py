import numpy as np


class MLP:
    """
    Multilayer Perceptron
    多层感知机
    """
    def __init__(self, hidden=6, alpha=0.1, max_iter=1000, mode='sigmoid', constant=0.01,
                 _w1=None, _w2=None, _b1=None, _b2=None):
        """
        :param hidden: 隐藏层单元数
        :param alpha: 学习率
        :param max_iter: 迭代次数
        :param mode: 选择激活函数
        :param constant: 随机参数放缩值
        :param _w1: 权重1
        :param _w2: 权重2
        :param _b1: 偏置1
        :param _b2: 偏置2
        """
        self.units = hidden
        self.alpha = alpha
        self.cycle = max_iter
        self.mode = mode
        self.constant = constant
        self.w1 = _w1
        self.w2 = _w2
        self.b1 = _b1
        self.b2 = _b2

    def fit(self, _x, _y):
        """
        数据拟合
        :param _x: x_train
        :param _y: y_train
        """

        def set_Y(_y):
            """
            将_y变为值为0或1的矩阵(独热编码)
            :param _y: 输出的类别，如1,2,3
            :return: 0和1组成的矩阵
            """
            _Y = np.zeros([len(set(_y[:, 0])), _y.shape[0]])
            _Y[_y[:, 0].astype('int'), range(_y.shape[0])] = 1
            return _Y

        def random_para(_x, _y, constant):
            """
            随机生成权重和偏置值
            :param constant: 随机参数放缩值
            :param _x: x_train
            :param _y: y_train
            :return: 权重:w和偏置值:b
            """
            input_layer = _x.shape[0]
            output_layer = _y.shape[0]
            _w_1 = np.random.rand(self.units, input_layer) * constant
            _b_1 = np.zeros([self.units, 1])
            _w_2 = np.random.rand(output_layer, self.units) * constant
            _b_2 = np.zeros([output_layer, 1])
            return _w_1, _b_1, _w_2, _b_2

        def forward(_x, _w_1, _b_1, _w_2, _b_2, mode):
            """前向传播"""
            def sigmoid(_z):
                """sigmoid激活函数"""
                return 1 / (1 + np.exp(-_z))

            def tanh(_z):
                """双曲正切激活函数"""
                return (np.exp(_z) - np.exp(-_z)) / (np.exp(_z) + np.exp(-_z))

            def ReLU(_z):
                """ReLU激活函数"""
                _z[_z <= 0] = 0
                return _z

            def linear_z(_x, _w, _b):
                """
                计算线性传播
                :param _x: x_train
                :param _w: 权重
                :param _b: 偏置
                :return: 计算结果
                """
                return np.dot(_w, _x) + _b

            if mode == 'sigmoid':
                g = sigmoid
            elif mode == 'tanh':
                g = tanh
            elif mode == 'ReLU':
                g = ReLU
            else:
                raise KeyError(mode)
            z_1 = linear_z(_x, _w_1, _b_1)
            hidden_output = g(z_1)
            z_2 = linear_z(hidden_output, _w_2, _b_2)
            final_output = g(z_2)
            return hidden_output, final_output

        def backward(_hidden, _output, _x, _y, _w_2, mode):
            """反向传播"""
            def d_ReLU(_g):
                """ReLU激活函数的导数"""
                _g[_g >= 0] = 1
                _g[_g < 0] = 0
                return _g

            if mode == 'sigmoid':
                d = _hidden * (1 - _hidden)
            elif mode == 'tanh':
                d = 1 - np.square(_hidden)
            elif mode == 'ReLU':
                d = d_ReLU(_hidden)
            else:
                raise KeyError(mode)
            _m = _y.shape[1]
            _dz_2 = _output - _y
            _dw_2 = (1 / _m) * np.dot(_dz_2, _hidden.T)
            _db_2 = (1 / _m) * np.sum(_dz_2, axis=1, keepdims=True)
            _dz_1 = np.dot(_w_2.T, _dz_2) * d
            _dw_1 = (1 / _m) * np.dot(_dz_1, _x.T)
            _db_1 = (1 / _m) * np.sum(_dz_1, axis=1, keepdims=True)
            return _dw_1, _db_1, _dw_2, _db_2

        def gradient(_w_1, _b_1, _w_2, _b_2, _dw_1, _db_1, _dw_2, _db_2, alpha):
            """梯度更新"""
            _w_1 -= alpha * _dw_1
            _b_1 -= alpha * _db_1
            _w_2 -= alpha * _dw_2
            _b_2 -= alpha * _db_2
            return _w_1, _b_1, _w_2, _b_2

        _x = np.array(_x).T
        if self.units <= _x.shape[0]:
            raise KeyError(f'the hidden layer ({self.units}) is too small, expected > ({_x.shape[0]})')
        _y = np.array(_y).reshape(-1, 1)
        _y = set_Y(_y)
        w1, b1, w2, b2 = random_para(_x, _y, constant=self.constant)
        hidden, output = forward(_x, w1, b1, w2, b2, mode=self.mode)
        old_cost = np.abs(output - _y).sum()
        for i in range(self.cycle):
            dw_1, db_1, dw_2, db_2 = backward(hidden, output, _x, _y, w2, mode=self.mode)
            w1, b1, w2, b2 = gradient(w1, b1, w2, b2, dw_1, db_1, dw_2, db_2, alpha=self.alpha)
            hidden, output = forward(_x, w1, b1, w2, b2, mode=self.mode)
            new_cost = np.abs(output - _y).sum()
            if np.abs(old_cost - new_cost) < 1e-3:
                print('iter:'+str(i))
                break
            old_cost = new_cost
        self.w1, self.w2, self.b1, self.b2 = w1, w2, b1, b2

    def predict(self, _xi):
        """预测函数"""
        def sigmoid(_z):
            """sigmoid激活函数"""
            return 1 / (1 + np.exp(-_z))

        def tanh(_z):
            """双曲正切激活函数"""
            return (np.exp(_z) - np.exp(-_z)) / (np.exp(_z) + np.exp(-_z))

        def ReLU(_z):
            """ReLU激活函数"""
            _z[_z <= 0] = 0
            return _z

        def linear_z(_x, _w, _b):
            """
            计算线性传播
            :param _x: 输入值,列向量
            :param _w: 权重
            :param _b: 偏置
            :return: 计算结果
            """
            return np.dot(_w, _x) + _b

        def softmax(_output):
            """使输出值总和为1"""
            return np.exp(_output) / np.sum(np.exp(_output), axis=0, keepdims=True)

        if self.mode == 'sigmoid':
            g = sigmoid
        elif self.mode == 'tanh':
            g = tanh
        elif self.mode == 'ReLU':
            g = ReLU
        else:
            raise KeyError(self.mode)

        _xi = np.array(_xi).T
        z_1 = linear_z(_xi, self.w1, self.b1)
        hidden_output = g(z_1)
        z_2 = linear_z(hidden_output, self.w2, self.b2)
        final_output = g(z_2)
        return np.argmax(softmax(final_output), axis=0)
