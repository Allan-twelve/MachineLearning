import numpy as np


class NeuralNetwork:  # bug
    """
    单隐藏层神经网络
    """
    def __init__(self, _hidden=5, alpha=0.01, cycle=1000, mode='sigmoid', constant=0.01,
                 _w1=None, _w2=None, _b1=None, _b2=None):
        """
        :param _hidden: 隐藏层单元数
        :param alpha: 学习率
        :param cycle: 迭代次数
        :param mode: 模式选择
        :param constant: 随机参数放缩值
        :param _w1: 权重1
        :param _w2: 权重2
        :param _b1: 偏置1
        :param _b2: 偏置2
        """
        self.units = _hidden
        self.alpha = alpha
        self.cycle = cycle
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
        :return: 返回权重:w与偏置值:b
        """

        def set_Y(_y):
            """
            将_y变为值为0或1的矩阵
            :param _y: 输出的类别，如1,2,3
            :return: 0和1组成的矩阵
            """
            _Y = np.zeros([len(set(_y[:, 0])), _y.shape[0]])
            _Y[_y[:, 0].astype('int'), range(_y.shape[0])] = 1
            return _Y

        def random_para(_x, _y):
            """
            对于隐藏层为单层的神经网络，随机生成权重和偏置值
            :param _x: x_train
            :param _y: y_train
            :return: 权重:w和偏置值:b
            """
            _input = _x.shape[0]
            _output = _y.shape[0]
            _w_1 = np.random.rand(self.units, _input) * self.constant
            _b_1 = np.zeros([self.units, 1])
            _w_2 = np.random.rand(_output, self.units) * self.constant
            _b_2 = np.zeros([_output, 1])
            return _w_1, _b_1, _w_2, _b_2

        def forward(_x, _w_1, _b_1, _w_2, _b_2):
            """前向传播"""
            def sigmoid(_z):
                """sigmoid激活函数"""
                return 1 / (1 + np.exp(-_z))

            def tanh(_z):
                """双曲正切激活函数"""
                return (np.exp(_z) - np.exp(-_z)) / (np.exp(_z) + np.exp(-_z))

            def reLU(_z):
                """reLU激活函数"""
                _z[_z <= 0] = 0
                return _z

            def set_Z(_x, _w, _b):
                """
                计算线性传播
                :param _x: x_train
                :param _w: 权重
                :param _b: 偏置
                :return: 计算结果
                """
                return np.dot(_w, _x) + _b

            if self.mode == 'sigmoid':
                g = sigmoid
            elif self.mode == 'tanh':
                g = tanh
            elif self.mode == 'relu':
                g = reLU
            else:
                raise KeyError(self.mode)
            z_1 = set_Z(_x, _w_1, _b_1)
            _hidden = g(z_1)
            z_2 = set_Z(_hidden, _w_2, _b_2)
            _output = g(z_2)
            return _hidden, _output

        def backward(_hidden, _output, _x, _y, _w_2):
            """反向传播,mode选择激活函数"""
            def d_relu(_g):
                """reLU激活函数的导数"""
                _g[_g >= 0] = 1
                _g[_g < 0] = 0
                return _g

            if self.mode == 'sigmoid':
                d = _hidden * (1 - _hidden)
            elif self.mode == 'tanh':
                d = 1 - np.square(_hidden)
            elif self.mode == 'relu':
                d = d_relu(_hidden)
            else:
                raise KeyError(self.mode)
            _m = _y.shape[1]
            _dz_2 = _output - _y
            _dw_2 = (1 / _m) * np.dot(_dz_2, _hidden.T)
            _db_2 = (1 / _m) * np.sum(_dz_2, axis=1, keepdims=True)
            dz_1 = np.dot(_w_2.T, _dz_2) * d
            _dw_1 = (1 / _m) * np.dot(dz_1, _x.T)
            _db_1 = (1 / _m) * np.sum(dz_1, axis=1, keepdims=True)
            return _dw_1, _db_1, _dw_2, _db_2

        def gradient(_w_1, _b_1, _w_2, _b_2, _dw_1, _db_1, _dw_2, _db_2):
            """梯度更新"""
            _w_1 -= self.alpha * _dw_1
            _b_1 -= self.alpha * _db_1
            _w_2 -= self.alpha * _dw_2
            _b_2 -= self.alpha * _db_2
            return _w_1, _b_1, _w_2, _b_2

        _x = np.array(_x).T
        _y = np.array(_y)
        _y = set_Y(_y)
        self.w1, self.b1, self.w2, self.b2 = random_para(_x, _y)
        for i in range(self.cycle):
            hidden, output = forward(_x, self.w1, self.b1, self.w2, self.b2)
            dw_1, db_1, dw_2, db_2 = backward(hidden, output, _x, _y, self.w2)
            self.w1, self.b1, self.w2, self.b2 = gradient(self.w1, self.b1, self.w2, self.b2, dw_1, db_1, dw_2, db_2)

    def predict(self, _xi):
        """预测函数"""
        def sigmoid(_z):
            """sigmoid激活函数"""
            return 1 / (1 + np.exp(-_z))

        def tanh(_z):
            """双曲正切激活函数"""
            return (np.exp(_z) - np.exp(-_z)) / (np.exp(_z) + np.exp(-_z))

        def reLU(_z):
            """reLU激活函数"""
            _z[_z <= 0] = 0
            return _z

        def set_Z(_x, _w, _b):
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
        elif self.mode == 'relu':
            g = reLU
        else:
            print('Mode error. Default mode=sigmoid')
            g = sigmoid

        _xi = np.array(_xi).T
        z_1 = set_Z(_xi, self.w1, self.b1)
        _hidden = g(z_1)
        z_2 = set_Z(_hidden, self.w2, self.b2)
        _output = g(z_2)
        return np.argmax(softmax(_output), axis=0)

    @staticmethod
    def cost(output, _y):
        """
        代价函数
        :param output: 输出结果
        :param _y: 实际结果
        :return: 代价值
        """
        loss = np.sum(- _y * np.log(output))
        _m = _y.shape[1]
        _J = (1 / _m) * loss
        return _J
