import numpy as np


class DecisionTree:
    """
    Decision Classification Tree
    决策分类树
    """
    def __init__(self, column=-1, cut=False, alpha=0.9, mode='ID3', decision_tree=None):
        """
        :param column: 类别所在的列
        :param alpha: 阈值参数
        :param mode: 模式选择: 1:ID3, 2:C4.5, 3:CART
        :param decision_tree: 决策树
        """
        self.col = column
        self.cut = cut
        self.alpha = alpha
        self.mode = mode
        self.tree = decision_tree

    def treeCreate(self, _dataframe):
        """
        递归生成分类决策树
        :param _dataframe:数据集
        :return: 字典形式的决策树
        """

        def chooseFeature(_dataframe, _col, mode):
            """
            选择信息增益最大的特征列，并返回列名
            :param mode: 决策树模式
            :param _col: 特征值所在列
            :param _dataframe: 数据集
            :return: 特征名
            """

            def ent(_dataframe, _col):
                """
                计算熵
                :param _col: 特征值所在列
                :param _dataframe: 数据集
                :return: 熵的值
                """
                _num = _dataframe.iloc[:, _col].value_counts()
                _length = _dataframe.shape[0]
                _p = _num / _length
                return np.sum(- _p * np.log2(_p))

            def giniIndex(_dataframe, _col):
                """
                计算基尼指数
                :param _col: 特征值所在列
                :param _dataframe:数据集
                :return: 基尼指数
                """
                _num = _dataframe.iloc[:, _col].value_counts()
                _length = _dataframe.shape[0]
                _p = _num / _length
                return 1 - np.sum(np.square(_p))

            old_ent = ent(_dataframe, _col=_col)
            _features = _dataframe.columns
            gain_list = []
            for feat in _features:
                if feat == _features[_col]:
                    pass
                else:
                    _values_ = list(set(_dataframe[feat]))  # .unique().tolist()
                    new_ent = 0
                    if mode == 'ID3':
                        for _val in _values_:
                            new_df = _dataframe[_dataframe[feat] == _val]
                            _r = new_df.shape[0] / _dataframe.shape[0]
                            new_ent += _r * ent(new_df, _col=_col)
                        gain_list.append(old_ent - new_ent)
                    if mode == 'C4.5':
                        _R = 0
                        for _val in _values_:
                            new_df = _dataframe[_dataframe[feat] == _val]
                            _r = new_df.shape[0] / _dataframe.shape[0]
                            new_ent += _r * ent(new_df, _col=_col)
                            _R += - _r * np.log2(_r)
                        gain_list.append((old_ent - new_ent) / _R)
                    if mode == 'CART':
                        _R = []
                        for _val in _values_:
                            new_df = _dataframe[_dataframe[feat] == _val]
                            g = giniIndex(new_df, _col=_col)
                            _R.append(g)
                        gain_list.append(np.sum(_R) / len(_R))
            if mode == 'CART':
                ind = np.argmin(gain_list)
            else:
                ind = np.argmax(gain_list)
            _feat = _features[ind]
            return _feat

        def dataSplit(_dataframe, _feature, _value):
            """
            以一个特征的不同属性值分割数据集,返回分割后的数据集
            :param _dataframe: 数据集
            :param _feature: 特征值
            :param _value: 属性值
            :return: 新数据集
            """
            new_df = _dataframe[_dataframe[_feature] == _value]
            _result = new_df.drop(_feature, axis=1)
            return _result

        _classes = _dataframe.iloc[:, self.col]
        if self.cut:
            if np.max(_classes.value_counts(normalize=True)) > self.alpha:
                return np.argmax(_classes.value_counts())
            if len(_dataframe.columns) == 1:
                return np.argmax(_classes.value_counts())
            _feature = chooseFeature(_dataframe, _col=self.col, mode=self.mode)
            decision_tree = {_feature: {}}
            _values = list(set(_dataframe[_feature]))  # .unique().tolist()
            for _value in _values:
                _df = dataSplit(_dataframe, _feature, _value)
                decision_tree[_feature][_value] = self.treeCreate(_df)
            return decision_tree
        else:
            if len(_dataframe.columns) == 1:
                return np.argmax(_classes.value_counts())
            _feature = chooseFeature(_dataframe, _col=self.col, mode=self.mode)
            decision_tree = {_feature: {}}
            _values = list(set(_dataframe[_feature]))  # .unique().tolist()
            for _value in _values:
                _df = dataSplit(_dataframe, _feature, _value)
                decision_tree[_feature][_value] = self.treeCreate(_df)
            return decision_tree

    def fit(self, _dataframe):
        """模型拟合函数"""
        self.tree = self.treeCreate(_dataframe)

    def treePredict(self, tree, test_data):
        """
        根据决策树进行预测分类
        :param tree: 决策树
        :param test_data: 测试数据
        :return: 预测分类的结果
        """
        _root = list(tree.keys())[0]
        _next = tree[_root]
        _index = test_data[_root]
        if _index in _next:
            _result = _next[_index]
        else:
            _result = _next[list(_next.keys())[0]]
        if type(_result) == dict:
            return self.treePredict(_result, test_data)
        else:
            return _result

    def predict(self, test_data):
        """预测函数"""
        return self.treePredict(self.tree, test_data)
