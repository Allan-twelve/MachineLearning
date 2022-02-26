import numpy as np
import pandas as pd
from DataSet_tools.data_split import bootstrapping
from RandomForest.decisionTree import DecisionTree


class RandomForest:
    """
    Random Forest
    随机森林
    """
    def __init__(self, tree_nums=10, sample_rate=0.3, forest=None):
        self.tree_nums = tree_nums
        self.rate = sample_rate
        self.trees = forest

    def fit(self, _x, _y):
        """
        :param _x: x_train
        :param _y: y_train
        :return:
        """
        trees = []
        for i in range(self.tree_nums):
            x_train, x_test, y_train, y_test = bootstrapping(_x, _y, int(_x.shape[0] * self.rate))
            train = pd.concat([x_train, y_train], axis=1)
            tr = DecisionTree(mode='CART')
            tr.fit(train)
            trees.append(tr.tree)
        self.trees = trees

    def predict(self, _x):
        """
        :param _x: x_test
        :return: y_predict
        """
        def predict_i(x_i, trees):
            """
            :param x_i: x_i
            :param trees: forest_list
            :return: y_predict_i
            """
            _tr = DecisionTree(mode='CART')
            pred = []
            for _tree in trees:
                _tr.tree = _tree
                pred.append(_tr.predict(x_i))
            return np.argmax(np.bincount(np.array(pred)))

        y_predict = []
        for i in range(_x.shape[0]):
            y_predict.append(predict_i(_x.iloc[i], self.trees))
        return np.array(y_predict)
