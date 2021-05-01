import numpy as np
import matplotlib.pyplot as plt
from DataSet_tools.data_processing import PreProcessing
pp = PreProcessing()


def MAE(y_test, y_predict):
    """
    Mean Absolute Error
    平均绝对误差
    :param y_test: y_true
    :param y_predict: y_predict
    :return: Score
    """
    # 将高维数据压至一维，提高容错率，后面的代码都这样做了
    while y_test.ndim != 1:
        y_test = np.squeeze(y_test)
    while y_predict.ndim != 1:
        y_predict = np.squeeze(y_predict)
    _result = np.abs(y_test - y_predict).mean()
    return _result


def MSE(y_test, y_predict):
    """
    Mean Square Error
    平均平方误差
    :param y_test: y_true
    :param y_predict: y_predict
    :return: Score
    """
    while y_test.ndim != 1:
        y_test = np.squeeze(y_test)
    while y_predict.ndim != 1:
        y_predict = np.squeeze(y_predict)
    _result = np.square(y_test - y_predict).mean()
    return _result


def RMSE(y_test, y_predict):
    """
    Root Mean Square Error
    均方根误差
    :param y_test: y_true
    :param y_predict: y_predict
    :return: Score
    """
    while y_test.ndim != 1:
        y_test = np.squeeze(y_test)
    while y_predict.ndim != 1:
        y_predict = np.squeeze(y_predict)
    _result = np.sqrt(np.square(y_test - y_predict).mean())
    return _result


def MAPE(y_test, y_predict):
    """
    Mean Absolute Percentage Error
    平均绝对百分比误差
    :param y_test: y_true
    :param y_predict: y_predict
    :return: Score
    """
    while y_test.ndim != 1:
        y_test = np.squeeze(y_test)
    while y_predict.ndim != 1:
        y_predict = np.squeeze(y_predict)
    _result = np.abs((y_test - y_predict) / y_test).mean()
    return _result


def rSquare(y_test, y_predict):
    """
    R Square Score
    R平方评价
    :param y_test: y_true
    :param y_predict: y_predict
    :return: Score
    """
    while y_test.ndim != 1:
        y_test = np.squeeze(y_test)
    while y_predict.ndim != 1:
        y_predict = np.squeeze(y_predict)
    _result = 1 - np.square(y_test - y_predict).sum() / np.square(y_test - np.mean(y_test)).sum()
    return _result


def PR(y_test, y_predict, draw=False):
    """
    Precise - Recall Curve
    PR曲线
    :param y_test: y_true
    :param y_predict: y_predict
    :param draw: Choose whether to draw or not
    :return: Precise and Recall
    """
    while y_test.ndim != 1:
        y_test = np.squeeze(y_test)
    while y_predict.ndim != 1:
        y_predict = np.squeeze(y_predict)
    _P = [1]
    _R = [0]
    for i in np.sort(y_predict)[::-1]:
        y_pre = y_predict.copy()
        y_pre = pp.Binarize(y_pre, i)
        _TP = np.sum(y_test[y_test == 1] == y_pre[y_test == 1])
        _FN = np.sum(y_test[y_test == 1] != y_pre[y_test == 1])
        _FP = np.sum(y_test[y_test == 0] != y_pre[y_test == 0])
        _TN = np.sum(y_test[y_test == 0] == y_pre[y_test == 0])
        _P.append(_TP / (_TP + _FP))
        _R.append(_TP / (_TP + _FN))
    _P.append(0)
    _R.append(1)
    _order = np.argsort(_R)
    _P = np.array(_P)[_order]
    _R = np.array(_R)[_order]
    if draw:
        plt.plot(_R, _P, '-r')
        plt.title('Precise-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precise')
        plt.show()
        return _P, _R
    else:
        return _P, _R


def ROC(y_test, y_predict, draw=False):
    """
    Receiver Operating Characteristic Curve
    ROC曲线
    :param y_test: y_true
    :param y_predict: y_predict
    :param draw: Choose whether to draw or not
    :return: TPR and FPR
    """
    while y_test.ndim != 1:
        y_test = np.squeeze(y_test)
    while y_predict.ndim != 1:
        y_predict = np.squeeze(y_predict)
    _FPR = [0]
    _TPR = [0]
    for i in np.sort(y_predict)[::-1]:
        y_pre = pp.Binarize(y_predict, i)
        _TP = np.sum(y_test[y_test == 1] == y_pre[y_test == 1])
        _FN = np.sum(y_test[y_test == 1] != y_pre[y_test == 1])
        _FP = np.sum(y_test[y_test == 0] != y_pre[y_test == 0])
        _TN = np.sum(y_test[y_test == 0] == y_pre[y_test == 0])
        _FPR.append(_FP / (_TN + _FP))
        _TPR.append(_TP / (_TP + _FN))
    _TPR.append(1)
    _FPR.append(1)
    if draw:
        plt.plot(_FPR, _TPR, '-r')
        plt.title('Receiver Operating Characteristic Curve')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.show()
        return _FPR, _TPR
    else:
        return _FPR, _TPR


def F1score(y_test, y_predict):
    """
    F1 Score
    f1-score评价
    :param y_test: y_true
    :param y_predict: y_predict
    :return: Score
    """
    while y_test.ndim != 1:
        y_test = np.squeeze(y_test)
    while y_predict.ndim != 1:
        y_predict = np.squeeze(y_predict)
    _TP = np.sum(y_test[y_test == 1] == y_predict[y_test == 1])
    _FN = np.sum(y_test[y_test == 0] != y_predict[y_test == 0])
    _FP = np.sum(y_test[y_test == 1] != y_predict[y_test == 1])
    _TN = np.sum(y_test[y_test == 0] == y_predict[y_test == 0])
    _P = (_TP / (_TP + _FP))
    _R = (_TP / (_TP + _FN))
    _result = (2 * _P * _R) / (_P + _R)
    return _result


def accuracy(y_test, y_predict):
    """
    Accuracy Score
    准确性评价
    :param y_test:
    :param y_predict:
    :return: Score
    """
    while y_test.ndim != 1:
        y_test = np.squeeze(y_test)
    while y_predict.ndim != 1:
        y_predict = np.squeeze(y_predict)
    result = np.mean(y_test == y_predict)
    return result

