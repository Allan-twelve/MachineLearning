import numpy as np
import pandas as pd


def train_test_split(_x, _y, size_of_test):
    """
    Leave out split
    留出法数据分割
    :param _x: x
    :param _y: y
    :param size_of_test: The size is from 0 to 1
    :return: Train data and test data
    """
    if isinstance(_x, pd.DataFrame):
        x_col = _x.columns
        y_col = _y.columns
        _x = np.array(_x)
        _y = np.array(_y)
        _l = _x.shape[0]
        length = _x.shape[1]
        size = int(size_of_test * _l)
        # 这里的随机数不存在重复的
        rand = np.random.choice(_l, size=size, replace=False)
        _con = np.concatenate([_x, _y], axis=1)
        _test = _con[rand]
        _train = np.delete(_con, rand, axis=0)
        np.random.shuffle(_train)
        x_train, y_train = np.split(_train, (length,), axis=1)
        x_test, y_test = np.split(_test, (length,), axis=1)
        x_train = pd.DataFrame(x_train, columns=x_col)
        y_train = pd.DataFrame(y_train, columns=y_col)
        x_test = pd.DataFrame(x_test, columns=x_col)
        y_test = pd.DataFrame(y_test, columns=y_col)
        return x_train, x_test, y_train, y_test
    elif isinstance(_x, np.ndarray):
        _l = _x.shape[0]
        length = _x.shape[1]
        size = int(size_of_test * _l)
        # 这里的随机数不存在重复的
        rand = np.random.choice(_l, size=size, replace=False)
        _y = np.array(_y).reshape(-1, 1)
        _con = np.concatenate([_x, _y], axis=1)
        _test = _con[rand]
        _train = np.delete(_con, rand, axis=0)
        np.random.shuffle(_train)
        x_train, y_train = np.split(_train, (length,), axis=1)
        x_test, y_test = np.split(_test, (length,), axis=1)
        return x_train, x_test, y_train, y_test
    else:
        raise TypeError(str(type(_x))+"is not <class 'numpy.ndarray'> or <class 'pandas.DataFrame'>")


def cross_validation(_x, _y, _class, score_fun, k=5):
    """
    Cross validation
    交叉验证法
    :param _x: x
    :param _y: y
    :param _class: The model you want to use for training the data
    :param score_fun: The function you want to use for scoring the data
    :param k: Times
    :return: The mean score
    """
    if isinstance(_x, pd.DataFrame):
        _x = np.array(_x)
        _y = np.array(_y)
        length = _x.shape[1]
        _con = np.concatenate([_x, _y], axis=1)
        np.random.shuffle(_con)
        _x, _y = np.split(_con, (length,), axis=1)
    elif isinstance(_x, np.ndarray):
        length = _x.shape[1]
        _con = np.concatenate([_x, _y], axis=1)
        np.random.shuffle(_con)
        _x, _y = np.split(_con, (length,), axis=1)
    else:
        raise TypeError(str(type(_x))+"is not <class 'numpy.ndarray'> or <class 'pandas.DataFrame'>")
    length = _x.shape[0]
    split_data_x = np.split(_x, [(i + 1) * int(length / k) for i in range(k - 1)], axis=0)
    split_data_y = np.split(_y, [(i + 1) * int(length / k) for i in range(k - 1)], axis=0)
    score = []
    for i in range(len(split_data_x)):
        x_copy = split_data_x.copy()
        y_copy = split_data_y.copy()
        x_test = x_copy[i]
        y_test = y_copy[i]
        x_copy.pop(i)
        y_copy.pop(i)
        x_train = np.concatenate(x_copy)
        y_train = np.concatenate(y_copy)
        # 模型必须要有fit和predict的用法
        _class.fit(x_train, y_train)
        y_predict = _class.predict(x_test)
        score_i = score_fun(y_predict, y_test)
        score.append(score_i)
    return np.mean(score)


def bootstrapping(_x, _y, m):
    """
    Boot strapping
    自助法分割数据
    :param _x: x
    :param _y: y
    :param m: Number of times to take out
    :return: train data and test data
    """
    if isinstance(_x, pd.DataFrame):
        x_columns = list(_x.columns.values)
        y_columns = list(_y.columns.values)
        _x = np.array(_x)
        _y = np.array(_y)
        length = _x.shape[1]
        # 将x，y先合并再分开
        _con = np.concatenate([_x, _y], axis=1)
        # 这里随机数存在重复，所以是有放回的抽取
        train_index = np.random.randint(0, _x.shape[0], m)
        _train = _con[train_index]
        del_index = list(set(train_index))
        _test = np.delete(_con, del_index, axis=0)
        x_train, y_train = np.split(_train, (length,), axis=1)
        x_test, y_test = np.split(_test, (length,), axis=1)
        x_train = pd.DataFrame(x_train, columns=x_columns)
        y_train = pd.DataFrame(y_train, columns=y_columns)
        x_test = pd.DataFrame(x_test, columns=x_columns)
        y_test = pd.DataFrame(y_test, columns=y_columns)
        return x_train, y_train, x_test, y_test
    elif isinstance(_x, np.ndarray):
        length = _x.shape[1]
        # 将x，y先合并再分开
        _con = np.concatenate([_x, _y], axis=1)
        # 这里随机数存在重复，所以是有放回的抽取
        train_index = np.random.randint(0, _x.shape[0], m)
        _train = _con[train_index]
        _test = np.delete(_con, train_index, axis=0)
        x_train, y_train = np.split(_train, (length,), axis=1)
        x_test, y_test = np.split(_test, (length,), axis=1)
        return x_train, y_train, x_test, y_test
    else:
        raise ValueError(str(type(_x))+"is not <class 'numpy.ndarray'> or <class 'pandas.DataFrame'>")
