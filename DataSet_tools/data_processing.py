import numpy as np
import pandas as pd


class PreProcessing:
    """
    Data pre-processing
    数据预处理
    """
    def __init__(self, _length=None, _min=None):
        """
        :param _length: MinMaxScale Parameter
        :param _min: MinMaxScale Parameter
        """
        self.len = _length
        self.min = _min

    def MinMaxScale(self, _data, keep=False, axis=0):
        """Make the data between 0 and 1
        数据归一化
        :param _data: Dataset
        :param keep: Use the last parameters
        :param axis: Axis
        """
        if keep:
            return (_data - self.min) / self.len
        else:
            _max = np.max(_data, axis=axis)
            self.min = np.min(_data, axis=axis)
            self.len = (_max - self.min)
            return (_data - self.min) / self.len

    @staticmethod
    def Standardization(_data, axis=0):
        """
        Make the mean of the data 0 and the standard deviation 1
        数据标准化
        :param _data: Dataset
        :param axis: Axis
        :return: New dataset
        """
        _std = np.std(_data, axis=axis)
        _mean = np.mean(_data, axis=axis)
        return (_data - _mean) / _std

    @staticmethod
    def Normalization(_data, axis=0, norm='l2'):
        """
        l1, l2 Normalization
        l1, l2正则化
        :param _data: Dataset
        :param axis: Axis
        :param norm: Choose l1 or l2
        :return:New dataset
        """
        if norm == 'l1':
            return _data / np.sum(np.abs(_data), axis=axis, keepdims=True)
        elif norm == 'l2':
            return _data / np.sqrt(np.sum(np.square(_data), axis=axis, keepdims=True))
        else:
            raise KeyError(norm)

    @staticmethod
    def OneHotEncode(_data):
        """
        OneHotEncode
        独热编码
        :param _data: Dataset
        :return: New dataset
        """
        _data = np.array(_data)
        if _data.shape[1] == 1:
            _Data = np.zeros([len(_data), len(set(_data))])
            _Data[range(len(_data)), _data] = 1
            return _Data
        for i in range(_data.shape[1]):
            _dt = _data[:, i]
            # 对每一列进行独热编码后，进行合并
            if i == 0:
                _Data = np.zeros([len(_dt), len(set(_dt))])
                _Data[list(range(len(_dt))), _dt] = 1
            else:
                _next = np.zeros([len(_dt), len(set(_dt))])
                _next[list(range(len(_dt))), _dt] = 1
                _Data = np.concatenate([_Data, _next], axis=1)
        return _Data

    @staticmethod
    def LabelEncode(_data):
        """
        LabelEncode
        标签编码
        :param _data: Dataset
        :return: New dataset
        """
        if isinstance(_data, pd.DataFrame):
            _columns = _data.columns
            _index = _data.index
            _data = np.array(_data)
            for i in range(_data.shape[1]):
                _df = _data[:, i].copy()
                for j in range(len(set(_df))):
                    _df[_df == list(set(_df.tolist()))[j]] = j
                _data[:, i] = _df
            return pd.DataFrame(_data, index=_index, columns=_columns, dtype='int64')
        elif isinstance(_data, np.ndarray):
            for i in range(_data.shape[1]):
                _df = _data[:, i]
                for j in range(len(set(_df))):
                    _df[_df == list(set(_df.tolist()))[j]] = j
                _data[:, i] = _df
            return _data
        else:
            raise TypeError(str(type(_data))+"is not <class 'numpy.ndarray'> or <class 'pandas.DataFrame'>")

    @staticmethod
    def FillNan(_data, limit=10, method='mean', _value=0):
        # 用了pandas的fillna方法进行了封装
        """
        Fill the NAN
        缺失值填补
        :param _value: The value to fill the nan
        :param _data: Dataset
        :param limit: The Limitations of filling
        :param method: Filling methods, mean or median
        :return: New dataset
        """
        if isinstance(_data, pd.DataFrame):
            _dt = _data.copy()
            if method == 'mean':
                for col in list(_data.columns[_data.isnull().sum() > 0]):
                    _val = _data[col].dropna().mean()
                    _dt[col].fillna(_val, limit=limit, inplace=True)
                return _dt
            if method == 'median':
                for col in list(_data.columns[_data.isnull().sum() > 0]):
                    _val = _data[col].dropna().median()
                    _dt[col].fillna(_val, limit=limit, inplace=True)
                return _dt
            if method == 'fb':
                # 使用前后值进行填充缺失值
                for col in list(_data.columns[_data.isnull().sum() > 0]):
                    _dt[col].fillna(method='ffill', limit=int(limit / 2), inplace=True)
                    _dt[col].fillna(method='bfill', limit=int(limit / 2), inplace=True)
                return _dt
            if method == 'value':
                for col in list(_data.columns[_data.isnull().sum() > 0]):
                    _dt[col].fillna(_value, limit=limit, inplace=True)
                return _dt
        else:
            raise TypeError(str(type(_data))+"is not <class 'pandas.DataFrame'>")

    @staticmethod
    def Clean(_data, column=None):
        """
        Cleaning based on the BoxPlot
        清理异常值
        :param _data: Dataset
        :param column: Class columns
        :return: New dataset
        """
        new_data = _data.copy()
        for i in range(_data.shape[1]):
            if isinstance(column, list):
                col = np.array(column)
                if np.any(col == i):
                    pass
                else:
                    _series = _data.iloc[:, i].copy()
                    _iq = _series.quantile(0.75) - _series.quantile(0.25)
                    _iq1 = _series.quantile(0.25) - 1.5 * _iq
                    _iq2 = _series.quantile(0.75) + 1.5 * _iq
                    _q = _series.quantile(0.5)
                    _series[_series < _iq1] = _q
                    _series[_series > _iq2] = _q
                    new_data.iloc[:, i] = _series
            elif isinstance(column, int) or (column is None):
                if i == column:
                    pass
                else:
                    _series = _data.iloc[:, i].copy()
                    _iq = _series.quantile(0.75) - _series.quantile(0.25)
                    _iq1 = _series.quantile(0.25) - 1.5 * _iq
                    _iq2 = _series.quantile(0.75) + 1.5 * _iq
                    _q = _series.quantile(0.5)
                    _series[_series < _iq1] = _q
                    _series[_series > _iq2] = _q
                    new_data.iloc[:, i] = _series
            else:
                raise KeyError(column)
        return new_data

    @staticmethod
    def Binarize(_data, threshold=0):
        """
        Data binarize
        数据二值化
        :param _data: Dataset
        :param threshold: Default 0
        :return: New dataset
        """
        _dt = _data.copy()
        _dt[_data >= threshold] = 1
        _dt[_data < threshold] = 0
        return _dt

    @staticmethod
    def Cut(_data, bins=2, labels=None):
        # 用了pandas的cut方法
        """
        Based on pandas.cut()
        数据切割
        :param _data: Series or array
        :param bins: The number of split
        :param labels: Labels for each bin
        :return: New data
        """
        if isinstance(_data, pd.Series):
            _index = _data.index
            _data = _data.values
            _data = pd.cut(_data, bins=bins, labels=labels)
            _data = pd.Series(_data, index=_index)
            return _data
        elif isinstance(_data, pd.DataFrame) and _data.shape[1] == 1:
            _index = _data.index
            _columns = _data.columns
            _data = _data.T.values[0]
            _data = pd.cut(_data, bins=bins, labels=labels)
            _data = pd.DataFrame([_data], index=_index, columns=_columns)
            return _data
        else:
            raise TypeError(str(type(_data))+"is not <class 'numpy.ndarray'> or <class 'pandas.DataFrame'>")
