import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataSet_tools.data_processing import PreProcessing
from sklearn.linear_model import LinearRegression
from DataSet_tools.score import accuracy

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
pp = PreProcessing()

data = pd.read_csv('train.csv').drop(['id_num'], axis=1)
data = pp.FillNan(data, method='fb')
obj = data.dtypes[data.dtypes == object].index
copy = data[obj].copy()
data[obj] = copy
y = data.iloc[:, [-1]]
x = data.iloc[:, :-1]

