import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from LinearRegression import NormalEquation
from LinearRegression import GradientDescent
from DataSet_tools.data_split import train_test_split, cross_validation
from DataSet_tools.data_processing import PreProcessing
from DataSet_tools.score import rSquare, MAE, MSE, MAPE, RMSE

# 数据读取
data = load_boston()

# 引入数据预处理库
pp = PreProcessing()
x = pd.DataFrame(data['data'], columns=data['feature_names'])
y = pd.DataFrame(data['target'].reshape(-1, 1), columns=['target'])

train = pd.concat([x, y], axis=1)
# 根据皮尔森系数进行特征选择
corr = train.corr(method='pearson')['target']
index = corr[np.abs(corr) > 0.1]
train = train[index.index]
x, y = np.split(train, [-1], axis=1)

# 引入线性模型
lr1 = NormalEquation.LinearRegression()
lr2 = GradientDescent.LinearRegression(alpha=0.000003, cycle=1000)

# 十次交叉验证
print('十次交叉验证(正规方程)：', cross_validation(x, y, lr1, rSquare, 10))

# 分割数据集
x_train, y_train, x_test, y_test = train_test_split(x, y, 0.2)

# 数据拟合
lr1.fit(x_train, y_train)
lr2.fit(x_train, y_train)
y_pre1 = lr1.predict(x_test)
y_pre2 = lr2.predict(x_test)

# 模型评估
s11 = rSquare(y_test, y_pre1)
s12 = MAE(y_test, y_pre1)
s13 = MSE(y_test, y_pre1)
s14 = RMSE(y_test, y_pre1)
s15 = MAPE(y_test, y_pre1)
print('正规方程：', 'r2:', s11, 'MAE:', s12, 'MSE:', s13, 'RMSE:', s14, 'MAPE:', s15)
s21 = rSquare(y_test, y_pre2)
s22 = MAE(y_test, y_pre2)
s23 = MSE(y_test, y_pre2)
s24 = RMSE(y_test, y_pre2)
s25 = MAPE(y_test, y_pre2)
print('梯度下降：', 'r2:', s21, 'MAE:', s22, 'MSE:', s23, 'RMSE:', s24, 'MAPE:', s25)

# 可见梯度下降拟合相当不好，进行归一化
new_x = pp.MinMaxScale(x_train)
new_x_test = pp.MinMaxScale(x_test, keep=True)
new_y = pp.MinMaxScale(y_train)
new_y_test = pp.MinMaxScale(y_test, keep=True)

lr3 = GradientDescent.LinearRegression(alpha=0.5, cycle=10000)
lr3.fit(new_x, new_y)
new_pre = lr3.predict(new_x_test)
s31 = rSquare(new_y_test, new_pre)
s32 = MAE(new_y_test, new_pre)
s33 = MSE(new_y_test, new_pre)
s34 = RMSE(new_y_test, new_pre)
s35 = MAPE(new_y_test, new_pre)
print('梯度下降(归一化)：', 'r2:', s31, 'MAE:', s32, 'MSE:', s33, 'RMSE:', s34, 'MAPE:', s35)
