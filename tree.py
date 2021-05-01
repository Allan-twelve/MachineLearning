import pandas as pd
import numpy as np
from DecisionTree.classfication import DecisionTree
from DataSet_tools.score import classification

data1 = pd.read_csv('watermelon1.csv', index_col=0)
data2 = pd.read_csv('watermelon_test.csv', index_col=0)
dt = DecisionTree()
dt.fit(data1)
# createPlot(dt.tree)
print(dt.tree)

pre = []
for i in range(data2.shape[0]):
    pre.append(dt.predict(data2.iloc[i]))
pre = np.array(pre)
test = data2.iloc[:, -1].values
s = classification(pre, test)
print('预测:', pre)
print('实际:', test)
print('正确率:', s)
