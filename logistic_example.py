import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from DataSet_tools.data_split import train_test_split, cross_validation
from LogisticRegression.LR import LogisticRegression
from DataSet_tools.score import accuracy, ROC, PR, F1score
from sklearn.metrics import precision_recall_curve

data = load_breast_cancer()  # 数据引入
x = pd.DataFrame(data['data'], columns=data['feature_names'])
y = pd.DataFrame(data['target'].reshape(-1, 1), columns=['target'])

# 模型引入
lr = LogisticRegression()

print('十次交叉验证', cross_validation(x, y, lr, accuracy, 10))

# 数据分割
x_train, y_train, x_test, y_test = train_test_split(x, y, 0.2)

# 数据拟合
lr.fit(x_train, y_train)
y_pre = lr.predict(x_test)

# 模型评估
a = accuracy(y_test, y_pre)
f = F1score(y_test, y_pre)
precise, recall = PR(y_test, y_pre)
TPR, FPR = ROC(y_test, y_pre)
print('accuracy:', a)
print('f1_score:', f)

# 作图
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax.plot(recall, precise)
ax.set(title='P-R curve', xlabel='recall', ylabel='precise')
ax = fig.add_subplot(1, 2, 2)
ax.plot(FPR, TPR)
ax.set(title='ROC curve', xlabel='FPR', ylabel='TPR')
plt.show()
