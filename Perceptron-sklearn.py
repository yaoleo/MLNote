# -*- coding: utf-8 -*-
# @Time       : 17-12-7 下午4:57
# @Author     : J.Y.Zhang
# @File       : Perceptron-sklearn.py
# @Description:

from sklearn.linear_model import Perceptron
import pandas as pd
#定义感知机
clf = Perceptron(fit_intercept=False,n_iter=30,shuffle=False)
#使用训练数据进行训练
trainpath = 'Data/Titanic/traindf.csv'
testpath  = 'Data/Titanic/testdf.csv'
traindf = pd.read_csv(trainpath,header=None,names = ["Age","Cabin","Deck","Embarked","Family_Size","Fare","Name","Parch",
                                         "PassengerId","Pclass","Sex","SibSp","Survived","Ticket","Title"])
testdf = pd.read_csv(testpath,header=None,names = ["Age","Cabin","Deck","Embarked","Family_Size","Fare","Name","Parch",
                                         "PassengerId","Pclass","Sex","SibSp","Survived","Ticket","Title"])
traindf = traindf[1:]
testdf  = testdf[1:]
x_data_train = traindf[["Parch"]]
x_data_test  = testdf[["Parch"]]
y_data_train = traindf[["Survived"]]
y_data_test  = testdf[["Survived"]]
print(x_data_train.head())
clf.fit(x_data_train,y_data_train)
#得到训练结果，权重矩阵
print(clf.coef_)
#输出为：[[-0.38478876,4.41537463]]

#超平面的截距，此处输出为：[0.]
print(clf.intercept_)

#利用测试数据进行验证
acc = clf.score(x_data_test,y_data_test)
print(acc)
"""
#得到的输出结果为0.995，这个结果还不错吧。最后，我们将结果用图形显示出来，直观地看一下感知机的结果：
from matplotlib import pyplot as plt
#画出正例和反例的散点图
plt.scatter(positive_x1,positive_x2,c='red')
plt.scatter(negetive_x1,negetive_2,c='blue')
#画出超平面（在本例中即是一条直线）
line_x = np.arange(-4,4)
line_y = line_x * (-clf.coef_[0][0] / clf.coef_[0][1]) - clf.intercept_
plt.plot(line_x,line_y)"""
"""
plt.show()
"""