from sklearn.tree import DecisionTreeClassifier,export_graphviz
from numpy import *
import pandas as pd
  #获取数据
  #数据处理
    #缺失值处理
    #特征值->字典类型
  #准备好特征值，目标值
  #划分数据集
  #特征工程：字典特征抽取
  #决策树预估器流程
  #模型评估
train = pd.read_csv("titanic/train.csv")
test = pd.read_csv("titanic/test.csv")
x_train = train[["Pclass","Age","Sex"]]
x_test = test[["Pclass","Age","Sex"]]
y_train= train["Survived"]
#缺失值处理，填补
x_train["Age"].fillna(x_train["Age"].mean(),inplace=True)
#转换成字典,因为python提供了字典一次性所有类别全部编码，所以先转字典
x_train = x_train.to_dict(orient="records")

x_test["Age"].fillna(x_test["Age"].mean(),inplace=True)
#转换成字典,因为python提供了字典一次性所有类别全部编码，所以先转字典
x_test = x_test.to_dict(orient="records")
#字典特征抽取,因为前边转成字典的，内容是字符串，所以需要抽取成数字特征
from sklearn.feature_extraction import DictVectorizer
transfer = DictVectorizer()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

estimator = DecisionTreeClassifier(criterion="entropy")#还有一个深度参数，可以用之前的网格调参来优化一下
estimator.fit(x_train,y_train)
export_graphviz(estimator, out_file="titanic_trees.dot",feature_names= transfer.get_feature_names())
y_predict = estimator.predict(x_test)
score = estimator.score(x_test,y_predict)
print("准确率为：\n",score)

