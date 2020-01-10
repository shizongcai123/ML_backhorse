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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
estimator =  RandomForestClassifier()
param_dict = {"n_estimators":[1,3,5,7,9,11],"max_depth":[5,8,15,25,30]}
estimator = GridSearchCV(estimator,param_grid= param_dict,cv=3)
estimator.fit(x_train,y_train)

print("最佳参数：\n",estimator.best_params_)
