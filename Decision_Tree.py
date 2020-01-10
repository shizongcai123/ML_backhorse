import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def decision_iris():
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22)
    estimator = DecisionTreeClassifier(criterion='entropy')
    estimator.fit(x_train,y_train)
    #可视化决策树
    export_graphviz(estimator,out_file="tree.dot",feature_names=iris.feature_names)

if __name__ =="__main__":
