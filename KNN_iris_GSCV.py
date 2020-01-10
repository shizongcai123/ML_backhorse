from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
def KNN_iris():
    iris = load_iris()
    x_train,x_test,y_trian,y_test = train_test_split(iris.data,iris.target,random_state=6)

    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    estimator = KNeighborsClassifier()
    #对模型进行优化GSCV，因此在前边就不需要加K值了
    #参数准备
    param_dict = {"n_neighbors":[1,3,5,7,9,11]}
    estimator = GridSearchCV(estimator,param_grid=param_dict,cv=10)
    estimator.fit(x_train,y_trian)

    y_predict = estimator.predict(x_test)
    print("y_predict:\n",y_predict)
    print("对比:\n",y_predict == y_test)

    k = estimator.score(x_test,y_test)
    print("准确率：\n",k)
    #最佳参数: best_params_
    print("最佳参数：\n",estimator.best_params_)
    #最佳结果: best_score_
    #最佳估计器: best_estimator_
    #交叉验证结果: CV_ results_



if __name__ == "__main__":
    KNN_iris()
