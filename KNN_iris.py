from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
def KNN_iris():
    iris = load_iris()
    x_train,x_test,y_trian,y_test = train_test_split(iris.data,iris.target,random_state=6)

    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    estimator = KNeighborsClassifier(n_neighbors=3)
    estimator.fit(x_train,y_trian)

    y_predict = estimator.predict(x_test)
    print("y_predict:\n",y_predict)
    print("对比:\n",y_predict == y_test)

    k = estimator.score(x_test,y_test)
    print("准确率：\n",k)


if __name__ == "__main__":
    KNN_iris()
