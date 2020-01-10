from sklearn.datasets import load_boston
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LinearRegression,SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
#波士顿放假预测
#获取数据划分数据集
#特征工程：标准化
#预估器
#调优
#模型评估
def linear1():
    boston = load_boston()
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22)
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    estimator = LinearRegression()
    estimator.fit(x_train, y_train)
    y_predict = estimator.predict(x_test)
    error = mean_squared_error(y_test,y_predict)

    print("正规方程-权重系数：\n",estimator.coef_)#特征有几个，权重系数就有几个
    print("正规方程-偏置为：\n",estimator.intercept_)
    print("正规方程-均方误差\n",error)

    return None

def linear2():
    boston = load_boston()
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22)
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    estimator = SGDRegressor(learning_rate="constant", eta0=0.01, max_iter=100)
    estimator.fit(x_train, y_train)
    print("梯度下降-权重系数：\n",estimator.coef_)
    print("梯度下降-偏置为：\n",estimator.intercept_)

    return None


if __name__=="__main__":
    linear1()
    linear2()