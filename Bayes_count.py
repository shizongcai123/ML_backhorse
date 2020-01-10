from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from

news= fetch_
x_train, x_test, y_train, y_test = train_test_split(news.data,news.target,random_state=8)
transfer = TfidfVectorizer()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)
estimator = MultinomialNB()
estimator.fit(x_train,y_train)
