import pandas as pd
import numpy as np
from numpy import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
first_data_train = pd.read_csv("旧金山犯罪率/train.csv")
first_data_test = pd.read_csv("旧金山犯罪率/test.csv")
first_data_train.info()