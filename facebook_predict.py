# train.csv, test.csv
# row_ jd: id of the check-in event
# xy: coordinates
# accuracy. location accuracy
# time: timestamp
# place. jd: id of the business, this is the target you are predicting，这个就是目标值
import pandas as pd
data = pd.read_csv()
#基本的数据处理
#缩小数据范围
data.query("x<2.5 & x>2 & y<1.5 & y>1.0")
#处理时间特征
time_value = pd.datetime(data["time"], unit = "s")
date = pd.DatetimeIndex(time_value)
date.weekday#可以得到星期天的数据
date.years#day,hour,

#将时间装进数据集
data["day"] = date.day

#过滤签到次数较少的地点
place_count = data.groupby("place_id").count()["row_id"]#按照palce_id分组,就拿到了一个签到次数的表,row_id表示出现的次数
place_count[place_count > 3]
databool = data["place_id"].isin(place_count[place_count > 3].index.values)#就得到了一组bool值，就是》3的布尔值
#最后对data进行bool索引就得到了bool值为真的数据
data_final = data[databool]


#筛选特征值和目标值
x = data_final[["x" , "y", "accuracy" , "day" , "weekday" , "hour"]]
y = data_final["palce_id"]

#数据集划分
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y)






