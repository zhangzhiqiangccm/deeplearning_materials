from numpy import genfromtxt
from sklearn import linear_model

data = genfromtxt(r"Delivery.csv",delimiter=',')          # 读入数据 
print(data)

x_data = data[:,:-1]                                      # 切分数据
y_data = data[:,-1]
print(x_data)
print(y_data)

model = linear_model.LinearRegression()                   # 创建模型
model.fit(x_data, y_data)

print("coefficients:",model.coef_)                        # 系数

print("intercept:",model.intercept_)                      # 截距

x_test = [[102,4]]
predict = model.predict(x_test)                           # 测试
print("predict:",predict)

model.score(x_data, y_data)

