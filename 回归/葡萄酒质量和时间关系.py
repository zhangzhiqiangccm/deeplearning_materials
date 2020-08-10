
# coding: utf-8

# In[134]:

import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[135]:

# 载入数据
data = np.genfromtxt('linear.csv', delimiter=',')
# 画图
plt.scatter(data[1:,0],data[1:,1])
plt.title('Age Vs Quality (Test set)')
plt.xlabel('Age')
plt.ylabel('Quality')
plt.show()


# In[136]:

# 数据拆分
x_train, x_test, y_train, y_test = train_test_split(data[1:, 0], data[1:, 1], test_size = 0.3)


# In[137]:

# 1D->2D，给数据增加一个维度，主要是训练模型的时候，函数要求传入2维的数据
x_train = x_train[:, np.newaxis]
x_test = x_test[:, np.newaxis]


# In[138]:

# 训练模型
model = LinearRegression()
model.fit(x_train, y_train)


# In[139]:

# 训练集的散点图
plt.scatter(x_train, y_train, color = 'b')
# 模型对训练集的预测结果
plt.plot(x_train,model.predict(x_train), color ='r' , linewidth=5)
# 画表头和xy坐标描述
plt.title('Age Vs Quality (Training set)')
plt.xlabel('Age')
plt.ylabel('Quality')
plt.show()


# In[133]:

# 测试集的散点图
plt.scatter(x_test, y_test, color = 'b')
# 模型对测试集的预测结果
plt.plot(x_test,model.predict(x_test), color ='r', linewidth=5)
# 画表头和xy坐标描述
plt.title('Age Vs Quality (Test set)')
plt.xlabel('Age')
plt.ylabel('Quality')
plt.show()


# In[ ]:



