
# coding: utf-8

# In[6]:

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# In[7]:

data = pd.read_csv('nba.csv')
data.head()


# In[8]:

minmax_scaler = MinMaxScaler()
# 标准化数据
X = minmax_scaler.fit_transform(data.iloc[:,1:])
X[:5]


# In[9]:

# 肘部法则
loss = []
for i in range(2,10):
    model = KMeans(n_clusters=i).fit(X)
    loss.append(model.inertia_)
    
plt.plot(range(2,10),loss)
plt.xlabel('k')
plt.ylabel('loss')
plt.show()


# In[10]:

k = 4
model = KMeans(n_clusters=k).fit(X)

# 将标签整合到原始数据上
data['clusters'] = model.labels_

data.head()


# In[11]:

for i in range(k):
    print('clusters:',i)
    label_data = data[data['clusters'] == i].iloc[:,0]
    print(label_data.values)


# In[ ]:



