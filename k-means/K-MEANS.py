from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt

data = genfromtxt(r"kmeans.txt",delimiter=' ')

data = np.array(data)

x = [x[0] for x in data]
y = [y[1] for y in data]
plt.scatter(x,y)
plt.scatter(data[:,0],data[:,1])
plt.show()

k = 4                                       # 训练模型
model = KMeans(n_clusters=k).fit(data)

center = model.cluster_centers_             # 分类中心点坐标
print(center)

result = model.predict(data)                # 预测结果
print(result)

color = ['r', 'b', 'g', 'y']                # 画出各数据点
for i,d in enumerate(data):
    plt.scatter(d[0],d[1],c=color[result[i]])

mark = ['*r', '*b', '*g', '*y']             # 各分类中心
for i,d in enumerate(center):
    plt.plot(d[0],d[1], mark[i], markersize=20)
    
plt.show()

model.labels_

