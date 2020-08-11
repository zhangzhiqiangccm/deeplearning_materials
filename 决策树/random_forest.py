from sklearn.feature_extraction import DictVectorizer
from sklearn import tree
from sklearn import preprocessing
import csv
import numpy as np
Dtree = open(r'AllElectronics.csv', 'r')           # 读入数据
reader = csv.reader(Dtree)
headers = reader.__next__()                        # 获取第一行数据（特征名）
featureList = []                                   # 定义两个列表
labelList = []
for row in reader:
    labelList.append(row[-1])
    rowDict = {}
    for i in range(1, len(row)-1):
        rowDict[headers[i]] = row[i]
featureList.append(rowDict)
x_data = np.array([[0, 0, 0, 0, 0, 1],
                   [0, 0, 1, 0, 0, 1],
                   [1, 0, 0, 0, 0, 1],
                   [0, 1, 0, 0, 1, 1],
                   [0, 1, 0, 1, 0, 0],
                   [0, 1, 1, 1, 0, 0],
                   [1, 0, 1, 1, 0, 0],
                   [0, 0, 0, 0, 1, 1],
                   [0, 0, 0, 1, 0, 0],
                   [0, 1, 0, 0, 1, 0],
                   [0, 0, 1, 0, 1, 0],
                   [1, 0, 1, 0, 1, 1],
                   [1, 0, 0, 0, 0, 0],
                   [0, 1, 1, 0, 1, 1]])
print("labelList: " + str(labelList))
lb = preprocessing.LabelBinarizer()
y_data = lb.fit_transform(labelList)
print("y_data: " + str(y_data))
# 创建决策树模型
# model = tree.DecisionTreeClassifier(criterion='entropy',  max_depth=3, min_samples_split=6)
model = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=6)          # 标准->熵；最小样本分裂->6
# 输入数据建立模型
model.fit(x_data, y_data)
x_test = x_data[0]                              # 测试
print("x_test: " + str(x_test))
predict = model.predict(x_test.reshape(1, -1))
print("predict: " + str(predict))
model.score(x_data, y_data)


