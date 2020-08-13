
# coding: utf-8

# In[2]:

import pandas 
titanic = pandas.read_csv("titanic_train.csv")
titanic


# In[3]:

# 空余的age填充整体age的中值
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
print(titanic.describe())


# In[4]:

titanic


# In[5]:

print(titanic["Sex"].unique())

# 把male变成0，把female变成1
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1


# In[6]:

titanic


# In[7]:

print(titanic["Embarked"].unique())
# 数据填充
titanic["Embarked"] = titanic["Embarked"].fillna('S')
# 把类别变成数字
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2


# In[8]:

titanic


# In[9]:

from sklearn.preprocessing import StandardScaler

# 选定特征
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
x_data = titanic[predictors]
y_data = titanic["Survived"]

# 数据标准化
scaler = StandardScaler()
x_data = scaler.fit_transform(x_data)


# # 逻辑回归

# In[10]:

from sklearn import cross_validation
from sklearn import model_selection  
from sklearn.linear_model import LogisticRegression

# 逻辑回归模型
LR = LogisticRegression()
# 计算交叉验证的误差
scores = cross_validation.cross_val_score(LR, x_data, y_data, cv=3)
# 求平均
print(scores.mean())


# # 神经网络

# In[11]:

from sklearn.neural_network import MLPClassifier

# 建模
mlp = MLPClassifier(hidden_layer_sizes=(20,10),max_iter=1000)
# 计算交叉验证的误差
scores = cross_validation.cross_val_score(mlp, x_data, y_data, cv=3)
# 求平均
print(scores.mean())


# # KNN

# In[23]:

from sklearn import neighbors

knn = neighbors.KNeighborsClassifier(21)
# 计算交叉验证的误差
scores = cross_validation.cross_val_score(knn, x_data, y_data, cv=3)
# 求平均
print(scores.mean())


# # 决策树

# In[26]:

from sklearn import tree

# 决策树模型
dtree = tree.DecisionTreeClassifier(max_depth=5, min_samples_split=4)
# 计算交叉验证的误差
scores = cross_validation.cross_val_score(dtree, x_data, y_data, cv=3)
# 求平均
print(scores.mean())


# # 随机森林

# In[27]:

# 随机森林
from sklearn.ensemble import RandomForestClassifier

RF1 = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2)
# 计算交叉验证的误差
scores = cross_validation.cross_val_score(RF1, x_data, y_data, cv=3)
# 求平均
print(scores.mean())


# In[28]:

RF2 = RandomForestClassifier(n_estimators=100, min_samples_split=4)
# 计算交叉验证的误差
scores = cross_validation.cross_val_score(RF2, x_data, y_data, cv=3)
# 求平均
print(scores.mean())


# # Bagging

# In[29]:

from sklearn.ensemble import BaggingClassifier

bagging_clf = BaggingClassifier(RF2, n_estimators=20)
# 计算交叉验证的误差
scores = cross_validation.cross_val_score(bagging_clf, x_data, y_data, cv=3)
# 求平均
print(scores.mean())


# # Adaboost

# In[31]:

from sklearn.ensemble import AdaBoostClassifier

# AdaBoost模型
adaboost = AdaBoostClassifier(bagging_clf,n_estimators=10)
# 计算交叉验证的误差
scores = cross_validation.cross_val_score(adaboost, x_data, y_data, cv=3)
# 求平均
print(scores.mean())


# # Stacking

# In[32]:

from sklearn.ensemble import VotingClassifier
from mlxtend.classifier import StackingClassifier 

sclf = StackingClassifier(classifiers=[bagging_clf, mlp, LR],   
                          meta_classifier=LogisticRegression())

sclf2 = VotingClassifier([('adaboost',adaboost), ('mlp',mlp), ('LR',LR),('knn',knn),('dtree',dtree)])  

# 计算交叉验证的误差
scores = cross_validation.cross_val_score(sclf2, x_data, y_data, cv=3)
# 求平均
print(scores.mean())


# In[ ]:



