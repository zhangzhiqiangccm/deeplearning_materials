{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.feature_extraction import DictVectorizer\n",
    "from sklearn import tree\n",
    "from sklearn import preprocessing\n",
    "import csv"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# 读入数据\n",
    "Dtree = open(r'AllElectronics.csv', 'r')\n",
    "reader = csv.reader(Dtree)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['RID', 'age', 'income', 'student', 'credit_rating', 'class_buys_computer']\n"
     ]
    }
   ],
   "source": [
    "# 获取第一行数据\n",
    "headers = reader.__next__()\n",
    "print(headers)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']\n",
      "[{'age': 'youth', 'income': 'high', 'student': 'no', 'credit_rating': 'fair'}, {'age': 'youth', 'income': 'high', 'student': 'no', 'credit_rating': 'excellent'}, {'age': 'middle_aged', 'income': 'high', 'student': 'no', 'credit_rating': 'fair'}, {'age': 'senior', 'income': 'medium', 'student': 'no', 'credit_rating': 'fair'}, {'age': 'senior', 'income': 'low', 'student': 'yes', 'credit_rating': 'fair'}, {'age': 'senior', 'income': 'low', 'student': 'yes', 'credit_rating': 'excellent'}, {'age': 'middle_aged', 'income': 'low', 'student': 'yes', 'credit_rating': 'excellent'}, {'age': 'youth', 'income': 'medium', 'student': 'no', 'credit_rating': 'fair'}, {'age': 'youth', 'income': 'low', 'student': 'yes', 'credit_rating': 'fair'}, {'age': 'senior', 'income': 'medium', 'student': 'yes', 'credit_rating': 'fair'}, {'age': 'youth', 'income': 'medium', 'student': 'yes', 'credit_rating': 'excellent'}, {'age': 'middle_aged', 'income': 'medium', 'student': 'no', 'credit_rating': 'excellent'}, {'age': 'middle_aged', 'income': 'high', 'student': 'yes', 'credit_rating': 'fair'}, {'age': 'senior', 'income': 'medium', 'student': 'no', 'credit_rating': 'excellent'}]\n"
     ]
    }
   ],
   "source": [
    "# 定义两个列表\n",
    "featureList = []\n",
    "labelList = []\n",
    "# \n",
    "for row in reader:\n",
    "    # 把label存入list\n",
    "    labelList.append(row[-1])\n",
    "    rowDict = {}\n",
    "    for i in range(1, len(row)-1):\n",
    "        #建立一个数据字典\n",
    "        rowDict[headers[i]] = row[i]\n",
    "    # 把数据字典存入list\n",
    "    featureList.append(rowDict)\n",
    "print(labelList)\n",
    "print(featureList)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "x_data: [[0. 0. 1. 0. 1. 1. 0. 0. 1. 0.]\n",
      " [0. 0. 1. 1. 0. 1. 0. 0. 1. 0.]\n",
      " [1. 0. 0. 0. 1. 1. 0. 0. 1. 0.]\n",
      " [0. 1. 0. 0. 1. 0. 0. 1. 1. 0.]\n",
      " [0. 1. 0. 0. 1. 0. 1. 0. 0. 1.]\n",
      " [0. 1. 0. 1. 0. 0. 1. 0. 0. 1.]\n",
      " [1. 0. 0. 1. 0. 0. 1. 0. 0. 1.]\n",
      " [0. 0. 1. 0. 1. 0. 0. 1. 1. 0.]\n",
      " [0. 0. 1. 0. 1. 0. 1. 0. 0. 1.]\n",
      " [0. 1. 0. 0. 1. 0. 0. 1. 0. 1.]\n",
      " [0. 0. 1. 1. 0. 0. 0. 1. 0. 1.]\n",
      " [1. 0. 0. 1. 0. 0. 0. 1. 1. 0.]\n",
      " [1. 0. 0. 0. 1. 1. 0. 0. 0. 1.]\n",
      " [0. 1. 0. 1. 0. 0. 0. 1. 1. 0.]]\n",
      "['age=middle_aged', 'age=senior', 'age=youth', 'credit_rating=excellent', 'credit_rating=fair', 'income=high', 'income=low', 'income=medium', 'student=no', 'student=yes']\n"
     ]
    }
   ],
   "source": [
    "# 把数据转换成01表示\n",
    "vec = DictVectorizer()\n",
    "x_data = vec.fit_transform(featureList).toarray()\n",
    "print(\"x_data: \" + str(x_data))\n",
    "# 打印属性名称\n",
    "print(vec.get_feature_names())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "labelList: ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']\n"
     ]
    }
   ],
   "source": [
    "# 打印标签\n",
    "print(\"labelList: \" + str(labelList))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "y_data: [[0]\n",
      " [0]\n",
      " [1]\n",
      " [1]\n",
      " [1]\n",
      " [0]\n",
      " [1]\n",
      " [0]\n",
      " [1]\n",
      " [1]\n",
      " [1]\n",
      " [1]\n",
      " [1]\n",
      " [0]]\n"
     ]
    }
   ],
   "source": [
    "# 把标签转换成01表示\n",
    "lb = preprocessing.LabelBinarizer()\n",
    "y_data = lb.fit_transform(labelList)\n",
    "print(\"y_data: \" + str(y_data))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,\n",
       "                       max_features=None, max_leaf_nodes=None,\n",
       "                       min_impurity_decrease=0.0, min_impurity_split=None,\n",
       "                       min_samples_leaf=1, min_samples_split=6,\n",
       "                       min_weight_fraction_leaf=0.0, presort=False,\n",
       "                       random_state=None, splitter='best')"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 创建决策树模型\n",
    "# model = tree.DecisionTreeClassifier(criterion='entropy',  max_depth=3, min_samples_split=6)\n",
    "model = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=6)\n",
    "# 输入数据建立模型\n",
    "model.fit(x_data, y_data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "x_test: [0. 0. 1. 0. 1. 1. 0. 0. 1. 0.]\n",
      "predict: [0]\n"
     ]
    }
   ],
   "source": [
    "# 测试\n",
    "x_test = x_data[0]\n",
    "print(\"x_test: \" + str(x_test))\n",
    "\n",
    "predict = model.predict(x_test.reshape(1,-1))\n",
    "print(\"predict: \" + str(predict))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8571428571428571"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.score(x_data, y_data) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "anaconda-cloud": {},
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
