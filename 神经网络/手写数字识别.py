from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

digits = load_digits()                            # 载入图片数据
x_data = digits.data                              # 数据
y_data = digits.target                            # 标签

print(x_data.shape)
print(y_data.shape)

plt.imshow(digits.images[0],cmap='gray')
plt.show()

x_train,x_test,y_train,y_test = train_test_split(x_data,y_data) #分割数据 -> 1/4为测试数据，3/4为训练数据

mlp = MLPClassifier(hidden_layer_sizes=(100,50),max_iter=500)
mlp.fit(x_train,y_train)

predictions = mlp.predict(x_test)
print(classification_report(y_test, predictions))
