import numpy as np
import matplotlib.pyplot as plt

X = np.array([[1,3,3],                  # 输入数据
              [1,4,3],
              [1,1,1]])
Y = np.array([[1],                      # 标签
              [1],
              [-1]])

W = (np.random.random([3,1])-0.5)*2     # 权值初始化，3行1列，-1到1
print(W)

lr = 0.11                               # 学习率
n = 0                                   # 计算迭代次数
O = 0                                   # 神经网络输出

def update():
    global X,Y,W,lr
    O = np.sign(np.dot(X,W)) # shape:(3,1)
    W_C = lr*(X.T.dot(Y-O))/int(X.shape[0])
    W = W + W_C


# In[9]:

for _ in range(100):
    update()                            # 更新权值
    n+=1
    print(W)                            # 打印当前权值
    print(n)                            # 打印迭代次数
    O = np.sign(np.dot(X,W))            # 计算当前输出  
    if(O == Y).all():                   # 如果实际输出等于期望输出，模型收敛，循环结束
        print('Finished')
        print('epoch:',n)
        break

x1 = [3,4]                              # 正样本
y1 = [3,3]

x2 = [1]                                # 负样本
y2 = [1]

k = -W[1]/W[2]                          # 计算斜率、截距
d = -W[0]/W[2]
print('k=',k)
print('d=',d)

xdata = (0,5)

plt.figure()
plt.plot(xdata,xdata*k+d,'r')
plt.scatter(x1,y1,c='b')
plt.scatter(x2,y2,c='y')
plt.show()

