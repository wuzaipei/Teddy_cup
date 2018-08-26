import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.misc import imresize
from sklearn.neighbors import KNeighborsClassifier

# 1、建立模型对数据进行训练
   #n_clusetrs 这个是设置你要分为多少类

def kmes(http):
    a = os.listdir(http)
    n = len(a)
    y_t = [None]*n
    b_q = ['YD1.xlsx', 'YD10.xlsx', 'YD11.xlsx', 'YD2.xlsx', 'YD3.xlsx', 'YD4.xlsx',
           'YD5.xlsx', 'YD6.xlsx', 'YD7.xlsx', 'YD8.xlsx', 'YD9.xlsx']
    for i in range(n):
        x_1 = pd.read_excel(http+'/'+a[i])
        x_1 = np.array(x_1)
        b_q.append(a[i])
        x_1 = x_1[::,[1,3]]
        # 训练
        if a[i] == b_q[1]:
            kms = KMeans(n_clusters=4)
            kms.fit(x_1)  # 这个是无监督学习没有预测训练值
            y_t[i] = kms.predict(x_1)
            plt.scatter(x_1[::, 1], x_1[::, 0], s=2, c=y_t[i])  # 0,len(x_1[::,0]))
        elif a[i] == b_q[2]:
            kms = KMeans(n_clusters=2)
            kms.fit(x_1)  # 这个是无监督学习没有预测训练值
            y_t[i] = kms.predict(x_1)
        elif a[i] == b_q[8]:
            kms = KMeans(n_clusters=9)
            kms.fit(x_1)  # 这个是无监督学习没有预测训练值
            y_t[i] = kms.predict(x_1)
            # plt.scatter(x_1[::,1],x_1[::,0],s=2,c=y_t[i]) # 0,len(x_1[::,0]))
        else:
            kms = KMeans()
            kms.fit(x_1)  # 这个是无监督学习没有预测训练值
            y_t[i] = kms.predict(x_1)
    return y_t,b_q

# Y_t,b_1 = kmes('./代码附件/附件一/电流电压')

# print(type(Y_t))
# plt.show()


# 2、对分解出来的单一态操作记录数据做模式识别。




def Analysis(http_htp):
    The_label = os.listdir(http_htp)
    n = len(The_label)
    for i in range(n):
        data = pd.read_excel(http_htp+'/'+The_label[i])
        data = np.array(data)
        plt.plot(data[::,1])
        plt.title(The_label[i])
        plt.xlabel('时间/(s)')
        plt.ylabel('IC/(A)')
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为SimHei显示中文
        plt.rcParams['axes.unicode_minus'] = False  # 设置正常显示符号
        plt.show()
# Analysis('./代码附件/附件一/电流电压')
Analysis('./代码附件/附件四/设备数据')


def kms_identify(http_http):
    a = os.listdir(http_http)
    n = len(a)
    y_ = [None]*n
    b_q = ['YD1.xlsx', 'YD10.xlsx', 'YD11.xlsx', 'YD2.xlsx', 'YD3.xlsx', 'YD4.xlsx',
           'YD5.xlsx', 'YD6.xlsx', 'YD7.xlsx', 'YD8.xlsx', 'YD9.xlsx']
    knn = KNeighborsClassifier()  # 构造分类器
    for i in range(n):
        data_1 = pd.read_excel(http_http + '/' + a[i])
        data_1 = np.array(data_1)
        data_x_train = data_1[::, [9,10]]
        data_1 = data_1[::, [1, 3, 4]]
        data_x = imresize(data_1, [len(data_1[::, 0]), 3], 'bilinear')
        if a[i] == b_q[0]:
            data_text = imresize(data_x, [394, 3], 'bilinear')
            knn.fit(data_x, data_x_train)
            y_[i] = knn.predict(data_text)  # 进行预测的结果
        elif a[i] == b_q[4]:
            data_text = imresize(data_x, [394, 3], 'bilinear')
            knn.fit(data_x, data_x_train)
            y_[i] = knn.predict(data_text)  # 进行预测的结果
        elif a[i] == b_q[3]: #and b_q[6] and b_q[2]):
            data_text = imresize(data_x, [1283, 3], 'bilinear')
            knn.fit(data_x, data_x_train)
            y_[i] = knn.predict(data_text)  # 进行预测的结果
        elif a[i]==b_q[9]:
            data_text = imresize(data_x, [1283, 3], 'bilinear')
            knn.fit(data_x, data_x_train)
            y_[i] = knn.predict(data_text)  # 进行预测的结果
        elif a[i]==b_q[2]:
            data_text = imresize(data_x, [1283, 3], 'bilinear')
            knn.fit(data_x, data_x_train)
            y_[i] = knn.predict(data_text)  # 进行预测的结果
        else:
            data_text = imresize(data_x, [1524, 3], 'bilinear')
            knn.fit(data_x, data_x_train)
            y_[i] = knn.predict(data_text)  # 进行预测的结果
    return y_

y_pred = kms_identify('./代码附件/单一态分解数据')




# 3、数据保存到txt文件夹中：

# file=open('YD1.txt','w')
# file.write(str(y_pred[0][0]))
# file.close()


import csv
for i in range(len(y_pred)):
    data = pd.DataFrame(y_pred[i])
    data.to_csv('./保存file4/data%d.csv'%(i), index=False, sep=',', quoting=csv.QUOTE_NONNUMERIC)


# 4、电量估计分析
import 泰迪杯.electric as ele
def ele_dl(http_1):
    a = os.listdir(http_1)
    n = len(a)
    x = [];y = []
    d_ele = [None]*2
    b_q = ['YD1.xlsx', 'YD10.xlsx', 'YD11.xlsx', 'YD2.xlsx', 'YD3.xlsx', 'YD4.xlsx',
           'YD5.xlsx', 'YD6.xlsx', 'YD7.xlsx', 'YD8.xlsx', 'YD9.xlsx']
    for i in range(n):
        data_sq = pd.read_excel(http_1+'/'+a[i])
        data_sq = np.array(data_sq)[::,1:-1]
        x_2d = data_sq
        x_8d = data_sq
        if a[i] == b_q[0]:
            x_2d[::,-1] = ele.electric_quantity(data_sq)
            x_2ele = imresize(x_2d,[893,len(x_2d[0,::])])
            x = x_2ele[::, -1]
        elif a[i] == b_q[4]:
            x_8d[::,-1] = ele.electric_quantity(data_sq)
            x_8ele = imresize(x_2d, [893, len(x_8d[0, ::])])
            y = x_8ele[::, -1]


    return x,y

x,y = ele_dl('./代码附件/附件一/电流电压')


plt.plot(x/1000)
plt.title('YD1电量')
plt.xlabel('时间/(s)')
plt.ylabel('实时电量/(kwh)')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为SimHei显示中文
plt.rcParams['axes.unicode_minus'] = False  # 设置正常显示符号

plt.show()
plt.plot(y/1000)
plt.title('YD3电量')
plt.xlabel('时间/(s)')
plt.ylabel('实时电量/(kwh)')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为SimHei显示中文
plt.rcParams['axes.unicode_minus'] = False  # 设置正常显示符号

plt.show()

