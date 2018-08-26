import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.misc import imresize
from numpy.fft import fft,ifft
from sklearn.neighbors import KNeighborsClassifier

def  harmonic(htp):
    # 建立主成分降维模型
    pca = PCA(n_components= 1,whiten= True,svd_solver='randomized')
    a = os.listdir(htp)
    n = len(a)

    data_bq = []
    date_x = [None]*n
    for i in range(n):
        http = htp+'/'+a[i]
        data = pd.read_excel(http)
        yy = data.iloc[::,1:51]

        yy = imresize(yy, [100, 50])
        yy = fft(data.iloc[::, 1:51])
        yy[np.where(np.abs(yy) < 1e1)] = 0
        cat_data_ifft = ifft(yy)
        yy = np.real(cat_data_ifft)

        pca.fit(yy)  # 里面可以传入需要降维的数据矩阵
        yy = pca.fit_transform(yy)  # 降维过后的数据

        # #imresize参数： 'nearest'最近邻插值（默认）'bilinear'双线性插值'bicubic'双三次插值
        yy = imresize(yy,[100,1])
        date_x[i] = yy.reshape(-1)
        data_bq.append(a[i])

        plt.plot(np.arange(len(yy)),yy,c='g')
        plt.title(a[i])
        plt.xlabel('时间/(s)')
        plt.ylabel('IC谐波值')
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为SimHei显示中文
        plt.rcParams['axes.unicode_minus'] = False  # 设置正常显示符号
        plt.show()
    return date_x,data_bq
xx,bq = harmonic('./代码附件/附件一/谐波数据')

data_x,data_bq = harmonic('./代码附件/附件三/无操作记录/谐波数据')


# data = np.array(xx)
# print(bq)
# data = np.array(data)
# y = data[3]+data[9]
# y1 = data[4]+data[6]+data[2]
# y3 = data[0]+data[3]+data[4]+data[7]+data[8]
# plt.plot(y,c='r'),plt.title('2_8'),plt.show()
# plt.plot(y1,c='r'),plt.title('3_5_11'),plt.show()
# plt.plot(y3,c='r'),plt.title('1_2_3_6_7'),plt.show()


# 多元线性回归做分解
'''
def fly(http):
    a = os.listdir(http)
    n = len(a)
    for i in range(n):
        x_1 = pd.read_excel(http+'/'+a[i])
        x_1 = np.array(x_1)
        y_1 = np.multiply(x_1[::,1],x_1[::,2],x_1[::,-1])/360000
        plt.plot(y_1)
        plt.title(a[i])
        plt.xlabel('time')
        plt.ylabel('P/(w)')
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为SimHei显示中文
        plt.rcParams['axes.unicode_minus'] = False  # 设置正常显示符号
        plt.show()
fly('./代码附件/附件一/电流电压')
'''

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
        if a[i] == b_q[3]:
            data_text = imresize(data_x, [892, 3], 'bilinear')
            knn.fit(data_x, data_x_train)
            y_[i] = knn.predict(data_text)  # 进行预测的结果
        elif a[i] == b_q[9]:
            data_text = imresize(data_x, [892, 3], 'bilinear')
            knn.fit(data_x, data_x_train)
            y_[i] = knn.predict(data_text)  # 进行预测的结果
        elif a[i] == b_q[4]: #and b_q[6] and b_q[2]):
            data_text = imresize(data_x, [694, 3], 'bilinear')
            knn.fit(data_x, data_x_train)
            y_[i] = knn.predict(data_text)  # 进行预测的结果
        elif a[i]==b_q[6]:
            data_text = imresize(data_x, [694, 3], 'bilinear')
            knn.fit(data_x, data_x_train)
            y_[i] = knn.predict(data_text)  # 进行预测的结果
        elif a[i]==b_q[2]:
            data_text = imresize(data_x, [694, 3], 'bilinear')
            knn.fit(data_x, data_x_train)
            y_[i] = knn.predict(data_text)  # 进行预测的结果
        else:
            data_text = imresize(data_x, [1520, 3], 'bilinear')
            knn.fit(data_x, data_x_train)
            y_[i] = knn.predict(data_text)  # 进行预测的结果
    return y_

y_pred = kms_identify('./代码附件/单一态分解数据')

print(type(y_pred[4]))

# 3、数据保存到cvs文件夹中：

import csv
for i in range(len(y_pred)):
    data = pd.DataFrame(y_pred[i])
    data.to_csv('./保存/data%d.csv'%(i), index=False, sep=',', quoting=csv.QUOTE_NONNUMERIC)

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
        if a[i] == b_q[3]:
            x_2d[::,-1] = ele.electric_quantity(data_sq)
            x_2ele = imresize(x_2d,[893,len(x_2d[0,::])])
            x = x_2ele[::, -1]
        elif a[i] == b_q[9]:
            x_8d[::,-1] = ele.electric_quantity(data_sq)
            x_8ele = imresize(x_2d, [893, len(x_8d[0, ::])])
            y = x_8ele[::, -1]


    return x,y

x,y = ele_dl('./代码附件/附件一/电流电压')


plt.plot(x/1000)
plt.title('YD2电量')
plt.xlabel('时间/(s)')
plt.ylabel('实时电量/(kwh)')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为SimHei显示中文
plt.rcParams['axes.unicode_minus'] = False  # 设置正常显示符号

plt.show()
plt.plot(y/1000)
plt.title('YD8电量')
plt.xlabel('时间/(s)')
plt.ylabel('实时电量/(kwh)')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为SimHei显示中文
plt.rcParams['axes.unicode_minus'] = False  # 设置正常显示符号

plt.show()

import csv

data = pd.DataFrame(x)
data.to_csv('./保存/YD2.csv', index=False, sep=',', quoting=csv.QUOTE_NONNUMERIC)
data = pd.DataFrame(x)
data.to_csv('./保存/YD8.csv', index=False, sep=',', quoting=csv.QUOTE_NONNUMERIC)
