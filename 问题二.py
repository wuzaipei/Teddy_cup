import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
from scipy.misc import imresize
import matplotlib.pyplot as plt


# 1、数据预处理，和数据特征提取

def pretreatment(htp):
    # 建立主成分降维模型
    pca = PCA(n_components= 1,whiten= True,svd_solver='randomized')
    a = os.listdir(htp)
    n = len(a)
    data_pred_1=[]
    data_bq = []
    for i in range(n):
        http = htp+'/'+a[i]
        data = pd.read_excel(http)
        data_pca = data.iloc[::,[1,3,4,5]]
        pca.fit(data_pca)  # 里面可以传入需要降维的数据矩阵
        data_pca = pca.fit_transform(data_pca)  # 降维过后的数据
        # #imresize参数： 'nearest'最近邻插值（默认）'bilinear'双线性插值'bicubic'双三次插值
        data_pred_1.append(imresize(data_pca, [100,1],'bicubic').reshape(-1))
        data_bq.append(a[i])
    return data_pred_1,data_bq

# def Cosine(data_x,data_y):
#     n = len(data_x)
#     m = len(data_y)
#     gl = np.zeros([n,m])
#     for i in range(m):
#         for j in range(n):
#             gl[j,i] = np.sum(np.multiply(data_y[i,:],data_x[j,:]))/(np.sqrt(np.sum(np.multiply(data_y[i,:],data_y[i,:])))
#                                                                     *np.sqrt(np.sum(np.multiply(data_x[j,:],data_x[j,:]))))
#     return gl


# 2、训练谐波数据

htp = './代码附件/附件一/电流电压'
x_train,bq_1 = pretreatment(htp)
x_train = np.array(x_train)
# print(x_train.shape)
# y_train = ['1','2','3','4','5','6','7','8','9','10','11']
y_train = bq_1
# print(y_train)
# 3、预测谐波数据predict。
htp = './代码附件/附件一/预测电压电流'
y_test,bq_2= pretreatment(htp)
y_test = np.array(y_test)


# # 夹角余弦判别
# g = Cosine(x_train,y_test)
# print(g)
# x_min =np.min(g[:,1])
# print(x_min)


# 4、建立机器识别模型
# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier() #构造分类器
# knn.fit(x_train,y_train)
# y_pred = knn.predict(y_test)  #进行预测的结果

# 5、建立支持向量机识别
from sklearn.svm import SVC
svc =SVC(kernel='linear')  #  linear’, ‘poly’, ‘rbf’,  sigmoid  precomputed
svc.fit(x_train,y_train)
y_pred = svc.predict(y_test)

print('设备识别结果：',dict(zip(list(bq_2),list(y_pred))))

# 6、估计实时电量
# 先导入问题一中求解实时电量的函数

from 泰迪杯.electric import electric_quantity
#使用之前的线性回归，将数据进行训练和学习
from sklearn.linear_model import LinearRegression
lrg = LinearRegression()

htp_path = './代码附件/附件一/电流电压'
def electric_forecast(y_pred,htp_path):
    n = len(y_pred)
    # power = pd.read_excel('./代码附件/附件一/额电功率/额电功率.xlsx')
    y_forecast = [None]*n
    for i in range(n):
        c_d = y_pred[i]
        data_1 = pd.read_excel(htp_path+'/'+c_d)
        data_electric = np.array(data_1)[:,::-3]

        data_2 = electric_quantity(data_electric)

        y_test_1 = pd.read_excel('./代码附件/附件一/预测电压电流/'+bq_2[i])
        y_test_2 = np.array(y_test_1)[:,::-3]

        lrg.fit(data_electric,data_2)
        y = lrg.predict(y_test_2)

        y_forecast[i] = lrg.predict(y_test_2)


        plt.figure()
        plt.plot(np.arange(0, len(data_2)), data_2, c='r')
        plt.plot(np.arange(0, len(y)),y, c='g')
        plt.title('设备编号为：' + c_d)
        plt.legend([bq_2[i],c_d])
        plt.xlabel('时间/(s)')
        # plt.ylabel('电量/(kwh)')
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为SimHei显示中文
        plt.rcParams['axes.unicode_minus'] = False  # 设置正常显示符号
        plt.show()

    return y_forecast


y_= electric_forecast(y_pred,htp_path)
# print(y_[0])
# c=y_[1]
# a=np.array(c)
# np.savetxt('data1.txt',a)


