# from 泰迪杯.问题二 import pretreatment
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def trait(file_path):
    b_list=os.listdir(file_path)
    n = len(b_list)
    for i in range(n):
        IC = pd.read_excel(file_path+'/'+b_list[i])
        IC = np.array(IC.iloc[::,1:-1])
        length = range(len(IC[::,1]))
        # print(length)
        plt.figure()
        plt.plot(length,IC[::,0])
        plt.plot(length,IC[::,2])
        plt.plot(length,IC[::,3])
        plt.legend(['IC','PC','QC'])
        plt.xlabel('时间/(s)')
        plt.ylabel('IC/PC/QC')
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为SimHei显示中文
        plt.rcParams['axes.unicode_minus'] = False  # 设置正常显示符号
        plt.title(b_list[i])
        plt.show()

def  harmonic(htp):
    # 建立主成分降维模型
    pca = PCA(n_components= 1,whiten= True,svd_solver='randomized')
    a = os.listdir(htp)
    n = len(a)
    data_pred_1=[]
    data_bq = []
    for i in range(n):
        http = htp+'/'+a[i]
        data = pd.read_excel(http)
        data_pca = data.iloc[::,1:51]
        pca.fit(data_pca)  # 里面可以传入需要降维的数据矩阵
        yy = pca.fit_transform(data_pca)  # 降维过后的数据
        plt.plot(np.arange(len(yy)),yy,c='g')
        plt.title(a[i])
        plt.xlabel('时间/(s)')
        plt.ylabel('IC谐波值')
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为SimHei显示中文
        plt.rcParams['axes.unicode_minus'] = False  # 设置正常显示符号
        plt.show()

# 1、特征数据的提取
file_path = './代码附件/附件一/电流电压'
trait(file_path)
harmonic('./代码附件/附件一/谐波数据')

# 2、用电设备电量

def electric(ele_path):
    bq_list = os.listdir(ele_path)
    n = len(bq_list)
    data_electric = []
    for i in range(n):
        data_electric = pd.read_excel(ele_path+'/'+bq_list[i])
        data_electric = np.array(data_electric)[:,1:-1]

    return data_electric

a = electric(file_path)

print((a[1][0]))

def electric_quantity(data,power):
    # data:类型为 np.array  .power :为设备的额电功率。
    n = len(data)
    data_ele = []
    q = data[0][0]*220/(1000*3600)
    for i in range(n):
        q = power+q
        data_ele.append(q)
    return data_ele

# 3、各用电设备电量的求解
power = 0.06  #额电功率
data_ele = electric_quantity(electric(file_path),power)  #单位千瓦时

print(len(data_ele))


