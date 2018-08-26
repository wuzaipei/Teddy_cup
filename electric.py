def electric_quantity(data):
    # data:类型为 np.array  .power :为设备的额电功率。
    n = len(data)
    data_ele = []
    for i in range(n):
         data_ele.append((data[i][0]*data[i][1])/(1000))
    return data_ele
