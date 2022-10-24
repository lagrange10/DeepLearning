import numpy as np
import copy


# func:插值将序列长度转换成目标长度
def discrete_scale(data, num):
    """
    :param data: original sequeeze
    :param num: target length
    :return d1: sequeeze with target length
    """
    len = data.shape[0]#len行数，这里表示（x,y,z）个数，也就是序列长度，num表示生成图像的长宽积

    if len < num:
        t = (len - 1) / (num - 1)
        d0 = np.array(range(num))
        d0 = d0 * t

        d0_1 = copy.deepcopy(d0).astype(int)
        d0_0 = d0 - d0_1
        dist = data[1:] - data[:-1]
        d1_1 = data[d0_1]
        d1_0 = dist[d0_1[:-1]]
        d1_0 = d1_0 * d0_0[:-1]
        d1 = copy.deepcopy(d1_1[:-1] + d1_0)
        d1 = np.hstack((d1, data[-1]))

    elif len > num:
        t = (len - 1) / num
        d0 = np.array(range(num + 1))
        d0 = d0 * t

        d0_1 = copy.deepcopy(d0).astype(int)
        list = []
        for i in range(d0_1.shape[0] - 1):
            list.append(np.mean(data[d0_1[i]:d0_1[i + 1] + 1]))
        d1 = np.array(list)

    else:
        d1 = data

    return d1
