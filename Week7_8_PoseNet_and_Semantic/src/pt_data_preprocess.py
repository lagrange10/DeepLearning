import sys
# sys.path.append("/home/koumengya/桌面/IP/AppServer_kmy")

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils.discrete_scale import discrete_scale

oj,ls = os.path.join,os.listdir

def data_preprocess(root,vname,pixel_a,pixel_b):
    """
    传入的是长轴的宽高比，和短轴的宽高比
    """
    aver_pixel_rate = (pixel_a+pixel_b)/2
    vdata = "image_" + vname[:-4]
    dir = oj(root,vdata,"data")
    os.chdir(dir)
    f_ls = ls(".")
    for f in f_ls:
        if vname[:-4] in f and len(f)>len(vname):
            os.rename(f,"imu.csv")
    imu_name = "imu.csv"
    imu_file = pd.read_csv(imu_name, dtype=str)
    acc_x = imu_file[[col for col in imu_file.columns if 'acc_x' in col]].values.astype(float).squeeze()
    acc_y = imu_file[[col for col in imu_file.columns if 'acc_y' in col]].values.astype(float).squeeze()
    acc_z = imu_file[[col for col in imu_file.columns if 'acc_z' in col]].values.astype(float).squeeze()
    gyro_x = imu_file[[col for col in imu_file.columns if 'gyro_x' in col]].values.astype(float).squeeze()
    gyro_y = imu_file[[col for col in imu_file.columns if 'gyro_y' in col]].values.astype(float).squeeze()
    gyro_z = imu_file[[col for col in imu_file.columns if 'gyro_z' in col]].values.astype(float).squeeze()
    mag_x = imu_file[[col for col in imu_file.columns if ('mag_x' in col) and ('r' not in col)]].values.astype(float).squeeze()
    mag_y = imu_file[[col for col in imu_file.columns if ('mag_y' in col) and ('r' not in col)]].values.astype(float).squeeze()
    mag_z = imu_file[[col for col in imu_file.columns if ('mag_z' in col) and ('r' not in col)]].values.astype(float).squeeze()
    timestamp = imu_file[[col for col in imu_file.columns if 'timestamp' in col]].values.squeeze()
    l = len(mag_x)
    print('len:', l)

    file_name = "pos_xy_data.csv"
    pos_file = pd.read_csv(file_name, dtype=str)
    pos_file.rename(columns={"1":'posX',"2":'posY'},inplace=True)
    pos_x = pos_file[[col for col in pos_file.columns if 'posX' in col]].values.astype(float).squeeze()  # [len,]
    pos_y = pos_file[[col for col in pos_file.columns if 'posY' in col]].values.astype(float).squeeze()  # [len,]
    pos_x_inter = discrete_scale(pos_x, l)
    pos_y_inter = discrete_scale(pos_y, l)
    pos_x_true = pos_x_inter * aver_pixel_rate #真实比例下
    pos_y_true = pos_y_inter * aver_pixel_rate

    L = [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z, pos_x_true, pos_y_true, timestamp]  # [2, len]
    L = list(map(list, zip(*L)))  # 转置
    column = ['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ', 'magX', 'magY', 'magZ', 'posX', 'posY',"timestamp"]
    test = pd.DataFrame(columns=column, data=L)
    _vname = vname.strip(".mp4")
    test.to_csv(oj(dir,f"{_vname}.csv"))


    index = np.arange(len(pos_x_inter))
    plt.plot(index, pos_x_inter)
    # plt.show()
    plt.plot(index, pos_y_inter)
    # plt.show()
    plt.plot(pos_x_inter, pos_y_inter)
    # plt.show()
    plt.plot(pos_x, pos_y)
    # plt.show()
    os.chdir(root)

if __name__ == "__main__":
    root = "D:\\Code\\DataSet\\gogo"
    vedio_dir = "vedios"
    vedio_ls = os.listdir(oj(root,vedio_dir))
    vedio_data_ls = ["image_"+v.strip(".mp4") for v in vedio_ls]
    root = oj(root,"dataset")
    vedio_data_ls = ls(root)
    vedio_ls = [v[6:]+".mp4" for v in vedio_data_ls if ".rar" not in v]
    for vname in vedio_ls:
        data_preprocess(root,vname,2,2) #随便填的像素比