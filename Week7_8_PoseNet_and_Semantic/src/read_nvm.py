import os
import torch
import numpy as np
import matplotlib.pyplot as plt

dir = "D:\\Code\\DataSet\\gogo\\image_eli-m3\\2.nvm"

def sparse_reconstruction(dir:str) -> list:
    with open(dir) as f:
        for i in range(3):
            next(f)
        pose:str = ""
        pose_ls = []
        while True:
            try:
                pose = next(f).strip("\n")
                if not len(pose):
                    break
                pose_ls.append([pose.split()[0],list(map(float,pose.split()[1:]))])
            except StopIteration:
                break
        return pose_ls

pose_ls = sparse_reconstruction(dir)
pose_ls.sort(key= lambda x:int(x[0].strip("frame").strip(".jpg\n")))
pose_xy = torch.tensor([pose[1] for pose in pose_ls])[:,[5,7]]
pose_xyz = torch.tensor([pose[1] for pose in pose_ls])[:,5:8]
def pose_pos_xoz_plot(pose:torch.Tensor) -> None :
    """
    给定一个姿势的tensor，画出位置的轨迹图在xoz平面上的投影
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)  # 111,几行几列，第几块的数据

    x = pose[:,0].numpy()
    z = pose[:,1].numpy()

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.scatter(x,z,c='r')
    plt.autoscale(True)
    plt.show()
def pose_pos_plot(pose:torch.Tensor) -> None :
    """
    给定一个姿势的tensor，画出位置的轨迹图
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # 111,几行几列，第几块的数据

    x = pose[:,0].numpy()
    y = pose[:,1].numpy()
    z = pose[:,2].numpy()

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    ax.scatter(x,y,z,c='r')
    # plt.autoscale(True)
    plt.ylim(-1,1)
    plt.show()

pose_pos_plot(pose_xyz)
pose_pos_xoz_plot(pose_xy)
