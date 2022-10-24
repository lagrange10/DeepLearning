from logging import root
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from file_op import get_num_frames
from utils.constant_func import *
oj = os.path.join
rootdir = "D:\\Code\\DataSet\\gogo"
vname = "eli-rand99.mp4"
dir_sparse = oj(rootdir,vname,"1.nvm")




def dense_restruction(dir:str) -> list:
    """读取数据，读入的格式为(文件名，姿势的xyzpqrs)"""
    pose_ls = []
    with open(dir) as f:
        for i in range(18):
            next(f)
        pose:str = "123"
        while True:
            try:
                fname = next(f)
                origin_fname = next(f)
                origin_fname = origin_fname.split("\\")[-1]
                for i in range(3):
                    next(f)
                pose = next(f).strip("/n")
                next(f)
                pose += (" " + next(f).strip("/n"))
                for i in range(6):
                    next(f)
                pose_ls.append((origin_fname.strip("\n"),list(map(float,pose.split()))))
            except StopIteration:
                break
        pose_ls.sort(key= lambda x:int(x[0].strip("frame").strip(".jpg\n")))
        return pose_ls

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
        pose_ls.sort(key= lambda x:int(x[0].strip("frame").strip(".jpg\n")))
        return pose_ls

def save_dense_restuction(rootdir,vname) -> None:
    vname = "image_" + vname[:-4]
    """把密集重建结果保存在data"""
    workspace = oj(rootdir,vname)
    dir_dense = oj(rootdir,vname,"sparse.nvm.cmvs\\00\\cameras_v2.txt")
    try:
        open(dir_dense)
    except FileNotFoundError:
        try:
            f = [i for i in ls(workspace) if ".nvm.cmvs" in i][0]
        except:
            return
        f = oj(workspace,f)
        cmd = f"move {f} {workspace}/sparse.nvm.cmvs"
        print(cmd)
        os.system(cmd)

    pose_ls = dense_restruction(dir_dense)
    TraceData = torch.tensor([pose[1] for pose in pose_ls])
    # TraceData = torch.tensor([pose[1] for pose in pose_ls if pose[1][0]<15 and pose[1][1]>-10])
    fname_ls = [pose[0].strip("\n") for pose in pose_ls] #\n来自文件    image_idx = [int(f.strip("frame").strip(".jpg")) for f in fname_ls]
    # print(fname_ls[0:10],TraceData[0:10,0:3])
    os.chdir(workspace)

    num_frames = get_num_frames(workspace)
    if not os.path.exists( f'data'):
        os.makedirs( f'data' )
    os.chdir('data')

    
    outls = []
    for i in range(len(pose_ls)):
        pose = [pose_ls[i][0].strip("\n")]+list(map(str,[pose[1] for pose in pose_ls][i]))
        outls.append(pose)
    if len(outls) < num_frames:
        """有缺失的帧"""
        print("output:",len(outls),"input:",num_frames)
        print("有缺失的帧")
    out = pd.DataFrame(outls)
    out.to_csv("posedata.csv")

    with open('posedata.txt',"w+") as f:
        f.write("# pose vec-7 xyz-pqrs\n\n")
        for i in range(len(pose_ls)):
            pose = [pose_ls[i][0].strip("\n")]+list(map(str,[pose[1] for pose in pose_ls][i]))
            f.write(" ".join(pose)+"\n")

    outls = []
    for i in range(len(pose_ls)):
        # 考虑插值
        # pose_x_np = TraceData[:,[0,2]].numpy()[:,0]
        # pose_y_np = TraceData[:,[0,2]].numpy()[:,1]
        pose_xz = [pose_ls[i][0].strip("\n")] + list(map(str,TraceData[:,[0,2]].tolist()[i]))
        outls.append(pose_xz)
    out = pd.DataFrame(outls,columns=["fname","posX","posY"])
    out.to_csv("pos_xy_data.csv")

    with open('pos_xy_data.txt',"w+") as f:
        f.write("# position vec-2 xy\n\n")
        for i in range(len(pose_ls)):
            pose_xz = [pose_ls[i][0].strip("\n")] + list(map(str,TraceData[:,[0,2]].tolist()[i]))
            f.write(" ".join(pose_xz)+"\n")
    os.chdir(rootdir)

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
    plt.ylim(-10,10)
    plt.show()

def pose_pos_xoz_plot(pose:torch.Tensor) -> None :
    """
    给定一个姿势的tensor，画出位置的轨迹图在xoz平面上的投影
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)  # 111,几行几列，第几块的数据

    x = pose[:,0].numpy()
    z = pose[:,2].numpy()

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.scatter(x,z,c='r')
    plt.autoscale(True)
    plt.show()

# pose_pos_plot(TraceData)
# pose_pos_xoz_plot(TraceData)


def read_data_plot(fname):
    with open(fname,"r") as f:
        next(f)
        next(f)
        pose_list = []
        while True:
            try:
                """line——— fname pose-vec"""
                line = next(f).strip("\n").split()
                pose_list.append(list(map(float,line[1:])))
            except StopIteration:
                break
        fig = plt.figure()
        ax = fig.add_subplot(111)  # 111,几行几列，第几块的数据
        pose_tensor = torch.tensor(pose_list)
        x = pose_tensor[:,0].numpy()
        z = pose_tensor[:,1].numpy()
        ax.set_xlabel('X Label')
        ax.set_ylabel('Z Label')
        ax.scatter(x,z,c='r')
        plt.autoscale(True)
        plt.show()

if __name__ == "__main__":
    save_dense_restuction(rootdir,vname="eli-rand100.mp4")