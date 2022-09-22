import cv2
import os
import torch
import matplotlib.pyplot as plt

def get_video_duration(filename:str):
    """获得视频长度"""
    cap = cv2.VideoCapture(filename)
    if cap.isOpened():
        rate = cap.get(5)
        frame_num =cap.get(7)
        duration = frame_num/rate
        return duration
    return -1

def get_num_lines(dir):
    """读取文件行数-1，第一行是csv的表头"""
    if os.path.exists(dir):
        count = -1
        for count, line in enumerate(open(dir, 'r')):
            pass
        count += 1
        return count
    else:
        return -1

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

def read_xy_pose(dir):
    ls = []
    with open(dir) as f:
        next(f)
        next(f)
        while True:
            try:
                ls.append(list(map(float,next(f).split(",")[2:])))
            except StopIteration:
                break
    return torch.tensor(ls)

# pose = read_xy_pose("D:\Code\DataSet\gogo\image_eli-m1\data\pos_xy_data.csv")
# pose_pos_xoz_plot(pose)
        
