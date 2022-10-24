import os
import cv2
from get_file_info import get_num_lines,get_video_duration
import pointToWH
from eli_cal_ab import eli_cal_a_b
from utils.constant_func import *

root = "D:\\Code\\DataSet\\gogo"
os.chdir(root)
process_data = ["image_eli-rand99"]

def get_num_frames(dir) -> int:
    return len([file for file in os.listdir(dir) if file[-4:] == ".jpg"])


def log_data(root,vname:str) -> tuple[float,float]:
    """
    返回:长轴方向的像素比，短轴方向的像素比
    """
    os.chdir(root)
    files = os.listdir(root)
    image_files = [file for file in files if file[:6] == "image_"]
    map_speed = {"f":"快速","m":"正常","s":"慢速","rand":"随机速度"}
    dataset = "image_"+vname[:-4]
    if not oe(dataset):
        os.chdir("..")
        print(f"error no {dataset} folder!")
        return -1.0,-1.0
    """生成日志文件"""
    """获得轨迹和速度信息"""
    video_name = dataset.strip("image")[1:]+".mp4"
    info = dataset.strip("image")[1:].split("-")
    trajectory,speed,id = info[0],info[1][0],info[1][1:]
    if speed == "r": #单独处理一下rand的情况
        speed,id = info[1][0:4],info[1][4:]
    print(id)
    os.chdir("vedios")
    if not oe(video_name):
        os.chdir("..")
        print(f"error no {video_name} video!")
        return -1.0,-1.0
    time = get_video_duration(video_name)
    os.chdir("..")
    num_frames = get_num_frames(dataset) #使用jpg的个数计算
    frame_rate = round(num_frames/time)
    
    """给imu.csv文件改名"""
    data_path = os.path.join(dataset,"data")
    imu_file = []
    if os.path.exists(data_path):
        imu_file = [file for file in os.listdir(data_path) if file[:4] == "path"]
    if len(imu_file):
        os.rename(os.path.join(data_path,imu_file[0]),os.path.join(data_path,"imu.csv"))
    num_frames_imu = get_num_lines(os.path.join(dataset,"data","imu.csv"))
    """创建readme.txt"""
    w,h = 1,1
    with open(os.path.join(dataset,"readme.txt"),"w+") as f:
        f.write("-轨迹形状：{}\n".format("椭圆" if trajectory == "eli" else "矩形"))
        trace_len = 18.128 if trajectory == "eli" else 22.8
        f.write("-行走方式：{}(≈ {:.3f}m/s)\n".format(map_speed[speed],trace_len/time))
        f.write(f"-行走时间：{time:.3f}s\n")
        f.write(f"\t视频帧率: {frame_rate}fps\n")
        f.write(f"\t视频帧数: {num_frames} ({num_frames/frame_rate:.3f} s)\n")
        f.write(f"\tIMU数据帧数: {num_frames_imu} ({num_frames_imu/100:.3f} s)\n")
        w,h = eli_cal_a_b(root,video_name)
        pointToWH.log_pixel_rate(f,w,h)
        pointToWH.log_whr(f,w,h)
    return 6.6/w,4.8/h


def rar_data(dir,vname):
    """打包数据"""
    files = os.listdir(dir) 
    image_files = [file for file in files if file[:6] == "image_"]
    dataset = "image_"+vname[:-4]
    compress_ls = []
    compress_ls.append(os.path.join("data"))
    compress_ls.append(os.path.join("readme.txt"))
    compress_ls.append(os.path.join("trajectory.png"))
    src_dir = os.path.join(root,dataset)
    obj_dir = os.path.join(root,"dataset",dataset)
    if not os.path.exists(obj_dir):
        os.mkdir(obj_dir)
    for file in compress_ls:
        if os.path.exists(src_dir+"\\"+file):
            if file == "data" and not os.path.exists(os.path.join(obj_dir,"data")):
                os.mkdir(os.path.join(obj_dir,"data"))
            cmd = "copy {} {}".format(src_dir+"\\"+file,obj_dir+"\\"+file)
            # print(cmd)
            os.system(cmd)


def process_and_transfer(root,vname):
    """process_data直接输入MP4的文件名即可"""
    log_data(root,vname)
    rar_data(root,vname)

if __name__ == "__main__":
    vedio_ls = os.listdir(os.path.join(root,"vedios"))
    vedio_ls = [f for f in vedio_ls if "process" not in f]
    # process_and_transfer(root,vedio_ls)
    vname = "eli-m10.mp4"
    process_and_transfer(root,vname)
# t = get_video_duration(os.path.join(root,"eli-f4.mp4"))
# print(t)