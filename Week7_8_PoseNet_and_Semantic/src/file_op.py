import os
import cv2
from get_file_info import get_num_lines,get_video_duration
import pointToWH

root = "D:\\Code\\DataSet\\gogo"
os.chdir(root)
success_data = ["image_eli-f2","image_eli-f3","image_eli-f4","image_eli-f5"
                "image_eli-m1","image_eli-m2","image_eli-m3",
                "image_eli-rand1","image_eli-rand3","image_eli-rand5",
                "image_eli-s3","image_eli-s4"
                "image_rect-f4","image_rect-f3","image_rect-f2","image_rect-m1"
                
                ]
success_data = []
process_data = ["image_eli-m2"]

def dfs(path):
    files = os.listdir(path)
    image_files = [file for file in files if file[:6] == "image_"]
    map_speed = {"f":"快速","m":"正常","s":"慢速","rand":"随机速度"}
    
    """对每个文件处理后打包"""
    for dataset in image_files:
        if dataset in success_data: #成功就不需要了
            continue
        if dataset not in process_data: #成功就不需要了
            continue
        """获得轨迹和速度信息"""
        video_name = dataset.strip("image")[1:]+".mp4"
        info = dataset.strip("image")[1:].split("-")
        trajectory,speed,id = info[0],info[1][0],info[1][1:]
        if speed == "r": #单独处理一下rand的情况
            speed,id = info[1][0:4],info[1][4:]
        time = get_video_duration(video_name)
        num_frames = len([file for file in os.listdir(dataset) if file[-4:] == ".jpg"])
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
        with open(os.path.join(dataset,"readme.txt"),"w+") as f:
            f.write("-轨迹形状：{}\n".format("椭圆" if trajectory == "eli" else "矩形"))
            trace_len = 18.128 if trajectory == "eli" else 22.8
            f.write("-行走方式：{}(≈ {:.3f}m/s)\n".format(map_speed[speed],trace_len/time))
            f.write(f"-行走时间：{time:.3f}s\n")
            f.write(f"\t视频帧率: {frame_rate}fps\n")
            f.write(f"\t视频帧数: {num_frames} ({num_frames/frame_rate:.3f} s)\n")
            f.write(f"\tIMU数据帧数: {num_frames_imu} ({num_frames_imu/100:.3f} s)\n")
            pointToWH.calc_pixel_rate(f)
            pointToWH.calc_whr(f)
            
dfs(root)

def rar_data(dir):
    """打包数据"""
    files = os.listdir(dir) 
    image_files = [file for file in files if file[:6] == "image_"]
    for dataset in image_files:
        if dataset in success_data:
            continue
        if dataset not in process_data:
            continue
        compress_ls = []
        compress_ls.append(os.path.join("data"))
        compress_ls.append(os.path.join("readme.txt"))
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


rar_data(root)

# t = get_video_duration(os.path.join(root,"eli-f4.mp4"))
# print(t)