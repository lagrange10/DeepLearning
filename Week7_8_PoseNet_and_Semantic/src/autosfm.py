import os
from time import sleep
from mp4tojpg import mp4_to_jpg
import readSfMData
from file_op import log_data,rar_data
from pt_data_preprocess import data_preprocess
from utils.constant_func import *

def vname_to_vdata(vname:str) -> str:
    return "image_"+vname[:-4]

def copy_imu(root,vname):
    os.chdir(root)
    if not oe(oj(root,"imu_data")):
        os.mkdir(oj(root,"imu_data"))
    os.chdir(oj(root,"imu_data"))

    f_ls = ls(".")
    for f in f_ls:
        if vname[:-4] in f:
            cmd = f"move \"{f}\" ../{vname_to_vdata(vname)}/data"
            print(cmd)
            os.system(cmd)
            break
    os.chdir(root)

def autosfm(root,vedio_ls,space=1):
    for vedio in vedio_ls:
        mp4_to_jpg(root,vedio,space=space)
        workspace = oj(root,"image_"+vedio[:-4])
        os.chdir(workspace)
        cmd = "VisualSFM sfm+pmvs . sparse.nvm  dense.nvm"
        os.system(cmd)
        readSfMData.save_dense_restuction(root,vedio)
        copy_imu(root,vedio)
        pixel_a,pixel_b = log_data(root,vedio)
        data_preprocess(root,vedio,pixel_a,pixel_b)
        rar_data(root,vedio)

def data_after_sfm(root,vedio_ls):
    for vedio in vedio_ls:
        readSfMData.save_dense_restuction(root,vedio)
        copy_imu(root,vedio)
        pixel_a,pixel_b = log_data(root,vedio)
        data_preprocess(root,vedio,pixel_a,pixel_b)
        rar_data(root,vedio)


if __name__ == "__main__":
    root = "D:\\Code\\DataSet\\gogo"
    vedio_dir = "vedios"
    os.chdir(root)
    if not oe(vedio_dir):
        os.mkdir(vedio_dir)
    vedio_ls = os.listdir(oj(vedio_dir))
    vedio_ls = [v for v in vedio_ls if "processed" not in v]
    vedio_data_ls = ["image_"+v[:-4] for v in vedio_ls]
    print(vedio_ls)

    # vedio_ls = ["eli-rand100.mp4"]
    autosfm(root,vedio_ls,space=1)
    # data_after_sfm(root,vedio_ls)