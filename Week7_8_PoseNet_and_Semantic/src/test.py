from logging import root
from autosfm import log_process,data_preprocess,copy_imu
from paramaters import PT_TEST_DATA_PATH
import os
root = PT_TEST_DATA_PATH
vedio_ls = ["eli-f10.mp4"]
vedio_ls = os.listdir(os.path.join(root,"vedios"))
vedio_ls = [f for f in vedio_ls if "process" not in f]
for vedio in vedio_ls:
    copy_imu(root,vedio)
    data_preprocess(root,vedio)
log_process(root,vedio_ls)