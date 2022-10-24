import sys,os
cur_path = os.path.dirname(__file__)
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)
import numpy as np
from ellipse import LsqEllipse
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from paramaters import PT_TEST_DATA_PATH
from constant_func import *

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


def eli_cal_a_b(root,vname) -> tuple[float,float]:
	root = PT_TEST_DATA_PATH
	# vname = "eli-f2.mp4"
	
	pose_ls = dense_restruction(oj(root,"image_"+vname[:-4],"sparse.nvm.cmvs","00","cameras_v2.txt"))
	pose = np.array([i[1] for i in pose_ls])[:,[0,2]]
	# print(pose)
	X1,X2 = pose[:,0],pose[:,1]
	X = np.array(list(zip(X1, X2)))
	reg = LsqEllipse().fit(X)
	center, width, height, phi = reg.as_parameters()

	print(f'center: {center[0]:.3f}, {center[1]:.3f}')
	print(f'width: {width:.3f}')
	print(f'height: {height:.3f}')
	print(f'phi: {phi:.3f}')

	fig = plt.figure(figsize=(6, 6))
	ax = plt.subplot()
	ax.axis('equal')
	ax.plot(X1, X2, 'ro', zorder=1)
	ellipse = Ellipse(
		xy=center, width=2*width, height=2*height, angle=np.rad2deg(phi),
		edgecolor='b', fc='None', lw=2, label='Fit', zorder=2
	)
	ax.add_patch(ellipse)

	plt.xlabel('$X_1$')
	plt.ylabel('$X_2$')

	plt.legend()
	# plt.show()
	print(f"save fig {vname}")
	plt.savefig(oj(root,"image_"+vname[:-4],"trajectory.png"))
	if width < height:
		width,height = height,width
	return 2*width,2*height

if __name__ == '__main__':
    # avalible in the `example.py` script in this repo
	root = PT_TEST_DATA_PATH
	image_ls = ["eli-f2.mp4"]
	for f in image_ls:
		w,h = eli_cal_a_b(root,f)
		print(w,h)