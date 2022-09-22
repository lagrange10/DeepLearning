import os

dir = "D:\Code\DataSet\gogo\image15"
os.chdir(dir)
for i in range(500,550):
    os.system(f"del frame{i}.jpg")
    os.system(f"del frame{i}.sift")
    os.system(f"del frame{i}.mat")