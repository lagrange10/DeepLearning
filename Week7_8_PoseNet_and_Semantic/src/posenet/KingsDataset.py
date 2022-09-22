import os
from matplotlib.colors import Normalize
from matplotlib.pyplot import show
from torchvision import transforms as T
import torch
import torch.utils.data
import torchvision
from d2l import torch as d2l
from models import PoseNet
import numpy as np


os.chdir("D:\Code\DeepLearning\Week7_8_PoseNet_and_Semantic\src\posenet\models")
kings_dir = "..\..\..\data\KingsCollege\KingsCollege"

class KingsDataset(torch.utils.data.Dataset):

    def __init__(self,is_train=True,resize=256,crop_size=224) -> None:
        super().__init__()
        self.train = is_train
        self.resize = resize
        self.crop_size = crop_size

        self.image_path = []
        self.image = [] #features
        self.pose = [] #labels

        self.resize_transforms = T.Resize(size=(256,455))
        self.read_images_and_labels(kings_dir)
        self.get_mean_image()
    
    def get_mean_image(self) -> None:
        self.mean_img = torch.zeros(3,256,455,dtype=torch.float32)
        for img in self.image:
            img = self.resize_transforms(img)
            self.mean_img += img
        self.mean_img /= len(self.image)

    def read_images_and_labels(self,kings_dir):
        '''把图片和标注读进image和pose里面'''
        txt_fname = os.path.join(kings_dir,'dataset_train.txt' if self.train 
                                            else 'dataset_test.txt') #读label所在的txt
        
        """读出txt中的文件名及label"""
        with open(txt_fname,'r') as f:
            next(f)
            next(f)
            next(f)
            n = 0
            for line in f:
                #p0-p2是位置，p3-p6是方向
                fname,p0,p1,p2,p3,p4,p5,p6 = line.split()
                self.image_path.append(fname)
                self.pose.append(torch.tensor(list(map(float,[p0,p1,p2,p3,p4,p5,p6]))))
                n+=1
                if n>50:
                    break
        
        """读出png的图片"""
        mode = torchvision.io.image.ImageReadMode.RGB
        for i,fname in enumerate(self.image_path):
            # img = torchvision.io.read_image()
            self.image.append( torchvision.io.read_image(
                                os.path.join(kings_dir,fname),mode) )

    def __getitem__(self, index) -> torch.Tensor:
        """获得特征和标注"""
        data:torch.Tensor
        data,label = self.image[index],self.pose[index]
        # 预处理
        data = self.resize_transforms(data)
        data = data.float()
        data -= self.mean_img
        # 数据增强
        
        transform = T.Compose([
            T.RandomCrop(self.crop_size),
        ])
        data = transform(data)

        return data,label

    def __len__(self):
        return len(self.image_path)

    #def load_data()




# data = KingsDataset()

# train_loader = torch.utils.data.DataLoader(dataset=data,batch_size=2,shuffle=True)
# i = 0
# for img,label in train_loader:
#     i+=1
#     if i>5:
#         break
#     imgs = [*img]
#     imgs = [img.permute(1,2,0) for img in imgs]
#     d2l.show_images(imgs,1,2)
# """打印出前几张图片"""
# n = 5
# imgs = data.image[0:n]
# imgs = [img.permute(1,2,0) for img in imgs]
# d2l.show_images(imgs,1,n)
# print(data.pose)

show()
