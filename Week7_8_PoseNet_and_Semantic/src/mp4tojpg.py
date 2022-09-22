# 导入所有必要的库
import cv2
import os
from get_file_info import get_video_duration
i = "rect-m" + "2"
i = "eli-s" + "2"
dir = "D:\\Code\\DataSet\\gogo\\"+ i +".mp4"

# 从指定的路径读取视频
cam = cv2.VideoCapture( dir )
t = get_video_duration(dir)
print(t)
try :
     # 创建名为data的文件夹
     os.chdir(f"D:\Code\DataSet\gogo")
     if not os.path.exists( f'image_{i}'):
         os.makedirs( f'image_{i}' )
  
# 如果未创建，则引发错误
except OSError:
     print ( 'Error: Creating directory of data' )
  
# frame
currentframe = 0
  
while ( True ):
      
     # reading from frame
     ret, frame = cam.read()
  
     if ret:
         # 如果视频仍然存在，继续创建图像
         name = f'./image_{i}/frame' + str (currentframe) + '.jpg'
         if currentframe % 1 == 0:
            print ( 'Creating...' + name)
            # 写入提取的图像
            cv2.imwrite(name, frame)
         # 增加计数器，以便显示创建了多少帧
         currentframe += 1
     else :
         break
  
# 一旦完成释放所有的空间和窗口
cam.release()
cv2.destroyAllWindows()
