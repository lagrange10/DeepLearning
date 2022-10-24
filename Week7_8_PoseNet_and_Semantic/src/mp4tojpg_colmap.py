# 导入所有必要的库
import cv2
import os
from get_file_info import get_video_duration

vname = "eli-rand3.mp4"
root = "D:\\Code\\DataSet\\gogo\\"


def mp4tojpg(root,vedio,space):
    """
    视频和生成的文件夹默认在同一目录root
    生成的目录目前写死在root/colmaptest
    """
    os.chdir(root)
    newdir_name = f'image_{vedio[:-4]}_{space}'
    # 从指定的路径读取视频
    cam = cv2.VideoCapture(vedio)
    t = get_video_duration(vedio)
    print(t)
    try :
        # 创建名为data的文件夹
        os.chdir(f"colmaptest")
        if not os.path.exists(newdir_name):
            os.makedirs(newdir_name)
        os.chdir(newdir_name)
        if not os.path.exists( 'images'):
            os.makedirs( 'images' )

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
            name = f'./images/frame' + str (currentframe) + '.jpg'
            if currentframe % space == 0:
                print ( 'Creating...' + root + name)
                # 写入提取的图像
                cv2.imwrite(name, frame)
            # 增加计数器，以便显示创建了多少帧
            currentframe += 1
        else:
            break

    # 一旦完成释放所有的空间和窗口
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    mp4tojpg(root,vname,space=10)