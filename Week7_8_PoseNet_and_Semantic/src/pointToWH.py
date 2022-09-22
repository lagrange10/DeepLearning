import math
import os



# x1,y1,x2,y2,x3,y3,x4,y4 = map(float,input().split())
# dir = ""

def calc_wh_and_tofile(x1,y1,x2,y2,x3,y3,dir):
    """给定矩形3个顶点"""
    d1,d2,d3 = math.dist((x1,y1),(x2,y2)),\
                math.dist((x1,y1),(x3,y3)),math.dist((x3,y3),(x2,y2))
    d1,d2,d3 = sorted((d1,d2,d3))
    print(d1,d2,d3)
    print("标注宽高比: {:.3f}".format(d2/d1),"\n真实宽高比:",6.6/4.8,\
        "\n误差：{:.3f}".format(abs(d2/d1-6.6/4.8)/1.375))
    os.chdir(dir)
    with open("acc_info.txt","w+") as f:
        f.write("标注宽高比: {:.3f}".format(d2/d1)+"\n真实宽高比:"+str(6.6/4.8)+\
        "\n误差：{:.3f}".format(abs(d2/d1-6.6/4.8)/1.375))

def calc_wh_and_tofile_eli(x1,y1,x2,y2,x3,y3,x4,y4,dir):
    """给定椭圆4个顶点"""
    d1,d2 = math.dist((x1,y1),(x2,y2)),\
                math.dist((x3,y3),(x4,y4))
    print(d1,d2)
    print("标注宽高比: {:.3f}".format(d1/d2),"\n真实宽高比:",6.6/4.8,\
        "\n误差：{:.3f}".format(abs(d1/d2-6.6/4.8)/1.375))
    os.chdir(dir)
    with open("acc_info.txt","w+") as f:
        f.write("标注宽高比: {:.3f}".format(d1/d2)+"\n真实宽高比:"+str(6.6/4.8)+\
        "\n误差：{:.3f}".format(abs(d1/d2-1.375)/1.375))

def calc_wh(dir):
    x1,y1,x2,y2,x3,y3,x4,y4 = map(float,input().split())
    if abs(x4) <= 1e-4:
        calc_wh_and_tofile(x1,y1,x2,y2,x3,y3,dir)
    else:
        calc_wh_and_tofile_eli(x1,y1,x2,y2,x3,y3,x4,y4,dir)

def calc_whr(f):
    w,h = map(int,input().split())
    f.write("\n\n标注宽高比: {:.3f}".format(w/h)+"\n真实宽高比:"+str(6.6/4.8)+\
    "\n误差：{:.3f}".format(abs(w/h-1.375)/1.375))

def calc_pixel_rate(f):
    x1,y1,x2,y2 = map(float,input().split())
    d = math.dist((x1,y1),(x2,y2))
    D = 6.6
    f.write(f"\n-像素比: 1(像素) : {D/d:.3f}m")

# dir = "D:\Code\DataSet\gogo\dataset\image_eli-f2\\readme.txt"
# calc_pixel_rate(dir)
# calc_whr(dir)
# calc_wh(dir)