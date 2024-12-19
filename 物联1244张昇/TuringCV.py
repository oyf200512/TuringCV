#引入库
import cv2
import numpy as np
import random


#定义一个读取图片的函数
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#创建画布
canvas = np.ones((800, 800, 3), dtype=np.uint8) * 255
#绘图
#圆
cv2.circle(canvas, (100, 100), 50, (0, 0, 255), -1)
cv2.circle(canvas,(300,100),60,(255,0,0),-1)
cv2.circle(canvas,(500,100),75,(100,50,65),-1)
#矩形
cv2.rectangle(canvas, (200, 200), (300, 300), (200, 55, 100), -1)
cv2.rectangle(canvas, (400, 200), (500, 400), (200, 1505, 0), -1)
cv2.rectangle(canvas, (600,200), (750, 350), (20, 110, 150), -1)
cv2.rectangle(canvas, (50,350), (300,500), (65,89,12), -1)
#椭圆
cv2.ellipse(canvas,(500,600),(100,50),0,0,360,(0,128,128),-1)

#不规则图形
cv2.ellipse(canvas,(100,600),(200,100),0,0,120,(128, 128, 128),-1)

#多边形（几个点就是几变形形）
pts = np.array([[[800,700], [700,600], [600,700]]], dtype=np.int32)


#连线
cv2.polylines(canvas, pts, True, (170,130,255), 5)
cv2.fillPoly(canvas, pts, (170,130,255))
cv_show("canvas",canvas)


#边缘检测
edges = cv2.Canny(canvas,100,200)
contours,hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv_show("edges",edges)
#因为contours会将所检测的轮廓计入列表，所以直接用len读取
num_shapes = len(contours)


#标记轮廓
cv2.drawContours(canvas,contours,-1,(0,0,0),1)#用了很多颜色画图，所以用黑线标
cv_show("canvas",canvas)


#选模板
if len(contours) > 0:
    random_index = random.randint(0, len(contours) - 1)
    x, y, w, h = cv2.boundingRect(contours[random_index])
    template = canvas[y-5:y + h +5, x-5:x + w+5]

else:
    print("没有检测到图形轮廓")
cv_show("template",template)


#模板匹配
result = cv2.matchTemplate(canvas,template,cv2.TM_CCOEFF_NORMED)


#定位标注
loc = np.where(result > 0.8)        #选出result数组中大于的元素位置
for pt in zip(*loc[::-1]):
    bottom_right = (pt[0] + w, pt[1] + h)
    cv2.rectangle(canvas,pt,bottom_right,(0,0,0),2)


#展示
cv_show("canvas",canvas)

print(num_shapes)