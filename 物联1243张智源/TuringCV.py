import cv2 as cv
import numpy as np

maps = np.ones((800,800,3),np.uint8)
maps = maps*255      #白色画布


cv.rectangle(maps,(20,5),(70,100),(255,0,0),1)  #绘制矩形

cv.ellipse(maps,(400,400),(30,40),0,0,360,(0,0,255),1)   #椭圆

cv.ellipse(maps,(600,600),(100,50),45,0,360,(0,255,0),1)   #椭圆

cv.circle(maps,(600,200),70,(0,255,0),1)   #圆

pts1 = np.array([(200,500),(400,500),(300,700)])
cv.polylines(maps,[pts1],True,(0,0,255),1)   #等腰三角形

pts2 = np.array([(400,100),(300,100),(200,300),(500,300)])
cv.polylines(maps,[pts2],True,(255,0,0),1)   #等腰梯形

cv.rectangle(maps,(330,120),(380,270),(0,0,255),1)   #矩形

cv.circle(maps,(100,400),100,(255,0,0),1)  #圆

pts3 = np.array([(750,400),(750,500),(550,500),(480,400)])
cv.polylines(maps,[pts3],True,(0,255,0),1)   #等腰梯形

pts4 = np.array([(100,50),(100,270),(200,270)])
cv.polylines(maps,[pts4],True,(0,0,255),1)    #直角三角形

cv.rectangle(maps,(50,580),(230,770),(255,0,0),1)  #矩形

cv.rectangle(maps,(400,580),(480,770),(255,0,0),1)   #矩形

cv.circle(maps,(700,80),80,(0,0,255),1)  #圆

#拷贝原图
map1 = maps.copy()
map2 = maps.copy()

#用拉普拉斯滤波边缘检测
edges = cv.Laplacian(maps,-1)

#转换为灰色单通道图像
gray = cv.cvtColor(maps,cv.COLOR_BGR2GRAY)

#图像二值化
ret,new = cv.threshold(gray,240,255,cv.THRESH_BINARY)

#找出轮廓
contours,hierarchy = cv.findContours(new,3,cv.CHAIN_APPROX_SIMPLE)

#计算图形轮廓面积，过滤掉小的噪点的轮廓
count = 0
for i in contours:
    area = cv.contourArea(i)
    if area > 100:
        count += 1

#打印图形数量
print(f"图形数量为:{count-1}")    #去掉最外面的的800*800大轮廓

#在原图上标注轮廓
cv.drawContours(maps,contours,-1,(0,0,255),2)

#读取模板
template = cv.imread("./template.png",0)   #读取灰度图像
h,w = template.shape[:2]   #取模板的高和宽

#模板匹配
result = cv.matchTemplate(gray,template,cv.TM_CCOEFF_NORMED)

threshold = 0.2  #设置阈值
loc = np.where(result>=threshold)   #生成阈值大于result的最大值的坐标
for pt in zip(*loc[::-1]):
    bottom_right = (pt[0]+w,pt[1]+h)
    cv.rectangle(map2,pt,bottom_right,(0,0,),2)  #画出匹配成功的图像


cv.imshow("new",new)
cv.imshow("maps",maps)
cv.imshow("edges",edges)
cv.imshow("map1",map1)
cv.imshow("template",template)
cv.imshow("matching",map2)

cv.imwrite("shape.png",map1)
cv.imwrite("contours.png", maps)
cv.imwrite("edges.png",edges)
cv.imwrite("template.png",template)
cv.imwrite("matching.png",map2)

k = cv.waitKey(0)
if k == 27:
    cv.destroyAllWindows()
