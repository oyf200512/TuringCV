#导入三方库
import cv2
import numpy as np
import random




def cv2_show(name,img): #图像展示
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def fill_image(image,pts,color):#多边形的绘制与填充
    cv2.polylines(image, [pts], True, color, 1)
    cv2.fillPoly(image, [pts], color)

def sort_contours(cnts): #对轮廓按照面积排序
    boundingboxes=[cv2.boundingRect(i) for i in cnts]
    (cnts,boundingboxes)=zip(*sorted(zip(cnts,boundingboxes),key=lambda b:b[1][2]*b[1][3]))
    return cnts







"""----------------------图形绘制部分---------------------------------"""
#创建空白画布
shapes=np.ones((800,800,3),dtype=np.uint8) #黑色画布
shapes*=255 #白色画布
cv2_show("shapes",shapes)

#绘制矩形
cv2.rectangle(shapes,(50,50),(150,150),(0,0,255),-1)
cv2.rectangle(shapes,(225,50),(375,100),(0,0,255),-1)

#自定义函数绘制三角形
def triangle(canvas,x1,x2,x3):
    cv2.line(canvas,x1,x2,(0,0,255),2)
    cv2.line(canvas,x2,x3,(0,0,255),2)
    cv2.line(canvas,x1,x3,(0,0,255),2)
    #填充图形
    pts=np.array([x1,x2,x3],np.int32)
    cv2.fillPoly(shapes,[pts],(0,0,255))

triangle(shapes,(450,50),(550,50),(550,150))
triangle(shapes,(650,50),(750,100),(700,150))

#绘制圆形
cv2.circle(shapes,(100,300),50,(0,255,0),-1)

#绘制椭圆
cv2.ellipse(shapes,(300,300),(75,50),0,0,360,(0,255,0),-1)

#绘制多边形
points=np.array([(500,250),(560,300),(525,350),(475,350),(440,300)])
fill_image(shapes,points,(0,255,0))

points=np.array([(660,250),(740,250),(775,300),(740,350),(660,350),(625,300)])
fill_image(shapes,points,(0,255,0))

#绘制梯形
points=np.array([(75,475),(125,475),(150,525),(50,525)])
fill_image(shapes,points,(255,0,0))

#绘制扇形
cv2.ellipse(shapes,(300,500),(75,75),0,0,-120,(255,0,0),-1)

#绘制平行四边形
points=np.array([(500,450),(550,450),(500,550),(450,550)])
fill_image(shapes,points,(255,0,0))

#绘制半圆
cv2.ellipse(shapes,(700,500),(75,75),0,0,-180,(255,0,0),-1)

cv2_show("shapes",shapes)

#图像保存
cv2.imwrite("shapes.png",shapes)








"""----------------------图形处理部分---------------------------------"""

# 图像预处理
shapes_gray=cv2.cvtColor(shapes,cv2.COLOR_BGR2GRAY) #灰度图
ret,shapes_binary=cv2.threshold(shapes_gray,150,255,cv2.THRESH_BINARY) #二值图
cv2_show("shapes_gray",shapes_gray)
cv2_show("shapes_binary",shapes_binary)

#canny边缘检测
edges=cv2.Canny(shapes_gray,127,255)
cv2_show("edges",edges)
cv2.imwrite("edges.png",edges)

#轮廓检测
contours,hierarchy=cv2.findContours(shapes_binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #tree检索所有轮廓
shapes_copy=shapes.copy() #先对原图复制 便于下面绘制轮廓

#轮廓按照面积排序
contours_sorted=sort_contours(contours)

#轮廓筛选
contours=contours_sorted[:-1]
print("一共有{}个图形".format(len(contours)))
cv2.drawContours(shapes_copy,contours,-1,(128,0,128),3) #紫色绘制轮廓
cv2_show("shapes_copy",shapes_copy)
cv2.imwrite("contours.png",shapes_copy)







"""------------------------模板匹配部分-----------------------"""
#随机选择模板图

template_number=random.randint(1,12) #随机选择一个轮廓序号
template_contours=contours[template_number-1]
x,y,w,h=cv2.boundingRect(template_contours) #对该轮廓获得外界矩形
template=shapes[y:y+h,x:x+w]
cv2_show("template",template)
cv2.imwrite("template.png",template)

#截取原图
shape=[] #将原图中每个图形截取保存与shape
for each in contours:
    x,y,w,h=cv2.boundingRect(each)
    shape.append(shapes[y:y+h,x:x+w])

#开始模板匹配
scores=[] #将每个图形与模板图的相关性保存到scores
for each in shape:
    h,w,_=each.shape
    template_resize=cv2.resize(template,(w,h)) #将模板resize称每个图形等大 便于计算相关
    score=cv2.matchTemplate(each,template_resize,cv2.TM_CCOEFF_NORMED) #对结果归一化
    _,max_val,_,_=cv2.minMaxLoc(score)
    scores.append(max_val)

result=contours[scores.index(max(scores))] #选取归一后最大 最接近1的
x,y,w,h=cv2.boundingRect(result)
shapes_copy2=shapes.copy()
matching=cv2.rectangle(shapes_copy2,(x,y),(x+w,y+h),(0,0,0),3) #在原图绘制匹配结果
cv2_show("matching",matching)
cv2.imwrite("matching.png",matching)
