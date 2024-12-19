import cv2
import numpy as np

### 1.几何图形绘制
# 创建800*800白色画布
canvas = np.ones((800,800,3),np.uint8)*255

# 圆形
cv2.circle(canvas, (100, 100), 50, (0, 0, 255), -1) # 红色实心圆
cv2.circle(canvas, (700, 100), 40, (200, 0, 0), 2) # 蓝色空心圆

# 矩形
cv2.rectangle(canvas, (50, 200), (150, 300), (0, 200, 0), -1) # 绿色实心矩形
cv2.rectangle(canvas, (650, 200), (750, 300), (255, 0, 0), 2) # 蓝色空心矩形

# 椭圆
cv2.ellipse(canvas, (200, 500), (50, 80), 30, 0, 360, (80, 150, 100), -1)  # 墨绿色实心椭圆
cv2.ellipse(canvas, (600, 500), (80, 40), 0, 0, 360, (255, 0, 255), 2) # 品红色空心椭圆

# 三角形
pts1 = np.array([[300, 300], [350, 250], [400, 300]], np.int32)
cv2.polylines(canvas, [pts1], True, (255, 128, 0), 2) # 橙色三角形边框

pts2 = np.array([[500, 300], [550, 250], [600, 300]], np.int32)
cv2.fillPoly(canvas, [pts2], (128, 0, 128)) # 紫色实心三角形

# 沙漏形
pts3 = np.array([[100,600],[100,700],[300,600],[300,700]])
cv2.polylines(canvas,[pts3],True,(96, 182, 255),2) # 橙色空心沙漏形

pts4 = np.array([[500,600],[500,700],[700,600],[700,700]])
cv2.polylines(canvas,[pts4],True,(211, 174, 255),2) # 粉色空心沙漏形状
# 这里换成实心沙漏的话在后面的轮廓检测会出现奇怪的bug QAQ


### 2.边缘检测与计数
canvas_contour = canvas.copy()
# 边缘检测
grey = cv2.cvtColor(canvas,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(grey,100,200)

# 轮廓检测
contours,hierarchy = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(canvas_contour,contours,-1,(240,207,137),2)
print(f"检测到的图形数量为：{len(contours)}")


### 3.模版检测
# 这里选取紫色实心三角形作为模版
template = canvas[250:300,500:600].copy()

# 查找图形
result = cv2.matchTemplate(canvas,template,cv2.TM_CCOEFF_NORMED)
min_val,max_val,min_loc,max_loc =cv2.minMaxLoc(result)

mark_canvas = canvas.copy() # 复制一个画布用来标记图形
h,w = template.shape[:2] # 获取标记图像的宽和高
cv2.rectangle(mark_canvas,max_loc,(max_loc[0]+w,max_loc[1]+h),(0,220,0),2)
cv2.putText(mark_canvas,"triangle",(max_loc[0],max_loc[1]+h+20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)


# 展示结果
cv2.imshow("mark",mark_canvas)
cv2.imshow("contours",canvas_contour)
cv2.imshow("edges",edges)
cv2.imshow("canvas_original",canvas)
cv2.imshow("template",template)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 用来保存文件的代码，这里注释掉了
# cv2.imwrite("shapes.png",canvas)
# cv2.imwrite("edges.png",edges)
# cv2.imwrite("contours.png",canvas_contour)
# cv2.imwrite("template.png",template)
# cv2.imwrite("matching.png",mark_canvas)