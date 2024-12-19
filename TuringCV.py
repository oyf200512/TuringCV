import cv2
import numpy as np
from matplotlib.pyplot import imshow

# 创建一个800x800的白色画布
canvas = np.ones((800, 800, 3), dtype=np.uint8) * 255

# 定义颜色
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]

# 绘制不同形状的几何图形
# 圆形
cv2.circle(canvas, (100, 100), 50, colors[0], -1)
cv2.circle(canvas, (250, 100), 60, (156,123,93), -1)
# 矩形
cv2.rectangle(canvas, (200, 200), (400, 400), colors[1], -1)
cv2.rectangle(canvas, (500, 200), (700, 400), colors[1], -1)
# 三角形
pts = np.array([[500, 500], [600, 700], [400, 700]], np.int32)
pts = pts.reshape((-1, 1, 2))
cv2.fillPoly(canvas, [pts], colors[2])
# 椭圆
cv2.ellipse(canvas, (100, 500), (100, 50), 0, 0, 360, colors[3], -1)
cv2.ellipse(canvas, (100, 700), (100, 50), 0, 0, 360, colors[3], -1)
# 多边形
pts = np.array([[700, 100], [750, 150], [700, 200], [650, 150]], np.int32)
pts = pts.reshape((-1, 1, 2))
cv2.polylines(canvas, [pts], True, colors[4], 3)
#保存图片
cv2.imwrite('shapes.png',canvas)

# 显示画布
cv2.imshow('Canvas', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
# 转换为灰度图像
gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

# 使用Canny算法进行边缘检测
edges = cv2.Canny(gray, 50, 150)
#显示检测结果
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
#保存边缘
cv2.imwrite('edges.png',edges)
# 查找轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 在原图上标注轮廓
contour_canvas = canvas.copy()
cv2.drawContours(contour_canvas, contours, -1, (0, 255, 0), 2)
#保存轮廓
cv2.imwrite('contours.png',contour_canvas)
# 显示结果
cv2.imshow('Contours', contour_canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 输出轮廓数量
print(f"Detected {len(contours)} shapes.")
# 截取一个图形作为模板，这里以圆形为例
template = canvas[50:150, 50:150]
#保存模板
cv2.imwrite('template.png',template)

# 将模板转换为灰度图像
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# 获取模板的宽度和高度
w, h = template.shape[:2]

# 进行模板匹配
res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)

# 设置阈值
threshold = 0.8
loc = np.where(res >= threshold)

# 在原图上标注匹配位置
match_canvas = canvas.copy()
for pt in zip(*loc[::-1]):
    cv2.rectangle(match_canvas, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)
#保存匹配
cv2.imwrite('matching.png',match_canvas)
# 显示结果
cv2.imshow('Matching', match_canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
