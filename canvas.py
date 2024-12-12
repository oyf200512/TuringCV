import pandas as pd
import cv2
import numpy as np
# 空白画布
canvas = np.full((800, 800, 3), 255, dtype = np.uint8)
# 正方形
canvas[100:200, 100: 200, :] = 0
# 三角行
for i in range(100, 201):
    for j in range(250, i + 150):
        canvas[i, j, 1] = 0
# 正位三角形
for k in range(100, 201):
    for l in range(600 - k, 401 + k):
        canvas[k, l, 2] = 0
# 使用rectangle函数绘制矩形
cv2.rectangle(canvas, (100, 250), (200, 450), (200, 200, 50), -1)

# 六边形
pts = np.array([[300, 250], [350, 250], [400, 300], [350, 350], [300, 350], [250, 300]])
#pts = pts.reshape((-1, 1, 2))
cv2.polylines(canvas, [pts], True, (255, 150, 0), 2)
cv2.fillPoly(canvas, [pts], (255, 150, 0))

# 相交五角星
pts1 = np.array([[50, 615], [208, 720], [155, 550], [85, 720], [250, 615]])
cv2.polylines(canvas, [pts1], True, (0, 100, 255), 10)
cv2.fillPoly(canvas, [pts1], (0, 250, 0))
# 不相交五角星(建系求精确坐标)
pts2 = np.array([[300, 615], [380, 615], [403, 550], [430, 615], [500, 615], [437, 652], [458, 720], [403, 677], [340, 720], [366, 652]])
cv2.polylines(canvas, [pts2], True, (150, 100, 255), 5)

# 圆
cv2.circle(canvas, (700, 150), 50, (0, 0, 255), -1)
#print(canvas)
# 椭圆
cv2.ellipse(canvas, (650, 350), (100, 50), 0, 0, 360, (160, 220, 60), -1)
# 半圆
cv2.ellipse(canvas, (650, 550), (100, 100), 180, 0, 180, (0, 220, 150), -1)

# 保存图像
cv2.imwrite("shapes.png", canvas)

if __name__ == "__main__":
    cv2.imshow("shapes", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



