import canvas
import cv2
canvas = cv2.imread("shapes.png")
# 包装方法
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 将图像灰度化并用Canny函数进行边缘检测，阈值为[80, 150]
canvas_edge = cv2.imread("shapes.png", cv2.IMREAD_GRAYSCALE)
edge = cv2.Canny(canvas_edge, 80, 150)
cv2.imwrite("edges.png", edge)
if __name__ == "__main__":
    cv_show("edges", edge)

# 将图像二值化处理
# finContours是黑中找白，出现绘制画布轮廓的原因是把检测内容二值化成了黑色，而背景是白色，故取反即可
ret, thresh = cv2.threshold(canvas_edge, 200, 255, cv2.THRESH_BINARY_INV)
#cv2.imwrite("thresh.png", thresh)
if __name__ == "__main__":
    cv_show("thresh", thresh)

# 检测并绘制轮廓
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 3)
cv2.imwrite("contours.png", canvas)

# 统计数量
if __name__ == "__main__":
    cv_show("contours", canvas)
    print("-" * 40)
    print("图形数量:", len(contours))





