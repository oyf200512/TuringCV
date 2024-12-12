import detection
from PIL import Image, ImageDraw, ImageFont
import cv2
import easygui
import numpy as np
canvas = cv2.imread("shapes.png")
canvas_edge = cv2.imread("shapes.png", cv2.IMREAD_GRAYSCALE)
ret, thresh = cv2.threshold(canvas_edge, 200, 255, cv2.THRESH_BINARY_INV)

# 包装方法
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 选取模板
font = ImageFont.truetype("simhei.ttf", 30, encoding="utf-8")
canvas_text = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
pilimg = Image.fromarray(canvas_text)
draw = ImageDraw.Draw(pilimg)
draw.text((20, 20), "按下q键输入作为模板的图形", (255, 0, 0), font = font)
cv2charimg = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
cv2.imshow("cv2charimg", cv2charimg)

# 截取图像作为模板
while True:
    key_code = cv2.waitKey(1000)
    if key_code & key_code == ord('q'):
        vol = int(easygui.enterbox("输入作为模板的图像索引(0~9):"))
        break
cv2.destroyAllWindows()
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
x, y, w, h = cv2.boundingRect(contours[vol])
canvas_model = canvas[y: y + h, x: x + w]
cv2.imwrite("template.png", canvas_model)
cv2.imshow("template", canvas_model)
cv2.waitKey(1000)
cv2.destroyAllWindows()
canvas_model = cv2.cvtColor(canvas_model, cv2.COLOR_BGR2GRAY)

# 模板匹配
height, wide = canvas_model.shape[: 2]
res = cv2.matchTemplate(canvas_edge, canvas_model, cv2.TM_SQDIFF)
'''
threshold = 0.90
loc = np.where(res >= threshold)
'''
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
'''
for pts in zip(*loc[::-1]):
    cv2.rectangle(canvas, pts, (pts[0] + canvas_model.shape[1], pts[1] + canvas_model.shape[0]), (0, 255, 0), 1)
'''
top_left = min_loc
bottom_right = (top_left[0] + wide, top_left[1] + height)
cv2.rectangle(canvas, top_left, bottom_right, (0, 255, 0), 3)

cv2.imwrite("matching.png", canvas)
cv_show("matching", canvas)


