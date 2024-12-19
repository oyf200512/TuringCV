# TuringCV 二轮





# 一.图形绘制部分

### 1.创建画布
用numpy创建一个全1的三维数组，数组*255得到全255的三维数组，得到白色画布

### 2.图形绘制
rectangle函数绘制矩形<br/>
创建自定义函数triangle(canvas,x1,x2,x3)绘制三角形输入三角形三顶点，三次使用cv2.line函数画出相连三条线形成三角形  
circle函数绘制圆形<br/>
ellipse函数设置不同长短轴和角度为360绘制椭圆 设置相同长短轴和角度为120绘制一个扇形 角度为180绘制半圆<br/>
创建自定义函数fill_image(img,pts,color)绘制多边形 依次输入n个点 通过cv2.polylines函数连接各个点并再通过cv2.fillPoly
对图形进行颜色填充<br/>

### 3.图像保存
cv2.imwrite将图像保存为shapes.png





# 二.图形处理部分

### 1.图像预处理
cv2.cvtColor BGR2GRAY 得到原图的灰度图
cv2.threshold 二值法图像阈值操作 设置阈值为150 最大值为255 得到二值图

### 2.canny边缘检测
canny 计算每个像素点的梯度 再通过双阈值检测排除一些非边缘数据

### 3.保存边缘检测结果edges.png

### 4.轮廓检测
findContours 检索方式使用RETR_TREE检索所有的轮廓 轮廓逼近方式为CHAIN_APPROX_SIMPLE压缩水平竖直只需要保留顶点部分 
可以在后面找到外界矩形 

### 5.对轮廓按照面积排序
由于轮廓包括了整个画布的最外层边界(即800*800的面积 是面积最大的) 可以对轮廓排序后去除最大的 得到真正几何图形的轮廓
创建自定义函数cort_contours(cnts) 先找出每个轮廓的外界矩形 以元组形式保存各个大小x,y,w,h数据于boundingboxes列表
将轮廓cnt和大小数据boundingboxes通过zip之后排序 运用lambda表达式用boundingboxes[1][2]xboundingboxes[1][3]
即WxH得到排序的key为每个几何图形的面积<br/>
sort后用切片去掉最后一个得到含有12个边界的contours列表

### 6.绘制轮廓
创建原图的一个copy
cv2.drawContours 索引设置为-1 在copy上画出全部轮廓得到contours.png

### 7.保存轮廓检测结果contours.png





# 三.模板匹配部分

### 1.随机选择一个模板图
用random获得1-12一个随机数得到一个随机模板的轮廓数据template_contours<br/>
用boundingRect得到模板的x,y,w,h<br/>
通过对原图的切 得到模板图template.png

### 2.保存模板图template.png

### 3.截取原图
类似于从原图切出模板图 把原图中的所有小几何图形都切出来保存在shape列表中 便于下一步模板匹配<br/>

### 4.开始模板匹配
先创建一个scores列表用于保存模板匹配的结果<br/>
遍历shape列表 与模板template用cv2.matchTemplate函数进行匹配 匹配方法使用TM_CCOEFF_NORMAL 即计算相关系数并归一化
有助于提高准确度 得到scores是一个有12个数值的列表 由于归一化 越接近1的与原图相关性越大<br/>
用index 与 max函数得到最接近的在scores中的索引 因为scores中的顺序与contours中是一致的 可以直接用此索引得到匹配图的轮廓<br/>
再创建一个copy 在上面绘制模板匹配的结果matching

### 5.保存模板匹配结果matching.png



 
