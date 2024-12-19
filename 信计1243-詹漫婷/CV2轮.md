

# CV二轮考核

## 考核内容

使用 OpenCV 和 NumPy 等实现一个综合图形处理程序，并上传到GitHub上

1. 几何图形绘制
2. 边缘检测与计数
3. 模板匹配

## 具体要求

### 1. 使用OpenCV几何图形绘制
- 创建 800×800 白色画布
- 绘制至少 10 个几何图形
  - 包含 3 种以上不同形状
  - 尽可能使用不同颜色
  - 合理安排大小和位置

### 2. 边缘检测与计数
- 对绘制的图形进行边缘检测
- 使用轮廓检测统计图形数量
- 在原图上标注检测到的轮廓

### 3. 模板匹配
- 从原图自选截取任意一个几何图形作为模板
- 在原图中查找相似图形
- 在结果图上标注匹配位置

## 提交内容
以文件夹形式上传 文件夹命名为`班级-姓名`  
文件夹内容如下
| 文件名 | 说明 |
|--------|------| 
| `TuringCV.py` | 源代码文件 |
| `shapes.png` | 原始几何图形 |
| `edges.png` | 边缘检测结果 |
| `contours.png` | 轮廓检测结果 |
| `template.png` | 模板图像 |
| `matching.png` | 模板匹配结果 |
| `README.md` | 代码思路说明 |


### 需要用到的技能
- [Markdown 基础语法教程](https://markdown.com.cn/basic-syntax/)
- [Git 入门教程](https://www.bilibili.com/video/BV1Cr4y1J7iQ)

## 注意事项
1. 代码需要规范，添加必要的注释
2. README 文档需清晰说明实现思路
3. 确保所有图片输出清晰可见
4. 上传流程参考：fork -> git clone -> git add  -> git commit -m "备注" -> git push origin main -> pull request
5. 在GitHub上提交Pull Request时，标题格式：`班级-姓名`

