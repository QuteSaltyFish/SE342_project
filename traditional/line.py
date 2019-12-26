#coding=utf-8
import cv2
import numpy as np
import torch as t 
t.nn.CrossEntropyLoss()
img = cv2.imread("image/img1.jpg")
img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
img_gray[img_gray>200]=0
img_gray[img_gray<5]=255
cv2.imwrite('grey.jpg',img_gray)
[h,w,c] = img.shape
img = cv2.GaussianBlur(img,(3,3),0)
edges = cv2.Canny(img, 50, 150, apertureSize = 3)
lines = cv2.HoughLines(edges,1,np.pi/180,118)
result = img.copy()
cv2.imwrite('edge.jpg', edges)
#经验参数
minLineLength = 20
maxLineGap = 5
lines = cv2.HoughLinesP(edges,1,np.pi/180,80,minLineLength,maxLineGap)
for x1,y1,x2,y2 in lines[0]:
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imwrite('Result.jpg', img)