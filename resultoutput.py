import cv2 as cv
import numpy as np

def closeopen_demor(b,g,r):
   gray = r
   ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
   kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
   binaryr = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
   binaryr = cv.morphologyEx(binaryr, cv.MORPH_CLOSE, kernel)
   r2 = binaryr
   gray = g
   ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
   kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
   binaryg = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
   binaryg = cv.morphologyEx(binaryg, cv.MORPH_CLOSE, kernel)
   g2 = binaryg
   gray = b
   ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
   kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
   binaryb = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
   binaryb = cv.morphologyEx(binaryb, cv.MORPH_CLOSE, kernel)
   b2 = binaryb
   src = cv.merge([b2, g2, r2])
   cv.namedWindow("output image", cv.WINDOW_AUTOSIZE)
   cv.imshow("output image", src)
   nrootdir = ("./")
   cv.imwrite(nrootdir + str(0) + ".png", src)

src = cv.imread(r"rgb.png", cv.IMREAD_COLOR)
b, g, r = cv.split(src)
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)

closeopen_demor(b,g,r)

cv.waitKey(0)

cv.destroyAllWindows()
