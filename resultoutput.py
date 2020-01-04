import cv2 as cv
import numpy as np

def closeopen_demor(b,g,r):
   gray = r
   ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
   kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
   binaryr = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
   r2 = cv.morphologyEx(binaryr, cv.MORPH_CLOSE, kernel)
   gray = g
   ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
   kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
   binaryg = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
   g2 = cv.morphologyEx(binaryg, cv.MORPH_CLOSE, kernel)
   gray = b
   ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
   kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
   binaryb = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
   b2 = cv.morphologyEx(binaryb, cv.MORPH_CLOSE, kernel)
   src = cv.merge([b2, g2, r2])

b, g, r = cv.split(src)
closeopen_demor(b,g,r)
