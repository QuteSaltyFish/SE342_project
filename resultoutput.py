import cv2 as cv
import numpy as np

def closeopen_demor(self):
   r, g, b = cv.split(self.img)
   gray = r
   ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
   kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
   binaryr = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
   r2 = cv.morphologyEx(binaryr, cv.MORPH_OPEN, kernel)
   gray = g
   ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
   kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
   binaryg = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
   g2 = cv.morphologyEx(binaryg, cv.MORPH_OPEN, kernel)
   gray = b
   ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
   kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
   binaryb = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
   b2 = cv.morphologyEx(binaryb, cv.MORPH_OPEN, kernel)
   self.result = cv.merge([r2, g2, b2])
