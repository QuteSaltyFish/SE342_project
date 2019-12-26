import cv2
img = cv2.imread('image/img1.jpg') 

img = cv2.medianBlur(img,5) 
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR) 
if img == None: 
    print("There is no image file. Quiting...")
    quit() 

circles = cv2.HoughCircles(img,cv.CV_HOUGH_GRADIENT,3,50, 
          param1=55,param2=125,minRadius=25,maxRadius=45) 

circles = np.uint16(np.around(circles)) 
for i in circles[0,:]: 
    # draw the outer circle 
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2) 
    # draw the center of the circle 
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3) 

print(len(circles[0,:]))
cv2.imshow('detected circles',cimg) 
cv2.waitKey(0) 
cv2.destroyAllWindows()