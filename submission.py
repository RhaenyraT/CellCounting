import cv2
import cv2.cv as cv
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


#*************************************************************************************************************
#DETECT IMAGE
#*************************************************************************************************************
img = cv2.imread('normal.jpg')


#*************************************************************************************************************
#COLOR TO GRAY
#*************************************************************************************************************
img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#Image is blurred to reduce noise and help in better prediction of Hough circles
img_gray = cv2.medianBlur(img_gray,5)



#*************************************************************************************************************
#CANNY
#*************************************************************************************************************
edges= cv2.Canny(img,0,100)
cv2.imshow('Canny',edges)
cv2.waitKey(0)


#*************************************************************************************************************
#HOUGH CIRCLES
#*************************************************************************************************************
circles = cv2.HoughCircles(img_gray,cv.CV_HOUGH_GRADIENT,1,50,param1=80,param2=20,minRadius=20,maxRadius=75)
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    cv2.circle(img,(i[0],i[1]),i[2],(0,0,255),3)
    cv2.circle(img,(i[0],i[1]),2,(255,0,0),1)
cv2.imshow('image',img)
cv2.imwrite("houghtest.jpg",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#print circles


#*************************************************************************************************************
#AVERAGE COLOR IN EACH HOUGH CIRCLE #BGR
#*************************************************************************************************************
height,width,depth = img.shape
nCircles = circles.shape[1]
mean_vals = np.zeros((nCircles,3))
for i in xrange(nCircles):
    yc,xc,r = circles[0,i,:]
    circle_img = np.zeros((height,width), np.uint8)
    cv2.circle(circle_img,(yc,xc),r,1,thickness=-1)
    masked_data = cv2.bitwise_and(img, img, mask=circle_img)
    cv2.imshow("circle_img", circle_img*255)
    cv2.waitKey(0)
    cv2.imshow("masked", masked_data)
    cv2.waitKey(0)
    mean_total = cv2.mean(img,circle_img)
    mean_vals[i,:] = np.array(mean_total)[:3]


#*************************************************************************************************************
#UPPER AND LOWER BOUND FOR DECIDING WHICH CIRCLES GO #BGR
#*************************************************************************************************************
minim=min(mean_vals[:,2]) 
maxim=max(mean_vals[:,2]) 
print minim
print maxim
red = mean_vals[:,2]
thresoldlower=minim
thresoldupper=maxim
redfin=red[red>thresoldlower]
redfinal=redfin[redfin<thresoldupper]
NumOfRedCells=len(redfinal)
print NumOfRedCells
print "We counted "+str(NumOfRedCells)+ " cells by BGR."


#*************************************************************************************************************
#AVERAGE COLOR IN EACH HOUGH CIRCLE #HSV
#*************************************************************************************************************
img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
height,width,depth = img.shape
nCircles = circles.shape[1]
mean_vals_hsv = np.zeros((nCircles,3))
for i in xrange(nCircles):
    yc,xc,r = circles[0,i,:]
    circle_img = np.zeros((height,width), np.uint8)
    cv2.circle(circle_img,(yc,xc),r,1,thickness=-1)
    #masked_data = cv2.bitwise_and(img, img, mask=circle_img)
    #cv2.imshow("circle_img", circle_img*255)
    #cv2.waitKey(0)
    #cv2.imshow("masked", masked_data)
    #cv2.waitKey(0)
    mean_total_hsv = cv2.mean(img_hsv,circle_img)
    mean_vals_hsv[i,:] = np.array(mean_total_hsv)[:3]


#*************************************************************************************************************
#UPPER AND LOWER BOUND FOR DECIDING WHICH CIRCLES GO #HSV
#*************************************************************************************************************
minim_hsv=min(mean_vals_hsv[:,0]) 
maxim_hsv=max(mean_vals_hsv[:,0]) 
print minim_hsv
print maxim_hsv
hsv = mean_vals_hsv[:,0]
thresoldlower_hsv=minim_hsv
thresoldupper_hsv=maxim_hsv
hsvfin=hsv[hsv>thresoldlower_hsv]
hsvfinal=hsvfin[hsvfin<thresoldupper_hsv]
NumOfRedCells_hsv=len(hsvfinal)
#print NumOfRedCells_hsv
print "We counted "+str(NumOfRedCells_hsv)+ " cells by HSV."


#HSV and BGR seem equally good for detection of red blood cells, HSV could be better since the
# variation across colors in this particular case is not much and hence hue alone could be used to predict the blood cells. There are
# false positives and negatives in both cases.