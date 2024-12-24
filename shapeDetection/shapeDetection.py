import cv2
import numpy as np

def nothing(x):
    pass
cap = cv2.VideoCapture(0)

cv2.namedWindow('trackbars')
cv2.createTrackbar('L-H', 'trackbars', 0, 10, nothing)
cv2.createTrackbar('L-S', 'trackbars', 100, 255, nothing)
cv2.createTrackbar('L-V', 'trackbars', 100, 255, nothing)
cv2.createTrackbar('U-H', 'trackbars', 180, 180, nothing)
cv2.createTrackbar('U-S', 'trackbars', 255, 255, nothing)
cv2.createTrackbar('U-V', 'trackbars', 255, 255, nothing)
while (True):
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    l_h = cv2.getTrackbarPos('L-H','trackbars')
    l_s = cv2.getTrackbarPos('L-S','trackbars')
    l_v = cv2.getTrackbarPos('L-V','trackbars')
    u_h = cv2.getTrackbarPos('U-H','trackbars')
    u_s = cv2.getTrackbarPos('U-S','trackbars')
    u_v = cv2.getTrackbarPos('U-V','trackbars')
    # lower_red = np.array([l_h,l_s,l_v])
    # upper_red = np.array([u_h,u_s,u_v])
    lower_red1 = np.array([0, l_s, l_v])
    upper_red1 = np.array([10, u_s, u_v])
    lower_red2 = np.array([170, l_s, l_v])
    upper_red2 = np.array([180, u_s, u_v])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    # mask = cv2.inRange(hsv, lower_red,upper_red)
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.erode(mask,kernel)


    # contours detection 
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for cont in contours :
        area = cv2.contourArea(cont)
        approx = cv2.approxPolyDP(cont,0.01*cv2.arcLength(cont,True),True)
        x = approx.ravel()[0]
        y= approx.ravel()[1]

        if area >200:
            cv2.drawContours(frame , [approx], 0 ,(0,0,0),5)
            if len(approx) == 4:
                cv2.putText(frame,"reactangle",(x,y),font,1,(0,0,0),2)
            if len(approx) == 3 :
                cv2.putText(frame,"triangle",(x,y),font,1,(0,0,0),2)
            if len(approx) >10:
                cv2.putText(frame,"circle",(x,y),font,1,(0,0,0),2)

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)

    if cv2.waitKey(1) == ord('q'):
        break
