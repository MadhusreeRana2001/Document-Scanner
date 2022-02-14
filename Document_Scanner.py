'''

---------------------------------DOCUMENT SCANNER------------------------------------------

This code takes the url of an image as the input, finds the four corners of it,
and finally, by applying a perspective transform and thresholding on it, returns and
displays a top-down, bird’s-eye view of it with a black and white look.

Requirement: Stable Internet Connection for using the URL of the image

URL for the image used in the code:
https://thumbs.dreamstime.com/z/klang-malaysia-may-paper-sales-receipt-
isolated-wooden-background-klang-malaysia-may-paper-sales-receipt-isolated-wooden
-218014311.jpg
'''


import cv2
from skimage import io
import numpy as np


widthImg=400
heightImg =560


def preProcessing(img):
    '''to return the image with the edges detected, which have been dilated thrice and
eroded once'''
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv2.Canny(imgBlur,200,200)
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny,kernel,iterations=3)
    imgThres = cv2.erode(imgDial,kernel,iterations=1)
    return imgThres


def getContours(img):
    '''to find contours of the input image and return the four corner points of the
biggest contour detected.'''
    biggest = np.array([])
    maxArea = 0
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>5000:
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            if area >maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    cv2.drawContours(imgContour, biggest, -1, (255, 0, 0), 20)
    return biggest


def reorder (myPoints):
    '''for reordering the corner points of the biggest contour detected with respect
to the standard arrangement as mentioned in the getWarp() function'''
    myPoints = myPoints.reshape((4,2))
    myPointsNew = np.zeros((4,1,2),np.int32)
    add = myPoints.sum(axis=1)  # for detecting the first and last points
    diff = np.diff(myPoints, axis=1)  # for detecting the second and third points
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    myPointsNew[1]= myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew


def getWarp(img,biggest):
    '''for returning the warped image with proper cropping'''
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    imgCropped = imgOutput[20:imgOutput.shape[0]-20,20:imgOutput.shape[1]-20]
    imgCropped = cv2.resize(imgCropped,(widthImg,heightImg))
    return imgCropped


def changeToBandW(img):
    '''for applying image thresholding on the final warped image and returning a fully
binarized or black and white version of it'''
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, b_w_img=cv2.threshold(gray, 178, 255, cv2.THRESH_OTSU)
    return b_w_img


def stackImages(scale,imgArray):
    '''to stack images of the same size, by adjusting their scale, irrespective of their
number of channels'''
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1],
                                            imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor(
                    imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1],
                                                imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x],
                                                                       cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


url="https://thumbs.dreamstime.com/z/klang-malaysia-may-paper-sales-receipt-isolated-" \
    "wooden-background-klang-malaysia-may-paper-sales-receipt-isolated-wooden-218014311.jpg"
img=io.imread(url)
imgContour = img.copy()
imgThres = preProcessing(img)
biggest = getContours(imgThres)

if biggest.size != 0:
    imgWarped=getWarp(img,biggest)
    b_w_img=changeToBandW(imgWarped)
    cv2.imshow("Scanned Image", b_w_img)

    img = cv2.resize(img, (widthImg, heightImg))
    imgThres = cv2.resize(imgThres, (widthImg, heightImg))
    b_w_img=cv2.resize(b_w_img, (widthImg, heightImg))
    Work_Flow=stackImages(0.6,([img, imgThres, b_w_img]))
    cv2.imshow("Work Flow", Work_Flow)

else:
    cv2.imshow("Original Image", img)

cv2.waitKey(0)
if 0xFF == ord('q'): exit()