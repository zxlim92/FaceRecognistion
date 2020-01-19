# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 12:25:54 2020

@author: limzi
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
id = input('enter user id')
sampleNum = 0
while(1):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        sampleNum = sampleNum + 1
        if(sampleNum >20):
            break
        print(sampleNum)
        cv2.imwrite("dataset/User."+str(id)+"."+ str(sampleNum)+".jpg", gray[y:y+h,x:x+h])
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h , x:x+w]
        roi_color = img[y:y+h , x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        if(sampleNum>20):
            break
        cv2.imshow('img',img)
        k = cv2.waitKey(30) & 0xff
        if k==27:
            break
cap.release()
cv2.destroyAllWindows()