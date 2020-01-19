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
rec =cv2.face.LBPHFaceRecognizer_create();
rec.read("recognizer\\trainingData.yml")
id = 0
font =cv2.FONT_HERSHEY_SIMPLEX
while(1):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),255)
        id, conf=rec.predict(gray[y:y+h,x:x+h])
        if(id==1):
            id="Xue"
        if(id ==2):
            id = "Clyde"
        if(id==3):
            id="LEBRON"
        if(id == 4):
            id = "Nursultan"
        cv2.putText(img,str(id),(x,y+h),font,2,(0,0,0))
        roi_gray = gray[y:y+h , x:x+w]
        roi_color = img[y:y+h , x:x +w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        """
        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    """
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
cap.release()
cv2.destroyAllWindows()