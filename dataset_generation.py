#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 20:53:59 2020

@author: anilcharles
"""


import cv2
import os

dir = os.getcwd() 

custom_ds_fol = os.getcwd() + "/custom dataset"
if not os.path.exists(custom_ds_fol):
    os.makedirs(custom_ds_fol)

camera = cv2.VideoCapture(0)

hand_cascade =  cv2.CascadeClassifier("assets/haarcascade_frontalface_default.xml")

count = 0
while True:
    count += 1
    _,frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (320, 120), interpolation = cv2.INTER_AREA)
    _,thresh = cv2.threshold(frame,140,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    
    #cv2.imwrite(custom_ds_fol + "/{}.jpg".format(count), thresh)
    hands = hand_cascade.detectMultiScale(frame, 1.1, 5)
    for (x,y,w,h) in hands:
        cv2.rectangle(frame,(x,y), (x+w, y+h), (0, 255, 0), 2 )
    cv2.imshow("img",frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

    if count == 1000:
        break
    
camera.release()
cv2.destroyAllWindows()


import cv2

img = cv2.imread("/Users/anilcharles/Downloads/new.jpg")

camera = cv2.VideoCapture(0)

while True:
    a = camera.read()
    print(a)


import cv2
from keras.models import load_model
import numpy as np


def lookup_table():
    
   dict =  {'01_palm': 0,
            '02_l': 1,
            '03_fist': 2,
            '04_fist_moved': 3,
            '05_thumb': 4,
            '06_index': 5,
            '07_ok': 6,
            '08_palm_moved': 7,
            '09_c': 8,
            '10_down': 9}
   
   return dict

reverse_dict = { lookup_table()[k]:k for k in lookup_table()}


model = load_model("/Users/anilcharles/Downloads/prototype-lights-on/trained.h5")
import numpy as np
camera = cv2.VideoCapture(0)
back = cv2.createBackgroundSubtractorMOG2()
while True:
    _,frame = camera.read()
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    color = frame
    #frame = cv2.resize(frame, (320, 120), interpolation = cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (320, 120), interpolation = cv2.INTER_AREA)
    #_,thresh = cv2.threshold(frame,140,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    #img_ = cv2.createBackgroundSubtractorMOG2().apply(img)
    img_ = back.apply(frame)
    #frame = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(img_,10,255, cv2.ADAPTIVE_THRESH_MEAN_C)
    cv2.imshow("aa",thresh)

    frame = np.array(thresh, dtype="float32") 
    frame = frame.reshape(1,frame.shape[0],frame.shape[1], 1)
    text = np.argmax(model.predict(frame))
    print(text)
    im = cv2.putText(color, "_".join(reverse_dict[text].split("_")[1:]) ,(304,662), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA)
    #cv2.imshow("b",img_)
    cv2.imshow("a",im)
    
    
    
    _,thresh = cv2.threshold(img_, 127 ,255, cv2.THRESH_BINARY)

    contours,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    hull = [cv2.convexHull(c) for c in contours]
    final = cv2.drawContours(img_, hull, -1, (0,255,0), 3)

    cv2.imshow("img",img_)
    
    
    
camera.release()
cv2.destroyAllWindows()



s = cv2.VideoCapture(a)





import cv2 
camera = cv2.VideoCapture(0)
_, frame1 = camera.read()
_, frame2 = camera.read()
while True:
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    diff =  cv2.absdiff(frame1, frame2)
    img = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5,5), 0)
    _,thresh = cv2.threshold(img, 20 ,255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < 700:
            continue
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 3)
    
    cv2.imshow("a",thresh)
    frame1 = frame2
    _,frame2 = camera.read()
    


camera.release()
cv2.destroyAllWindows()








cv2.bgsegm.createBackgroundSubtractorGMG().apply(img)







import cv2 
camera = cv2.VideoCapture(0)


face_classifier = cv2.CascadeClassifier("Haarcascades/haarcascade_frontalface_default.xml")
#hand.load(cv2.samples.findFile('Haarcascades/hand'))

while True:
    
    _,frame = camera.read()
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        camera.release()
        cv2.destroyAllWindows()
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.1, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow("a",frame)
    

camera.release()
cv2.destroyAllWindows()

img = cv2.imread("new.jpg")

import numpy as np
import cv2

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontal/face_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

import requests
out = requests.get("https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml")




classifier = cv2.CascadeClassifier("/Users/anilcharles/Downloads/prototype-lights-on/Haarcascades/haarcascade_frontalface_default.xml")
img = cv2.imread("new.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = classifier.detectMultiScale(gray, 1.3, 5)



"/Users/anilcharles/Downloads/prototype-lights-on/Haarcascades/haarcascade_frontal/face_default.xml"
