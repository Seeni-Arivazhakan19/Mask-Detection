import cv2
import numpy as np
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    if(True):
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),((x+w),(y+h)),(0,0,255),2)
            cv2.putText(frame, ' Please Wear a Mask', (100, 120), font, 1,  (255, 0, 0), 2, cv2.LINE_4)
            roi_gray=gray[y:y+h,x:x+w]
            roi_color=frame[y:y+h,x:x+w]
            smiles = smile_cascade.detectMultiScale(roi_gray,1.8,20)
            for (sx,sy,sw,sh) in smiles:
                cv2.rectangle(roi_color,(sx, sy),((sx+sw),(sy+sh)),(255,0,0),2)
    else:
        call()
    return frame
def call():
    while(True):
        ret,frame=cap.read()
        cv2.putText(frame, 'Thanks, follow social distancing', (50, 50), font, 1,  (0, 0, 255), 2, cv2.LINE_4)
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        canvas=detect(gray,frame)
        cv2.imshow('frame',canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap=cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
call()
cap.release()
cv2.waitkey(0)
cv2.destroyAllWindows()
