import cv2
from random import randrange
#importing the image data 
trained_face_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#setting up the image 
#img=cv2.imread('all.jpg')
#To capture iamge from webcam
webcam=cv2.VideoCapture(0)
#Iterate forever over frame
while True:
    successful_frame_read,frame=webcam.read()



    #changing color to black and white
    grayscaled_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #decteting the face
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    #drawing a rectangle around the face
    for(x,y,w,h) in face_coordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(randrange(128,256),randrange(128,256),randrange(128,256)),10)

    #showing the image 
    cv2.imshow('Anurag Face Detector',frame)

    #telling the window to to wait until we see the image 
    key=-cv2.waitKey(1)

    #for stoping if q key is pressed
    if key==81  or key==113:
        break

    #relesing  webcam
webcam.release()

    


print("Code Completed ")

