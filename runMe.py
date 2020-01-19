
import tkinter as tk
import sys
import os
import cv2
import numpy as np
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt

from tkinter.font import Font
FONT = ('Fixedsys', 30)
class AppWindow(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)# Create Window
        self._frame = None
        self.Switch(TitlePage, 0, 0)# Title Scene

    def Switch(self, frame, num, name):
        newFrame = frame(self)#
        if self._frame is not None:
            self._frame.destroy()
        self._frame = newFrame
        self._frame.pack()
        if (num == 1):
            NewFace(name)

def NewFace(name):
    """  
Detects the face and then gives the ID the name
Arguments: name: Given name from user input
Returns Nothing
Will go to Training()
"""
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    
    #id = input('enter user id(number)')
    sampleNum = 0
    dontUpdate =0
    with open("def.txt",'r+') as file:
        data = file.read()
        data = data.split()
        lenData = len(data)
        try:
            exist = data.index(name)
        except ValueError:
            pass
        else:
            dontUpdate == 1
        # if file blank start at 0
        if(dontUpdate == 0):
            try:
                holder = data[lenData-2]
            except:
                holder =0
            id = int(holder) +1
            tempdat = "{0} {1} \r\n".format(id,name)
            file.write(tempdat)
            while(1):
                ret, img = cap.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray,1.3,5)
                for (x,y,w,h) in faces:
                    sampleNum = sampleNum + 1
                    if(sampleNum >25):
                        break
                    print(sampleNum)
                    cv2.imwrite("dataset/User."+str(id)+"."+ str(sampleNum)+".jpg", gray[y:y+h,x:x+h])
                    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                    roi_gray = gray[y:y+h , x:x+w]
                    roi_color = img[y:y+h , x:x+w]
                    eyes = eye_cascade.detectMultiScale(roi_gray)
                    """
                    for(ex,ey,ew,eh) in eyes:
                        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                    """
                    cv2.imshow('img',img)
                if(sampleNum >25):
                        break
        cap.release()
        cv2.destroyAllWindows()
        Training()

def Training():
    """  
Trains the computer to recognise the face
Returns Nothing
Go back to Proccess GUI
    """
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    path = 'dataset'
    
    def getImageWithID(path): #get all pictures saved
         imagePaths =[os.path.join(path,f) for f in os.listdir(path)]
         faces = []
         IDs=[]
         for imagePath in imagePaths:
             faceImg = Image.open(imagePath).convert('L')
             faceNp = np.array(faceImg,'uint8')
             ID=int(os.path.split(imagePath)[-1].split('.')[1])
             faces.append(faceNp)
             print (ID)
             IDs.append(ID)
             cv2.imshow("training",faceNp)
             cv2.waitKey(10)
         return IDs,faces
    Ids,faces = getImageWithID(path)
    recognizer.train(faces,np.array(Ids))
    recognizer.save('recognizer/trainingData.yml')
    cv2.destroyAllWindows()
    
    os.system('python3 training.py')

def Detect():
    """  
Trains the computer to recognise the face
Returns Nothing
Go back to Proccess GUI
    """
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    rec =cv2.face.LBPHFaceRecognizer_create();
    rec.read("recognizer\\trainingData.yml")
    id = 0
    nameDic= {}
    font =cv2.FONT_HERSHEY_SIMPLEX
    with open("def.txt",'r') as file:
        data = file.read()
        data = data.split()
        lenData = len(data)
        for x in range(0,lenData,2):
            nameDic[int(data[x])] = data[x+1]
        while(1):
            ret, img = cap.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray,1.3,5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),255)
                id, conf=rec.predict(gray[y:y+h,x:x+h])
                id = nameDic.get(id)
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

class TitlePage(tk.Frame):
    def __init__(self, master): 
        img = ImageTk.PhotoImage(Image.open("L.jpg"))# Image
        tk.Frame.__init__(self, master)# Create Frame
        tk.Frame.configure(self, height = 400, width = 400, bg = 'black')
        Fill = tk.Label(self, bg = 'black', pady = 50)# Fill top
        B_Start = tk.Button(self, image = img, fg = 'green', padx = 90, font = FONT,
            command = lambda: master.Switch(Proccess, 0, 0), activeforeground = 'green',
            activebackground = 'black', bd = 0, highlightthickness = 0)# Button Image
        
        # Show widgets
        Fill.pack()
        B_Start.photo = img
        B_Start.pack(fill = 'both', expand = 'yes')

class Proccess(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        tk.Frame.configure(self, height = 400, width = 400, bg = 'black')
        Fill = tk.Label(self, bg = 'black', pady = 50)

        B_NewFace = tk.Button(self, text = "New Face ID", bg = 'black', fg = 'green', padx = 66, pady = 60, font = FONT,
            command = lambda: master.Switch(NewID, 0, 0), activeforeground = 'green',
            activebackground = 'black', highlightthickness = 0, relief = 'ridge')# New Face Button
        
        B_Detection = tk.Button(self, text = "Detection", bg = 'black', fg = 'green', padx = 90, pady = 60, font = FONT,
            command = Detect, activeforeground = 'green',
            activebackground = 'black', highlightthickness = 0, relief = 'ridge')# Detection
        
        # Show widgets
        Fill.pack()
        B_NewFace.pack()
        B_Detection.pack()

class NewID(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        tk.Frame.configure(self, height = 400, width = 400, bg = 'black')
        Fill = tk.Label(self, bg = 'black', pady = 80)
        E = tk.Entry(self, font = FONT)
        B_Enter = tk.Button(self, text = "Enter", bg = 'black', fg = 'green', font = FONT,
            command = lambda: master.Switch(Proccess, 1, E.get()), activeforeground = 'green',
            activebackground = 'black', highlightthickness = 0, relief = 'ridge')
        
        Fill.pack()
        E.pack()
        B_Enter.pack()

if __name__ == "__main__":
    app = AppWindow()
    app.title("FaceID")
    app.configure(background = 'black')
    app.geometry("600x600")# Resolution
    app.mainloop()