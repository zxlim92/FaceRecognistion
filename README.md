# FaceRecognistion
TO RUN THE PROGRAM PLZ OPEN THE "runMe.py"
Name: Nursultan Aibekuly, Le Minh Dang, Darius Fang, Zi Xue Lim, Chen Zhou
Project: D_END Face Recognition 
HackED 2020

Modules:
	- PIL
	- tkinter
	- tkinter.font
	- sys
	- os
	- cv2
	- numpy
	- matplotlib.pyplot

Included Files:
	- runMe.py
	- README
	- def.txt
	-dataset

Descriptions:
	- Recognizes faces and output names.
	- If a new face is detected, the program saves pictures and stores that in a folder. 
	- The machine then learns the face and outputs with a name.

Running Instructions:
	- Download python with Tkinter and opencv, opencv-contrib and clone the git repository
	- Run the program runMe.py
	- Click on the icon, in the pop up window, you should see two options: "New Face ID", and "Detection". 
	- If you are a new user, choose "New Face ID" this will save your face to the system. It will ask you to input your name. Now click  "Detection", and the program, it should be able to recognize your face. 

Notes and Assumptions:
	Running training and detector files sometimes will lag the program. When the computer takes the piture, the orentation of your face should be the same as how you originally set up the id. The new face id will take around 15 seconds. The training will take 5-10 seconds. To close the face detection, press escape key. It can do up to three people, the more people the harder it is to detect more faces
