import cv2
import os

vidcap = cv2.VideoCapture(r"C:\Users\Dell\Downloads\frame_vid.mp4")
count = 0

if not os.path.exists("frames"):
    os.makedirs("frames")


while True:   
  success,frame = vidcap.read()
  
  cv2.imshow("output",frame)
  cv2.imwrite("./frames/frame"+str(count)+".jpg",frame)
  
  count += 70