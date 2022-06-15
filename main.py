import cv2
import face_recognition

freda=face_recognition.load_image_file('database/IMG_11030.jpg') #to load image
freda=cv2.cvtColor(freda,cv2.COLOR_BGR2RGB)                  

freda_test_image=face_recognition.load_image_file('database/group.jpg')
freda_test_image=cv2.cvtColor(freda_test_image,cv2.COLOR_BGR2RGB)


freda_loc= face_recognition.face_locations(freda)[0]
print(freda_loc)
freda_encodings= face_recognition.face_encodings(freda)[0]
cv2.rectangle(freda,(freda_loc[3],freda_loc[0]),(freda_loc[1],freda_loc[2]),(255,0,0),3)


freda_test_loc= face_recognition.face_locations(freda_test_image)[0]
freda_test_encodings= face_recognition.face_encodings(freda_test_image)[0]
cv2.rectangle(freda_test_image,(freda_test_loc[3],freda_test_loc[0]),(freda_test_loc[1],freda_test_loc[2]),(255,0,0),3)




compare_faces=face_recognition.compare_faces([freda_encodings],freda_test_encodings)
distances_img=face_recognition.face_distance([freda_encodings],freda_test_encodings)
print(compare_faces)
print(distances_img)

cv2.putText(freda,f'{compare_faces} {round(distances_img[0],1)}' ,(50,50),  cv2.FONT_ITALIC,2,(0,255,0),2)

cv2.imshow("freda",freda)
cv2.imshow("freda test",freda_test_image)
cv2.waitKey(0)

