import cv2
import face_recognition
import numpy as np
import os


file_location = 'database'
imagess = os.listdir(file_location)
print(imagess)
read_images=[]
person_name_only=[]

for image in imagess:
    read_image=cv2.imread(f'{file_location}/{image}') 
    read_images.append(read_image)
    person_name_only.append(os.path.splitext(image)[0])
print(person_name_only)


def encoder(read_images):
    encoded_images_list=[]
    
    for unencoded_image in read_images:
        unencoded_image=cv2.cvtColor(unencoded_image,cv2.COLOR_BGR2RGB) 
        encode_image= face_recognition.face_encodings(unencoded_image)[0]
        encoded_images_list.append(encode_image)
    return encoded_images_list


start= encoder(read_images)
print(len(start),start)
        
    
    
#     reda_encodings= face_recognition.face_encodings(freda)[0]
# cv2.rectangle(freda,(freda_loc[3],freda_loc[0]),(freda_loc[1],freda_loc[2]),(255,0,0),3)

# freda_test_loc= face_recognition.face_locations(freda_test_image)[0]
# freda_test_encodings= face_recognition.face_encodings(freda_test_image)[0]
# cv2.rectangle(freda_test_image,(freda_test_loc[3],freda_test_loc[0]),(freda_test_loc[1],freda_test_loc[2]),(255,0,0),3)


# print(freda_loc,freda_test_loc)

# compare_faces=face_recognition.compare_faces([freda_encodings],freda_test_encodings)
# distances_img=face_recognition.face_distance([freda_encodings],freda_test_encodings)
# print(compare_faces)