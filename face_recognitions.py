import cv2
import face_recognition
from PIL import Image
from pathlib import Path

# create a class for the face recognition
class FaceRecognition:
    def __init__(self, image_path:any):
        self.image_path = image_path
        self.image = cv2.imread(self.image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)  # change the color of the image
        self.image_encodings = face_recognition.face_encodings(self.image)
        self.image_encodings = self.image_encodings[0]

    # face locations
    def face_locations(self, image:any) -> any:
        self.image_face_locations = face_recognition.face_locations(image)
        return self.image_face_locations

    # number of faces in the image
    def number_of_faces(self, image:any) -> any:
        faces = self.face_locations(image)
        self.number_of_face = len(faces)
        return self.number_of_face

    # face encodings
    def face_encodings(self, image:any) -> any:
        self.image_face_encodings = face_recognition.face_encodings(image)
        return self.image_face_encodings

    # face landmarks
    def face_landmarks(self, image:any) -> any:
        self.image_face_landmarks = face_recognition.face_landmarks(image)
        return self.image_face_landmarks

    # face distance
    def face_distance(self, image_path:any) -> any:
        self.image_path = image_path
        self.image = cv2.imread(self.image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)  # change the color of the image
        self.image_encodings = self.face_encodings(self.image)
        self.image_encodings = self.image_encodings[0]
        self.face_distance = face_recognition.face_distance([self.image_encodings], self.image_encodings)
        return self.face_distance

    # compare faces
    def compare_faces(self, image_path:any) -> any:
        self.image_path = image_path
        self.image = cv2.imread(self.image_path)
        
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)  # change the color of the image
        self.image_encodings = face_recognition.face_encodings(self.image)
        self.image_encodings = self.image_encodings[0]
        self.compare_faces = face_recognition.compare_faces([self.image_encodings], self.image_encodings)
        return self.compare_faces


    # face similarity
    def face_similarity(self, image:any, all_images_path:any) -> any:
        # Load the image of the person we want to find similar people for
        known_image = face_recognition.load_image_file(image)

        # Encode the known image
        known_image_encodings = self.face_encodings(known_image)
        known_image_encoding = known_image_encodings[0]

        # Variables to keep track of the most similar face match we've found
        best_face_distance = 1.0
        best_face_image = None

        # Loop over all the images we want to check for similar people
        for image_path in Path(all_images_path).glob("*.png"):
            # Load an image to check
            unknown_image = face_recognition.load_image_file(image_path)

            # Get the location of faces and face encodings for the current image
            face_encodings = self.face_encodings(unknown_image)

            # Get the face distance between the known person and all the faces in this image
            face_distance = face_recognition.face_distance(face_encodings, known_image_encoding)[0]

            # If this face is more similar to our known image than we've seen so far, save it
            if face_distance < best_face_distance:
                # Save the new best face distance
                best_face_distance = face_distance
                # Extract a copy of the actual face image itself so we can display it
                best_face_image = unknown_image

        # Display the face image that we found to be the best match!
        pil_image = Image.fromarray(best_face_image)
        pil_image.show()

    # draw rectangle around face
    def draw_rectangle(self, image:any) -> any:
        # gee the face locations
        face_locations = self.face_locations(image)
        # check the number of faces
        faces = self.number_of_faces(image)
        if faces < 1: # no faces detected
            return None
        # draw a rectangle around all faces
        for face_location in face_locations:
            top, right, bottom, left = face_location
            cv2.rectangle(self.image, (left, top), (right, bottom), (255, 0, 0), 2)
        return self.image

    # draw landmarks around face
    def draw_landmarks(self, image:any) -> any:
        face_landmarks = self.face_landmarks(image)
        faces = self.number_of_faces(image=image)
        if faces < 1:
            return None
        for face_landmark in face_landmarks:
            for facial_feature in face_landmark.keys():
                for (x, y) in face_landmark[facial_feature]:
                    cv2.circle(self.image, (x, y), 1, (0, 0, 255), -1)
        return self.image

    # draw encodings around face
    def draw_encodings(self, image:any) -> any:
        face_encodings = self.face_encodings(image)
        faces = self.number_of_faces(image=image)
        if faces < 1:
            return None
        for face_encoding in face_encodings:
            cv2.putText(self.image, str(face_encoding), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return self.image


def main():
    image_path = 'database/people.jpg'
    # all images for database
    all_images_path = 'database/people'
    face_path = 'database/freda_org.png'
    face_recognition = FaceRecognition(image_path)
    # print the number of faces found in the image
    a=face_recognition.number_of_faces(face_recognition.image)
    print(a)
    print(f'The number of faces detected : {face_recognition.number_of_faces(face_recognition.image)}')
    if a>1:
        for face in face_recognition.faces(face_recognition.image): 
            face_recognition.face_similarity(image=face, all_images_path=all_images_path)
    # check for similar faces
    face_recognition.face_similarity(image=face_path, all_images_path=all_images_path)
    # draw rectabgle around face
    image = face_recognition.draw_rectangle(face_recognition.image)  # draw a rectangle around the faces
    # show the encoding values on image (not advisable because its an array of floating point numbers)
    # image = face_recognition.draw_encodings(face_recognition.image) 

    # use dots to indicate face landmarks (mouth, nose, eyes etc) 
    image = face_recognition.draw_landmarks(face_recognition.image) 
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
