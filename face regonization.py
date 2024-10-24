import cv2
import face_recognition
import os

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Path to the folder containing images
folder_path = r"E:\PROJECT\COMPUTER VISION\Images"

# Lists to store names and encodings
names = []
encoded_images = []

# Encoding all the images
for filename in os.listdir(folder_path):
    fullpath = os.path.join(folder_path, filename)
    if os.path.isfile(fullpath):
        name_without_extension = os.path.splitext(filename)[0]
        image = face_recognition.load_image_file(fullpath)
        image_encodings = face_recognition.face_encodings(image)
        if image_encodings:
            # Assuming one face per image
            encoded_images.append(image_encodings[0])
            names.append(name_without_extension)

print(names)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(img_rgb)
    face_encodings = face_recognition.face_encodings(img_rgb, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Draw a rectangle around the face
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

        # Compare the current face encoding with all known face encodings
        matches = face_recognition.compare_faces(encoded_images, face_encoding)
        name = "Unknown"

        # Find the best match
        if True in matches:
            first_match_index = matches.index(True)
            name = names[first_match_index]

        # Add text (name) above the rectangle
        cv2.putText(img, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Face", img)
    key = cv2.waitKey(1)
    if key == 27:  # Esc key to exit
        break

cap.release()
cv2.destroyAllWindows()
