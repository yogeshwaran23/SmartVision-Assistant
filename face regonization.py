# face_recognition_module.py

import cv2
import face_recognition
import os
import pyttsx3


def initialize_camera():
    """Initialize the video capture from the default camera."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not accessible.")
        return None
    return cap


def load_encodings(folder_path):
    """Load and encode images from a specified folder path."""
    names = []
    encoded_images = []
    for filename in os.listdir(folder_path):
        fullpath = os.path.join(folder_path, filename)
        if os.path.isfile(fullpath):
            name_without_extension = os.path.splitext(filename)[0]
            try:
                image = face_recognition.load_image_file(fullpath)
                image_encodings = face_recognition.face_encodings(image)
                if image_encodings:
                    # Assuming one face per image
                    encoded_images.append(image_encodings[0])
                    names.append(name_without_extension)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    return names, encoded_images


def initialize_text_to_speech(rate=150):
    """Initialize the text-to-speech engine with a specific speech rate."""
    engine = pyttsx3.init()
    engine.setProperty('rate', rate)
    return engine


def recognize_faces(cap, names, encoded_images, engine):
    """Perform face recognition and announce recognized faces using TTS."""
    announced_names = set()

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to grab frame")
            break

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(img_rgb)
        face_encodings = face_recognition.face_encodings(img_rgb, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            matches = face_recognition.compare_faces(encoded_images, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = names[first_match_index]
                if name not in announced_names:
                    engine.say(f"person name is , {name}")
                    engine.runAndWait()
                    announced_names.add(name)

            cv2.putText(img, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Face", img)
        key = cv2.waitKey(1)
        if key == 27:  # Esc key to exit
            break

    cap.release()
    cv2.destroyAllWindows()
