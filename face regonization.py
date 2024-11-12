import cv2
import face_recognition
import os
import pyttsx3
import speech_recognition as sr

class FaceRecognitionApp:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.speaker = pyttsx3.init()
        self.speaker.setProperty('rate', 150)
        self.listener = sr.Recognizer()

    def load_encodings(self):
        names, encoded_images = [], []
        for filename in os.listdir(self.folder_path):
            fullpath = os.path.join(self.folder_path, filename)
            if os.path.isfile(fullpath):
                name_without_extension = os.path.splitext(filename)[0]
                image = face_recognition.load_image_file(fullpath)
                image_encodings = face_recognition.face_encodings(image)
                if image_encodings:
                    encoded_images.append(image_encodings[0])
                    names.append(name_without_extension)
        return names, encoded_images

    def speak(self, text):
        self.speaker.say(text)
        self.speaker.runAndWait()

    def listen_for_exit(self):
        with sr.Microphone() as source:
            print("Listening for 'exit' to stop recognition...")
            self.listener.adjust_for_ambient_noise(source)
            try:
                audio = self.listener.listen(source, timeout=5, phrase_time_limit=3)
                command = self.listener.recognize_google(audio).lower()
                if "exit" in command:
                    return True
            except Exception as ex:
                print(f"Error: {ex}")
            return False

    def run(self):
        names, encoded_images = self.load_encodings()
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(img_rgb)
            face_encodings = face_recognition.face_encodings(img_rgb, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(encoded_images, face_encoding)
                name = "Unknown"
                if True in matches:
                    first_match_index = matches.index(True)
                    name = names[first_match_index]
                    self.speak(f"Detected {name}.")

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.imshow("Face Recognition", frame)

            if self.listen_for_exit():
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
