import cv2
from ultralytics import YOLO
import numpy as np
import pyttsx3  # Text-to-speech
import speech_recognition as sr

class ObjectDetection:
    def __init__(self, model_path, real_object_width=0.2, known_distance=1.0, known_width_pixels=200):
        self.model = YOLO(model_path)
        self.real_object_width = real_object_width
        self.focal_length = (known_width_pixels * known_distance) / real_object_width

    def detect_objects(self, frame):
        results = self.model(frame, conf=0.5, iou=0.4)
        return results[0]

    def calculate_distance(self, object_width_pixels):
        if object_width_pixels > 0:
            return (self.real_object_width * self.focal_length) / object_width_pixels
        else:
            return float('inf')

    @staticmethod
    def get_direction(x1, x2, frame_width):
        object_center = (x1 + x2) / 2
        if object_center < frame_width / 3:
            return "left"
        elif object_center > 2 * frame_width / 3:
            return "right"
        else:
            return "center"

class SpeechHandler:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)

    def announce(self, message):
        self.engine.say(message)
        self.engine.runAndWait()

class SpeechRecognizer:
    def __init__(self):
        self.recognizer = sr.Recognizer()

    def listen_query(self):
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            print("Listening for your query...")
            try:
                recorded_audio = self.recognizer.listen(source, timeout=15, phrase_time_limit=15)
                print("Recognizing the text...")
                query = self.recognizer.recognize_google(recorded_audio, language="en-US")
                return query.lower()
            except Exception as ex:
                print("Error: ", ex)
                return ""

class ObjectDetectionApp:
    def __init__(self, model_path):
        self.detector = ObjectDetection(model_path)
        self.speaker = SpeechHandler()
        self.listener = SpeechRecognizer()
        self.previous_objects = set()
        self.recognized_objects = set()

    def process_frame(self, frame):
        frame_height, frame_width = frame.shape[:2]
        result = self.detector.detect_objects(frame)
        detected_objects = {}

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = int(box.cls[0])
            object_name = result.names[label]
            object_width_pixels = x2 - x1
            distance = self.detector.calculate_distance(object_width_pixels)
            direction = self.detector.get_direction(x1, x2, frame_width)

            if object_name in detected_objects:
                detected_objects[object_name].append((distance, direction))
            else:
                detected_objects[object_name] = [(distance, direction)]

        return detected_objects, frame_width

    def announce_new_objects(self, detected_objects):
        unique_objects = set(detected_objects.keys())
        if unique_objects != self.previous_objects:
            new_objects = unique_objects - self.recognized_objects
            if new_objects:
                object_list = ', '.join(new_objects)
                announcement = f"Detected new objects: {object_list}."
                self.speaker.announce(announcement)
                print(announcement)
                self.recognized_objects.update(new_objects)
            self.previous_objects = unique_objects

    def respond_to_query(self, query, detected_objects):
        if "repeat objects" in query or "what objects" in query:
            object_list = ', '.join(self.previous_objects)
            self.speaker.announce(f"Currently detected objects are: {object_list}.")
            return

        for object_name in detected_objects:
            if object_name.lower() in query:
                nearest_object = min(detected_objects[object_name], key=lambda x: x[0])
                distance, direction = nearest_object
                response = f"The {object_name} is {direction} and is {distance:.2f} meters away."
                self.speaker.announce(response)
                print(response)
                return
        print("Object not found in the current frame.")

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Camera not accessible.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detected_objects, frame_width = self.process_frame(frame)
            self.announce_new_objects(detected_objects)
            frame_with_results = detected_objects.plot()
            cv2.imshow("Object Detection with Direction", frame_with_results)

            query = self.listener.listen_query()
            if query:
                self.respond_to_query(query, detected_objects)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Running the app
if __name__ == "__main__":
    app = ObjectDetectionApp('yolov8l.pt')
    app.run()
