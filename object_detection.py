import cv2
from ultralytics import YOLO
import pyttsx3
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
        return float('inf')

    @staticmethod
    def get_direction(x1, x2, frame_width):
        object_center = (x1 + x2) / 2
        if object_center < frame_width / 3:
            return "left"
        elif object_center > 2 * frame_width / 3:
            return "right"
        return "center"

class ObjectDetectionApp:
    def __init__(self, model_path):
        self.detector = ObjectDetection(model_path)
        self.speaker = pyttsx3.init()
        self.speaker.setProperty('rate', 150)
        self.listener = sr.Recognizer()
        self.previous_objects = set()

    def speak(self, message):
        self.speaker.say(message)
        self.speaker.runAndWait()

    def listen_for_exit(self):
        """Listen for the word 'exit' to stop object detection and return to main module."""
        with sr.Microphone() as source:
            print("Listening for 'exit' to stop detection...")
            self.listener.adjust_for_ambient_noise(source)
            try:
                audio = self.listener.listen(source, timeout=5, phrase_time_limit=3)
                command = self.listener.recognize_google(audio).lower()
                print(f"Command heard: {command}")
                if "exit" in command:
                    return True
            except Exception as ex:
                print(f"Error: {ex}")
            return False

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Camera not accessible.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

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

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{object_name} {distance:.2f}m {direction}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            unique_objects = set(detected_objects.keys())
            if unique_objects != self.previous_objects:
                new_objects = unique_objects - self.previous_objects
                if new_objects:
                    object_list = ', '.join(new_objects)
                    self.speak(f"Detected objects: {object_list}.")
                    for obj, details in detected_objects.items():
                        for distance, direction in details:
                            self.speak(f"The {obj} is {direction} and is {distance:.2f} meters away.")
                self.previous_objects = unique_objects

            cv2.imshow("Object Detection with Direction", frame)

            if self.listen_for_exit():
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
