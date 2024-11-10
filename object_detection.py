import cv2
from ultralytics import YOLO
import numpy as np
import pyttsx3  # Replacing naturalreader with pyttsx3
import speech_recognition as sr

# Initialize pyttsx3 for text-to-speech
engine = pyttsx3.init()

# Load the YOLOv8 model
model = YOLO('yolov8l.pt')  # Ensure you have a compatible YOLO model file

# Open a video capture for the webcam
cap = cv2.VideoCapture(0)

# Constants for distance and focal length calculation
real_object_width = 0.2  # Approximate width of a typical object (in meters)
known_distance = 1.0  # Distance from camera to object for calibration (in meters)
known_width_pixels = 200  # Measured object width in pixels at the known distance
focal_length = (known_width_pixels * known_distance) / real_object_width


# Helper function to get direction based on bounding box position
def get_direction(x1, x2, frame_width):
    object_center = (x1 + x2) / 2
    if object_center < frame_width / 3:
        return "left"
    elif object_center > 2 * frame_width / 3:
        return "right"
    else:
        return "center"


# Initialize variables to track announced objects and changes
previous_objects = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get frame dimensions
    frame_height, frame_width = frame.shape[:2]

    # Perform inference with YOLO
    results = model(frame, conf=0.5, iou=0.4)
    result = results[0]
    boxes = result.boxes
    names = result.names

    # Dictionary to store detected objects and their distances/directions
    detected_objects = {}
    unique_objects = set()

    # Process each detected object
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        label = int(box.cls[0])  # Object class ID
        object_name = names[label]  # Object class name
        confidence = box.conf[0]

        # Calculate distance using the width of the bounding box
        object_width_pixels = x2 - x1
        if object_width_pixels > 0:
            distance = (real_object_width * focal_length) / object_width_pixels
        else:
            distance = float('inf')

        # Get the object's direction
        direction = get_direction(x1, x2, frame_width)

        # Store object data in dictionary
        if object_name in detected_objects:
            detected_objects[object_name].append((distance, direction))
        else:
            detected_objects[object_name] = [(distance, direction)]

        # Add object name to the unique list
        unique_objects.add(object_name)

    # Announce detected objects if there are new objects
    if unique_objects != previous_objects:
        object_list = ', '.join(unique_objects)
        announcement = f"Detected objects are: {object_list}."
        engine.say(announcement)  # Speak with pyttsx3
        engine.runAndWait()
        print(announcement)
        previous_objects = unique_objects  # Update previous_objects

    # Speech recognition to replace manual input
    recognizer = sr.Recognizer()

    # Record audio using the microphone
    with sr.Microphone() as source:
        print("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Listening for your query...")

        # Listen for a query (timeout set to 15 seconds)
        try:
            recorded_audio = recognizer.listen(source, timeout=15, phrase_time_limit=15)
            print("Done recording")

            # Try to recognize the audio
            print("Recognizing the text...")
            query = recognizer.recognize_google(recorded_audio, language="en-US")
            print(f"Decoded Text: {query}")

        except Exception as ex:
            print("Error: ", ex)
            query = ""  # No query if there's an error

    if query:
        # NLP to check if the query contains object names
        query_lower = query.lower()

        # Check if the user wants to hear the object names again
        if "repeat objects" in query_lower or "what objects" in query_lower:
            announcement = f"Detected objects are: {', '.join(unique_objects)}."
            engine.say(announcement)  # Speak with pyttsx3
            engine.runAndWait()
            print(announcement)
            continue

        # Check if the user asked for a specific object
        for object_name in detected_objects:
            if object_name.lower() in query_lower:
                # Find the nearest instance if multiple are detected
                nearest_object = min(detected_objects[object_name], key=lambda x: x[0])
                distance, direction = nearest_object
                response = f"The {object_name} is {direction} and is {distance:.2f} meters away."
                engine.say(response)  # Speak with pyttsx3
                engine.runAndWait()
                print(response)
                break
        else:
            print("Object not found in the current frame.")

    # Display the annotated frame
    frame_with_results = result.plot()
    cv2.imshow("Object Detection with Direction", frame_with_results)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
