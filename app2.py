from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from pygame import mixer
import tensorflow_hub as hub  # Import TensorFlow Hub for custom KerasLayer
import time  # For cooldown mechanism
from collections import deque  # For smoothing predictions

# Initialize the Flask app
app = Flask(__name__)

# Replace with the URL of your mobile camera's stream
MOBILE_CAMERA_URL = "http://192.168.1.55:8080/video"  # Replace with your IP webcam URL

# Model path
MODEL_PATH = "C:/Users/vatsa/Downloads/Driver_drowsiness_detection/DriveSafe-AI-Driver-Monitoring-System/model/20250103-2118-full-image-set-mobilenetv2-Adam.h5"
# Load model with the custom KerasLayer
model = load_model(MODEL_PATH, custom_objects={'KerasLayer': hub.KerasLayer})

# Labels for the model's output
labels = ['Closed', 'Open']

# Initialize pygame mixer for sound
mixer.init()
beep_sound = mixer.Sound("C:/Users/vatsa/Downloads/Driver_drowsiness_detection/DriveSafe-AI-Driver-Monitoring-System/alarm.wav")  # Replace with the path to your beep sound file

# Load Haar cascade for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Cooldown variables for the beep sound
last_alert_time = 0
alert_cooldown = 1  # seconds

# Initialize a deque for smoothing predictions
predictions = deque(maxlen=5)

# Function to preprocess the image
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = cv2.resize(image, (224, 224))  # Resize to match the input size of MobilenetV2
    image = image / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to detect eyes and return the region of interest
def detect_roi(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in eyes:
        return frame[y:y+h, x:x+w]  # Return the first detected eye region
    return None

# Smooth predictions using a rolling window
def smooth_predictions(new_prediction):
    predictions.append(new_prediction)
    return max(set(predictions), key=predictions.count)

# Video capture generator
def generate_frames():
    cap = cv2.VideoCapture(MOBILE_CAMERA_URL)
    #cap = cv2.VideoCapture(0)


    if not cap.isOpened():
        print("Unable to access the mobile camera stream. Check the URL and connection.")

    global last_alert_time

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Detect ROI dynamically using eye detection
        roi = detect_roi(frame)
        if roi is not None:
            # Preprocess the region of interest
            processed_roi = preprocess_image(roi)

            try:
                # Predict the state of the eyes
                prediction = model.predict(processed_roi)
                state = labels[np.argmax(prediction)]
                confidence = np.max(prediction)

                # Smooth predictions
                state = smooth_predictions(state)

                # Display the prediction on the frame
                cv2.putText(frame, f"State: {state} ({confidence:.2f})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Trigger beep if eyes are closed and cooldown has passed
                if state == 'Closed' and confidence > 0.8 and (time.time() - last_alert_time) > alert_cooldown:
                    beep_sound.play()
                    last_alert_time = time.time()

            except Exception as e:
                cv2.putText(frame, "Prediction Error", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                print(f"Prediction failed: {e}")

        else:
            # Indicate no face or eye was detected
            cv2.putText(frame, "No Eye Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Encode the frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
