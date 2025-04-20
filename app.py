from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import dlib
from scipy.spatial import distance as dist
from pygame import mixer
import time
import threading

# Initialize the Flask app
app = Flask(__name__, static_folder='static')

# Initialize pygame mixer for sound
mixer.init()
beep_sound = mixer.Sound("static/sounds/alarm.wav")

# Load face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Detection settings (configurable through UI)
settings = {
    "EYE_AR_THRESH": 0.25,  # Eye aspect ratio threshold
    "EYE_AR_CONSEC_FRAMES": 30,  # Number of consecutive frames for drowsiness alert
    "YAWN_THRESH": 0.6,  # Yawn threshold
    "EYE_CLOSED_TIME_THRESH": 5.0,  # Seconds of eye closure before alarm (new)
    "ALARM_ENABLED": True
}

# Global state
detection_running = True
eye_closed_start_time = None
stats = {
    "blinks": 0,
    "microsleeps": 0,
    "yawns": 0,
    "drowsy_time": 0,
    "eye_state": "Open",
    "yawn_state": "No Yawn"
}

# Counters for detection
COUNTER = 0
TOTAL_BLINKS = 0
YAWN_COUNTER = 0
TOTAL_YAWNS = 0
MICROSLEEP_COUNTER = 0
LAST_ALARM_TIME = 0
ALARM_COOLDOWN = 3  # seconds

# Calculate the eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the two sets of vertical eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    
    # Compute the euclidean distance between the horizontal eye landmarks
    C = dist.euclidean(eye[0], eye[3])
    
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

# Calculate the mouth aspect ratio (MAR) for yawn detection
def mouth_aspect_ratio(mouth):
    # Compute the euclidean distances between the vertical mouth landmarks
    A = dist.euclidean(mouth[1], mouth[7])
    B = dist.euclidean(mouth[2], mouth[6])
    C = dist.euclidean(mouth[3], mouth[5])
    
    # Compute the euclidean distance between the horizontal mouth landmarks
    D = dist.euclidean(mouth[0], mouth[4])
    
    # Compute the mouth aspect ratio
    mar = (A + B + C) / (2.0 * D)  # Changed from 3.0 to 2.0 for more sensitivity
    return mar

# Improved function to detect facial landmarks
def detect_landmarks(frame):
    global COUNTER, TOTAL_BLINKS, YAWN_COUNTER, TOTAL_YAWNS, MICROSLEEP_COUNTER, LAST_ALARM_TIME
    global eye_closed_start_time, stats
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = detector(gray)
    
    # Process each face
    for face in faces:
        # Get facial landmarks
        landmarks = predictor(gray, face)
        
        # Extract eye coordinates
        left_eye = []
        right_eye = []
        for n in range(36, 42):  # Left eye landmarks
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            left_eye.append((x, y))
            
        for n in range(42, 48):  # Right eye landmarks
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            right_eye.append((x, y))
        
        # Extract mouth coordinates (improved for better yawn detection)
        mouth = []
        for n in range(60, 68):  # Outer mouth landmarks
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            mouth.append((x, y))
        
        # Calculate eye aspect ratios
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        
        # Average the eye aspect ratio together for both eyes
        ear = (left_ear + right_ear) / 2.0
        
        # Calculate mouth aspect ratio for yawn detection
        mar = mouth_aspect_ratio(mouth)
        
        # Draw the face bounding box
        x, y = face.left(), face.top()
        w, h = face.width(), face.height()
        
        # YOLO-style bounding box for entire face with awake/drowsy classification
        confidence_awake = 0.90 if ear >= settings["EYE_AR_THRESH"] else 0.70
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"awake {confidence_awake:.2f}", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # YOLO-style bounding box for eye detection
        eye_y = y + int(h * 0.25)
        eye_h = int(h * 0.15)
        eye_box_color = (0, 255, 0)
        cv2.rectangle(frame, (x, eye_y), (x + w, eye_y + eye_h), eye_box_color, 2)
        eye_label = f"open_eye {ear:.2f}" if ear >= settings["EYE_AR_THRESH"] else f"closed_eye {ear:.2f}"
        cv2.putText(frame, eye_label, (x, eye_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, eye_box_color, 2)
        
        # YOLO-style bounding box for yawn detection
        mouth_y = y + int(h * 0.65)
        mouth_h = int(h * 0.15)
        mouth_box_color = (0, 255, 0)
        cv2.rectangle(frame, (x, mouth_y), (x + w, mouth_y + mouth_h), mouth_box_color, 2)
        mouth_label = f"no_yawn {mar:.2f}" if mar <= settings["YAWN_THRESH"] else f"yawn {mar:.2f}"
        cv2.putText(frame, mouth_label, (x, mouth_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mouth_box_color, 2)
        
        # Process eye state
        current_time = time.time()
        if ear < settings["EYE_AR_THRESH"]:
            COUNTER += 1
            
            # Update eye state stats
            stats["eye_state"] = "Closed"
            
            # Start timing eye closure if not already started
            if eye_closed_start_time is None:
                eye_closed_start_time = current_time
            
            # Check if eyes have been closed long enough for alarm (5+ seconds)
            if eye_closed_start_time is not None:
                closed_duration = current_time - eye_closed_start_time
                if closed_duration >= settings["EYE_CLOSED_TIME_THRESH"] and settings["ALARM_ENABLED"]:
                    # Only trigger alarm if cooldown has passed
                    if current_time - LAST_ALARM_TIME > ALARM_COOLDOWN:
                        beep_sound.play()
                        LAST_ALARM_TIME = current_time
                        MICROSLEEP_COUNTER += 1
                    
                    # Display warning on frame
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        else:
            # Eyes are open
            if COUNTER >= 3:  # Consider it a blink if closed for at least 3 frames
                TOTAL_BLINKS += 1
            
            COUNTER = 0
            eye_closed_start_time = None
            stats["eye_state"] = "Open"
        
        # Process yawn state with improved sensitivity
        if mar > settings["YAWN_THRESH"]:
            YAWN_COUNTER += 1
            stats["yawn_state"] = "Yawn"
    
    # Count as a yawn if detected for enough consecutive frames
            if YAWN_COUNTER >= 8:
                TOTAL_YAWNS += 1
        # Don't reset to 0, just reduce to maintain state
                YAWN_COUNTER = 4
        # Trigger alarm for yawn if enabled
                if settings["ALARM_ENABLED"]:
                    current_time = time.time()
                    if current_time - LAST_ALARM_TIME > ALARM_COOLDOWN:
                        beep_sound.play()
                        LAST_ALARM_TIME = current_time
        else:
    # More gradual decrease for smoother detection
            YAWN_COUNTER = max(0, YAWN_COUNTER - 0.5)  # Changed from 1 to 0.5
            if YAWN_COUNTER < 4:
                stats["yawn_state"] = "No Yawn"
        
        # Add EAR and MAR values to the frame
        cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Update stats
        stats["blinks"] = TOTAL_BLINKS
        stats["microsleeps"] = MICROSLEEP_COUNTER
        stats["yawns"] = TOTAL_YAWNS
        if eye_closed_start_time is not None:
            stats["drowsy_time"] = round(time.time() - eye_closed_start_time, 1)
        else:
            stats["drowsy_time"] = 0.0
    
        return frame

# Video capture generator
def generate_frames():
    cap = cv2.VideoCapture(0)
    
    # Set frame size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        if detection_running:
            try:
                # Process the frame
                processed_frame = detect_landmarks(frame)
                
                # Encode the frame for streaming
                ret, buffer = cv2.imencode('.jpg', processed_frame)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as e:
                print(f"Error processing frame: {e}")
                # Display error message on frame
                cv2.putText(frame, "Processing Error", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            # Display paused message when detection is stopped
            cv2.putText(frame, "Detection Paused", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def get_stats():
    return jsonify(stats)

@app.route('/reset_stats')
def reset_stats():
    global TOTAL_BLINKS, TOTAL_YAWNS, MICROSLEEP_COUNTER
    TOTAL_BLINKS = 0
    TOTAL_YAWNS = 0
    MICROSLEEP_COUNTER = 0
    
    # Reset the stats dict
    stats["blinks"] = 0
    stats["microsleeps"] = 0
    stats["yawns"] = 0
    stats["drowsy_time"] = 0
    
    return jsonify({"status": "success"})

@app.route('/update_settings', methods=['POST'])
def update_settings():
    try:
        data = request.get_json()
        settings["EYE_AR_THRESH"] = float(data.get("eye_threshold", settings["EYE_AR_THRESH"]))
        settings["EYE_AR_CONSEC_FRAMES"] = int(data.get("consecutive_frames", settings["EYE_AR_CONSEC_FRAMES"]))
        settings["YAWN_THRESH"] = float(data.get("yawn_threshold", settings["YAWN_THRESH"]))
        settings["ALARM_ENABLED"] = bool(data.get("alarm_enabled", settings["ALARM_ENABLED"]))
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/start_detection', methods=['POST'])
def start_detection():
    global detection_running
    detection_running = True
    return jsonify({"status": "success"})

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    global detection_running
    detection_running = False
    return jsonify({"status": "success"})

@app.route('/test_alarm', methods=['POST'])
def test_alarm():
    beep_sound.play()
    return jsonify({"status": "success"})

@app.route('/toggle_alarm', methods=['POST'])
def toggle_alarm():
    settings["ALARM_ENABLED"] = request.form.get('enabled') == 'true'
    return jsonify({"status": "success"})

@app.route('/trigger_alarm', methods=['POST'])
def trigger_alarm():
    current_time = time.time()
    global LAST_ALARM_TIME
    if current_time - LAST_ALARM_TIME > ALARM_COOLDOWN and settings["ALARM_ENABLED"]:
        beep_sound.play()
        LAST_ALARM_TIME = current_time
    return jsonify({"status": "success"})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8501)