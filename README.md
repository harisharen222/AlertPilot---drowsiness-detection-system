# DriveSafe AI: Driver Drowsiness Detection System

A real-time drowsiness detection system utilizing deep learning, computer vision, and facial landmarks to enhance road safety by identifying signs of driver fatigue and alerting accordingly.


## 🚀 Features

- **Real-time Face & Eye Monitoring**: Capture and analyze video feed from webcam or mobile camera
- **Dual Detection Approach**: 
  - Facial landmark-based detection with dlib (68-point model)
  - Deep learning-based classification with MobileNetV2
- **Comprehensive Drowsiness Indicators**:
  - Eye Aspect Ratio (EAR) monitoring
  - Mouth Aspect Ratio (MAR) for yawn detection
  - Temporal analysis for microsleep detection
- **Alert System**: Audio alerts when drowsiness is detected
- **Web Interface**: Flask-based dashboard for visualization and configuration
- **Statistics Tracking**: Monitor blinks, yawns, and microsleep events

## 📋 System Requirements

- Python 3.7 or higher
- Webcam or mobile camera access
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **Computer Vision**: OpenCV, dlib
- **Deep Learning**: TensorFlow, TensorFlow Hub, Keras
- **Model Architecture**: MobileNetV2
- **Frontend**: HTML, CSS, JavaScript
- **Audio**: Pygame

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.7+ installed
- pip (Python package manager)
- Virtual environment (optional but recommended)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/DriveSafe-AI.git
cd DriveSafe-AI
```

### Step 2: Create and Activate Virtual Environment (Optional)
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Facial Landmark Predictor
Download the dlib facial landmark predictor file from:
http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

Extract and place it in the project root directory.

### Step 5: Set Up Audio Alert
Ensure `alarm.wav` is in the `static/sounds/` directory.

## 🚀 Usage

### Option 1: Using the Direct Facial Landmark Approach (app.py)
```bash
python app.py
```

### Option 2: Using the MobileNetV2 Model (app2.py)
```bash
python app2.py
```

### Accessing the Web Interface
Open your browser and navigate to:
```
http://127.0.0.1:8501/
```
or
```
http://localhost:8501/
```

## 🧠 System Workflow

1. **Video Capture**: System captures video frames from webcam/camera
2. **Face Detection**: Identifies face regions in each frame
3. **Facial Landmark Detection**: Extracts 68 facial landmarks using dlib
4. **Feature Extraction**:
   - Calculates Eye Aspect Ratio (EAR) from eye landmarks
   - Calculates Mouth Aspect Ratio (MAR) from mouth landmarks
5. **State Classification**:
   - **Approach 1**: Threshold-based detection (EAR < 0.25 indicates closed eyes)
   - **Approach 2**: MobileNetV2 classification ("Open" vs "Closed" states)
6. **Temporal Analysis**: Tracks duration of closed eye state
7. **Alert Generation**: Triggers audio alert if drowsiness conditions met
8. **Dashboard Update**: Updates UI with current statistics and state

## 📊 Model Details

### Facial Landmark Detection
- **Model**: dlib's 68-point facial landmark predictor
- **Functionality**: Identifies key facial points around eyes, mouth, nose, and jawline
- **Performance**: Real-time detection at ~30 FPS on modern hardware

### MobileNetV2 (Optional Approach)
- **Architecture**: MobileNetV2 with transfer learning
- **Input Size**: 224x224x3 RGB images
- **Output**: Binary classification (Eyes Open/Closed)
- **Training Dataset**: Combination of NTHU-DDD and UTA-RLDD datasets
- **Accuracy**: ~94% on test dataset

## 📁 Project Structure

```
DriveSafe-AI/
├── app.py                           # Main application using facial landmarks
├── app2.py                          # Alternative approach using MobileNetV2
├── requirements.txt                 # Project dependencies
├── shape_predictor_68_face_landmarks.dat  # Facial landmark predictor
├── model/                           # Directory containing trained models
│   └── mobilenetv2-model.h5         # MobileNetV2 pre-trained model
├── static/                          # Static files for web interface
│   ├── css/                         # CSS stylesheets
│   ├── js/                          # JavaScript files
│   └── sounds/                      # Alert sound files
│       └── alarm.wav                # Audio alert file
├── templates/                       # HTML templates
│   └── index.html                   # Main dashboard template
└── README.md                        # Project documentation
```

## 🧪 Testing

### Functionality Testing
1. **Normal State Detection**: Test with normal blinking patterns
2. **Drowsiness Detection**: Test with prolonged eye closure
3. **Yawn Detection**: Test with yawning
4. **Variable Lighting**: Test under different lighting conditions

### Performance Testing
1. **Frame Rate**: Monitor FPS under different conditions
2. **CPU Usage**: Track system resource utilization
3. **Memory Usage**: Monitor RAM consumption during operation


## 🛠️ Future Improvements

1. **Multi-person Detection**: Support for monitoring multiple faces simultaneously
2. **Integration with Vehicle Systems**: Connect to vehicle control systems
3. **Mobile Application**: Develop mobile app version for standalone use
4. **Enhanced Alert System**: Add vibration and visual alerts
5. **Cloud Integration**: Support for cloud-based model updates and statistics

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

