# ✋ Gesture_Craft 

Gesture-Craft is an advanced computer vision–based application that enables users to interact with a system using hand gestures instead of traditional input devices like a mouse or keyboard. The project uses real-time hand tracking to provide a smooth, touchless, and interactive user experience.

---

## 🚀 Features

### 🎨 Gesture-Based Drawing
- Draw on a virtual canvas using **index + middle fingers**
- Smooth drawing using motion filtering and stabilization
- Real-time hand tracking with high accuracy

### 🧽 Eraser Mode
- Activate eraser using **open palm gesture**
- Adjustable eraser size for better control

### 🎥 YouTube Gesture Control
Activate using **thumb + pinky gesture (hold)**:
- 👉 Index + Middle → Next Video  
- 👈 Thumb + Index + Middle → Previous Video  
- ✋ Four Fingers → Scroll  
- 👍 Thumbs Up → Volume Up  
- 🤙 Pinky Up → Volume Down  
- 🖐 Open Palm → Open YouTube  

### 🧠 AI Features
- Basic color-based object detection (Red, Green, Blue)
- Smart drawing suggestions for creativity
- Predefined shape drawing (circle, square, triangle, star, etc.)

### ⌨️ Typing Mode
- Search queries directly on Google
- Draw shapes by typing their names
- Toggle typing mode using keyboard

### 🔊 Feedback System
- Audio feedback (beeps) for gesture recognition
- Visual indicators for actions and modes

---

## 🛠️ Technologies Used

- **Python** – Core programming language  
- **OpenCV** – Image processing and video capture  
- **MediaPipe** – Hand tracking and landmark detection  
- **NumPy** – Numerical computations  
- **Threading & Math Libraries** – Performance and calculations  

---

## 📦 Installation

### 1. Clone Repository
```bash
git clone https://github.com/PiyushSalunke007/gesture_Craft.git
cd gesture_Craft
2. Install Dependencies
pip install opencv-python mediapipe numpy
3. Run the Project
python hand_gesture.py
🎮 Controls Summary
Action	Gesture
Draw	Index + Middle Fingers
Erase	Open Palm
YouTube Mode	Thumb + Pinky (hold)
Next Video	Index + Middle
Previous Video	Thumb + Index + Middle
Volume Up	Thumbs Up
Volume Down	Pinky Up
💡 Use Cases
Digital drawing and sketching

Touchless human-computer interaction

Smart classroom systems

Accessibility tools for physically challenged users

Gesture-based control systems

⚠️ Requirements
Webcam required

Good lighting conditions for better accuracy

Windows OS recommended (for sound feedback)

🔮 Future Improvements
Integration with advanced AI/ML models

Custom gesture configuration

Mobile and web-based version

Voice + gesture hybrid system

Cloud-based processing

👨‍💻 Author
Piyush Salunke