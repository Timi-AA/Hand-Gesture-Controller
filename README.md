# Hand-Gesture-Controller
A real-time computer vision application that lets you control your computer using hand gestures captured through your webcam, no mouse, no keyboard, just your hand.
Built with Python, OpenCV, and MediaPipe's hand landmark detection, the system tracks 21 points on your hand every frame, classifies your gesture using a rule-based engine, and maps it to a system action instantly.

# Gestures
GestureActionBehaviour✋ Open PalmPlay / PauseFires once✊ Closed FistMute / UnmuteFires once✌️ Peace SignVolume UpRepeats while held👍 Thumbs UpVolume DownRepeats while held☝️ One Finger UpNext TrackFires once

#How it works
MediaPipe detects 21 landmark coordinates on your hand in real time. A custom classifier analyses finger extension and curl depth to identify which gesture you're making. A smoothing buffer requires the gesture to be consistent across multiple frames before triggering, eliminating false positives. Volume gestures repeat automatically while held so you don't have to reposition your hand.

# Features

Real-time hand landmark overlay with bounding box
Gesture label displayed directly above your hand
Confidence loading bar shows detection progress
Hold-to-repeat for volume gestures
Smooth FPS counter
On-screen legend showing all gesture mappings
Auto-downloads the MediaPipe model on first run


# Stack

Python 3.8+
OpenCV — webcam capture and frame rendering
MediaPipe Tasks API — hand landmark detection
pyautogui — system media key control


# Getting started
bashpip install opencv-python mediapipe pyautogui
python hand_gesture_controller.py
The model file (~5 MB) downloads automatically on first launch. Press Q or ESC to exit.
