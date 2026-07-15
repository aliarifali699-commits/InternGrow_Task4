# 🎯 Task 4: Real-Time Multi-Object Detection & Analytics

## InternGrow AI Track – Internship Program

---

### 📌 Project Overview

This project implements a **Real-Time Multi-Object Detection & Analytics System** using YOLOv8, a state-of-the-art deep learning model for computer vision. The system processes live webcam footage to detect and classify multiple objects simultaneously with high accuracy.

The system draws bounding boxes around detected objects, displays class labels with confidence scores, and provides real-time visual feedback. As an upgrade feature, an **automated counting zone** was implemented that tracks specific objects (people, cars, buses, and trucks) crossing a defined boundary line, updating a live counter on the screen.

---

### ✨ Features

| Feature | Description |
|---------|-------------|
| **Real-Time Detection** | Detects multiple objects from live webcam feed at 250-370ms per frame |
| **80+ Object Classes** | Recognizes people, vehicles, animals, furniture, electronics, and more |
| **Visual Feedback** | Green bounding boxes with class labels and confidence scores |
| **Counting Zone (Upgrade)** | Tracks people, cars, buses, and trucks crossing a virtual boundary line |
| **Live Counter** | Real-time count display on the screen |
| **Unique Object Tracking** | Each object assigned a unique ID to prevent double-counting |
| **Console Logging** | Every crossing event is logged in the terminal |

---

### 🛠️ Technologies Used

| Tool/Technology | Version | Purpose |
|-----------------|---------|---------|
| **Python** | 3.14 | Primary programming language |
| **YOLOv8 (Ultralytics)** | 8.0+ | Pre-trained object detection model |
| **OpenCV (cv2)** | 4.8.0+ | Webcam capture, image processing, and display |
| **NumPy** | 1.24.0+ | Numerical operations and array handling |
| **Windows 10** | 19045.6466 | Operating system |
| **GitHub** | - | Source code repository |

---

### 📂 Project Structure
