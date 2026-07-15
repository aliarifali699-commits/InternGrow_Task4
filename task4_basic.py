import cv2
from ultralytics import YOLO

# Load pre-trained YOLOv8 model
print("Loading YOLO model...")
model = YOLO('yolov8n.pt')
print("Model loaded successfully!")

# Open webcam
cap = cv2.VideoCapture(0)

# Check if webcam opened
if not cap.isOpened():
    print("❌ Error: Could not open webcam!")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\n✅ Press 'q' to quit")
print("📸 Webcam is starting...")

while True:
    success, frame = cap.read()
    if not success:
        print("❌ Failed to capture frame")
        break
    
    # Run detection
    results = model(frame, stream=True)
    
    # Draw results
    for r in results:
        annotated_frame = r.plot()
    
    # Show the frame
    cv2.imshow('YOLOv8 Object Detection - Task 4', annotated_frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("\n✅ Program closed successfully!")