import cv2
from ultralytics import YOLO

# Load model
print("Loading YOLO model...")
model = YOLO('yolov8n.pt')
print("Model loaded!")

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Webcam not found!")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Counting line position (middle of screen)
LINE_Y = 240
CROSSED_OBJECTS = []
COUNT = 0
object_tracker = {}
next_id = 0

# Target classes: 0=person, 2=car, 5=bus, 7=truck
TARGET_CLASSES = [0, 2, 5, 7]

print("\n📸 Webcam started!")
print(f"📏 Counting objects crossing line at Y={LINE_Y}")
print("🎯 Targets: Person, Car, Bus, Truck")
print("❌ Press 'q' to quit\n")

while True:
    success, frame = cap.read()
    if not success:
        break
    
    # Run detection
    results = model(frame, stream=True)
    display_frame = frame.copy()
    
    # Draw counting line
    cv2.line(display_frame, (0, LINE_Y), (640, LINE_Y), (0, 255, 0), 3)
    cv2.putText(display_frame, f"COUNT: {COUNT}", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Process detections
    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Only track target classes with good confidence
                if class_id in TARGET_CLASSES and confidence > 0.5:
                    # Draw bounding box
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{model.names[class_id]} {confidence:.2f}"
                    cv2.putText(display_frame, label, (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Draw center point
                    cv2.circle(display_frame, (center_x, center_y), 4, (0, 0, 255), -1)
                    
                    # ---- COUNTING LOGIC ----
                    # Track object
                    matched_id = None
                    for obj_id, (prev_x, prev_y) in object_tracker.items():
                        if abs(prev_x - center_x) < 30 and abs(prev_y - center_y) < 30:
                            matched_id = obj_id
                            break
                    
                    if matched_id is None:
                        object_tracker[next_id] = (center_x, center_y)
                        matched_id = next_id
                        next_id += 1
                    
                    prev_x, prev_y = object_tracker[matched_id]
                    
                    # Check if crossed the line
                    if matched_id not in CROSSED_OBJECTS:
                        if (prev_y < LINE_Y and center_y >= LINE_Y) or (prev_y > LINE_Y and center_y <= LINE_Y):
                            COUNT += 1
                            CROSSED_OBJECTS.append(matched_id)
                            print(f"✅ Object {matched_id} crossed! Total: {COUNT}")
                    
                    # Update position
                    object_tracker[matched_id] = (center_x, center_y)
    
    # Show the frame
    cv2.imshow('Task 4: Object Detection + Counting Zone', display_frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print(f"\n📊 Final Count: {COUNT} objects crossed the line!")