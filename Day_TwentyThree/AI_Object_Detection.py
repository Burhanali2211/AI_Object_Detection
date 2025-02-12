import cv2
import torch
import numpy as np
import threading
from ultralytics import YOLO

# Load pre-trained YOLOv8 Small model (more accurate than Nano)
model = YOLO("yolov8s.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Ensure webcam opens properly
if not cap.isOpened():
    print("Error: Couldn't access the webcam.")
    exit()

# Function to process frame


def process_frame():
    global frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to NumPy array for faster processing
        frame_np = np.array(frame, dtype=np.uint8)

        # Run YOLOv8 model on frame
        # Disable print output for speed
        results = model(frame_np, verbose=False)

        # Extract bounding boxes, confidence scores, and class labels
        for result in results:
            for box in result.boxes:
                # Bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = round(box.conf[0].item(), 2)  # Confidence score
                cls = int(box.cls[0].item())  # Class index
                label = f"{model.names[cls]} {conf:.2f}"

                # Draw bounding box and label using OpenCV optimizations
                cv2.rectangle(frame_np, (x1, y1), (x2, y2),
                              (0, 255, 0), 2, lineType=cv2.LINE_AA)
                cv2.putText(frame_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2, lineType=cv2.LINE_AA)

        # Show frame
        cv2.imshow("Object Detection By DevelopersMindset", frame_np)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# Run detection in a separate thread for faster performance
thread = threading.Thread(target=process_frame)
thread.start()

thread.join()
cap.release()
cv2.destroyAllWindows()
