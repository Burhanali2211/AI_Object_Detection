# AI-Based Object Detection for Home Security

## Overview
This project implements real-time **AI-based object detection** using **YOLOv8** for home security. It detects objects, humans, and animals using a webcam and provides fast and accurate results.

## Features
✅ **Real-time object detection** using YOLOv8.  
✅ **Optimized for speed** with NumPy and multi-threading.  
✅ **Accurate predictions** using `yolov8s.pt`.  
✅ **Displays detected objects** with confidence scores.  
✅ **Runs smoothly on mid-range laptops & PCs**.

## Requirements
Make sure you have **Python 3.8+** installed, then install dependencies:
```bash
pip install opencv-python torch torchvision torchaudio ultralytics numpy
```

## Usage
Run the following command to start the object detection system:
```bash
python Object_Detection_for_Home_Security.py
```

## How It Works
1. Loads the **YOLOv8-Small (`yolov8s.pt`)** model.
2. Captures video from the **webcam**.
3. Uses **NumPy** for fast frame processing.
4. Runs YOLOv8 to **detect objects in real-time**.
5. Draws **bounding boxes & labels** for detected objects.
6. Press **'q'** to exit the detection system.

## Code
```python
import cv2
import torch
import numpy as np
import threading
from ultralytics import YOLO

# Load YOLOv8 Small model (better accuracy than Nano)
model = YOLO("yolov8s.pt")

# Open webcam
cap = cv2.VideoCapture(0)
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
        
        # Convert frame to NumPy array for fast processing
        frame_np = np.array(frame, dtype=np.uint8)
        
        # Run YOLOv8 on frame
        results = model(frame_np, verbose=False)  # Disable logs for speed
        
        # Extract detected objects
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
                conf = round(box.conf[0].item(), 2)  # Confidence score
                cls = int(box.cls[0].item())  # Class index
                label = f"{model.names[cls]} {conf:.2f}"
                
                # Draw bounding box & label
                cv2.rectangle(frame_np, (x1, y1), (x2, y2), (0, 255, 0), 2, lineType=cv2.LINE_AA)
                cv2.putText(frame_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, lineType=cv2.LINE_AA)
        
        # Show frame
        cv2.imshow("AI Object Detection", frame_np)
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Run detection in a separate thread for better performance
thread = threading.Thread(target=process_frame)
thread.start()
thread.join()
cap.release()
cv2.destroyAllWindows()
```

## Notes
- If accuracy is not enough, you can replace `yolov8s.pt` with `yolov8m.pt` for a better model.
- If running on a **low-end PC**, use `yolov8n.pt` for a lighter version.

## Future Improvements
🚀 Add **motion detection** to trigger alerts.  
🚀 Implement **email or SMS notifications** for security alerts.  
🚀 Save **detected objects with timestamps** for logging purposes.

## License
This project is open-source under the **MIT License**.

