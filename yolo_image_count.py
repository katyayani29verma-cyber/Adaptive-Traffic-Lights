import cv2
import csv
from datetime import datetime
from ultralytics import YOLO
import os

# -------------------- CONFIG --------------------
IMAGE_PATH = "images"          # folder containing images
OUTPUT_CSV = "traffic_data.csv"
MODEL_PATH = "yolov8n.pt"      # YOLOv8 nano model
VEHICLE_CLASSES = [2, 3, 5, 7] # car, motorcycle, bus, truck (COCO)
# ------------------------------------------------

# Load YOLO model
model = YOLO(MODEL_PATH)

# Create CSV file if not exists
if not os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "image_name", "vehicle_count"])

# Process each image
for image_name in os.listdir(IMAGE_PATH):
    if not image_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    image_path = os.path.join(IMAGE_PATH, image_name)
    image = cv2.imread(image_path)

    # Run YOLO detection
    results = model(image)

    vehicle_count = 0

    # Count vehicles
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        if cls_id in VEHICLE_CLASSES:
            vehicle_count += 1

    # Draw detections
    annotated = results[0].plot()

    # Display vehicle count
    cv2.putText(
        annotated,
        f"Vehicles: {vehicle_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Show image
    cv2.imshow("YOLO Vehicle Detection", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save to CSV
    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            image_name,
            vehicle_count
        ])

print("✅ All images processed successfully")
