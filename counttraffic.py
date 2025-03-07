import cv2 as cv
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Polygon, LineString

# Load the YOLO model
model = YOLO('yolov8n.pt') 

# Load the video file using OpenCV
video = 'TrafficVideo.mp4'
cap = cv.VideoCapture(video)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

fps = cap.get(cv.CAP_PROP_FPS)
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

people = 0
bicycles = 0
cars = 0
people_passed = set()
bicycles_passed = set()
cars_passed = set()


# Lines for crosswalk
line1 = [(550, 850), (1250, 600)]
line2 = [(1500, 950), (1700, 630)]
line3 = [(1250, 600), (1700, 630)]
line4 = [(550, 850), (1500, 950)]


crosswalk = Polygon([line1[0], line1[1], line2[1], line2[0]])


# Initialize the VideoWriter to save the output
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('output_video.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))


def cross_line(box, frame):
    obj_box = Polygon([(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])])
    height, width, _ = frame.shape
    # Define line coordinates based on the middle of the frame
    center_x = width // 2
    p1 = (center_x - 50, 550)
    p2 = (center_x - 50, 900)
    line = LineString([p1, p2])
    #cv.line(frame, p1, p2, (0, 0, 255), 3)
    return obj_box.intersects(line)

# Process each frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get detections from your model
    results = model.track(frame, persist=True)

    # Prepare detections
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = box.conf[0].item()
            cls = box.cls[0].item()
            id = int(box.id[0].item()) if box.id is not None else -1
            detections.append([x1, y1, x2, y2, conf, cls, id])

    for detection in detections:
        x1, y1, x2, y2, conf, cls, id = detection

        # Assign color based on object class
        if model.names[int(cls)] == 'car':
            color = (0, 0, 255)  # Red for cars
        elif model.names[int(cls)] == 'person':
            color = (255, 0, 0)  # Blue for people
        elif model.names[int(cls)] == 'bicycle':
            color = (0, 255, 0)  # Green for bicycles

        label = f'{model.names[int(cls)]} {conf:.2f} ID: {id}'

        cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv.putText(frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        if cross_line((x1, y1, x2, y2), frame):
            if model.names[int(cls)] == 'person' and id not in people_passed:
                people += 1
                people_passed.add(id)
            elif model.names[int(cls)] == 'bicycle' and id not in bicycles_passed:
                bicycles += 1
                bicycles_passed.add(id)
            elif model.names[int(cls)] == 'car' and id not in cars_passed:
                cars += 1
                cars_passed.add(id)

    # Background for counter
    cv.rectangle(frame, (5, 20), (150, 120), (255, 200, 100), -1)

    # Draw the text with white color
    cv.putText(frame, f"Humans: {people}", (15, 40), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv.putText(frame, f"Bicycles: {bicycles}", (15, 70), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv.putText(frame, f"Cars: {cars}", (15, 100), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)


    # Draw crosswalk area
    cv.polylines(frame, [np.array(crosswalk.exterior.coords, dtype=np.int32)], True, (0, 255, 255), 5)
    
    cv.imshow('frame', frame)

    # Write the frame to the output video
    out.write(frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv.destroyAllWindows()

