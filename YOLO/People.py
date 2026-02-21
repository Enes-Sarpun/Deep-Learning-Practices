"""
People Detection with YOLO.
"""
# Import libraries;
import cv2
import numpy as np
from ultralytics import YOLO

# Load Model;
model = YOLO("yolov8n.pt")

# Video Capture;
cap = cv2.VideoCapture("2.mp4")

success,frame = cap.read()
if not success:
    print("Video açılamadı.")
    exit()  

# Resize;
frame = cv2.resize(frame, (0,0),fx=0.6,fy=0.6)
frame_height, frame_width = frame.shape[:2]

# Line Describe;
line_x = int(frame_width*0.5)
offset = 10

# Counters;
entering=0
exiting=0
counted_ids=set()
person_last_x={}

# People counting;
while True:
    success,frame = cap.read()
    if not success:
        break
    
    # Again Resize;
    frame = cv2.resize(frame, (0,0),fx=0.6,fy=0.6)

    # Tracking ve Detection;
    results = model.track(frame,persist=True,stream=False,conf=0.25,iou=0.3,tracker="bytetrack.yaml") # only person class
    
    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.int().tolist()# ID.
        classes = results[0].boxes.cls.int().tolist() # Class ids
        xyxy = results[0].boxes.xyxy # Cordinates of the bounding boxes.

        for i,box in enumerate(xyxy):
            cls_id = classes[i]
            track_id = ids[i]
            class_name = model.names[cls_id]
            if class_name !='person':
                continue # Just count people.
            
            # Let's find the coordinates and the centers of the people found;
            x1,y1,x2,y2 = map(int,box)
            cx = int((x1+x2)/2)
            cy = int((y1+y2)/2)

            previous_x = person_last_x.get(track_id, None)
            person_last_x[track_id] = cx

            if previous_x is not None:
                # Those who move from right to left;
                if previous_x > line_x >= cx:
                    if track_id not in counted_ids:
                        counted_ids.add(track_id)
                        exiting+=1
                # Those passing on the left and right;
                elif previous_x < line_x <= cx:
                    if track_id not in counted_ids:
                        counted_ids.add(track_id)
                        entering+=1
            
            # Box and Center Drawing;
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,f"{class_name} ID:{track_id}",(x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
            cv2.circle(frame,(cx,cy),4,(255,0,0),-1)

    # Counter Display;
    cv2.putText(frame,f"Entering: {entering}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
    cv2.putText(frame,f"Exiting: {exiting}",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)

    # Line;
    cv2.line(frame,(line_x,0),(line_x,frame_height),(255,0,0),2)

    # Visualization;
    cv2.imshow("Direction Tracking:",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



# Finished.