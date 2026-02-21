"""
Araba seçme özelliği YOLO üzerinde default olarak geliyor.
Detection: Nesneleri seçer, hepsine aynı ID'yi atar.
Tracking: Nesneleri seçer, sınıflandırarak hepsine ayrı ID atar.
"""
from ultralytics import YOLO
import cv2

# Model describe;
model = YOLO("yolov8n") # n,m,l Models avaliable.

video_path = "IMG_5268.MOV"
cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) # Frame per second, Number of frames per second in the video.
out = cv2.VideoWriter("Output.avi",cv2.VideoWriter_fourcc(*"XVID"),fps,(width,height))

# Tracking algorihtms;
while cap.isOpened(): # Continues until the video is finished.
    success,frame = cap.read() # Read the one frame of the video.
    if not success: # if the video is finished, break the loop.
        break
    # YOLO Tracking;
    results = model.track(
        frame, # Input frame.
        persist=True, # So that the tracking IDs are the same across frames.
        conf = 0.3, # Confidence threshold for detections. (0-1)
        iou = 0.5, # intersection over union: how much the object boxes should overlap.
        tracker = "bytetrack.yaml", # Tracker algorithm to use. (bytetrack, strongsort, deepsort)
        # classes = [2] just car. 
    )
    annotated_frame = results[0].plot() # Write the car ID on the frame and return the annotated frame.
    cv2.imshow("YOLO v8 Tracking",annotated_frame) # Show the annotated frame in a window.
    out.write(annotated_frame) # Write the annotated frame to the output video.

    if cv2.waitKey(1) & 0xFF == ord("q"): # If the user presses the "q" key, video will be finished.
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Finished.