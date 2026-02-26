from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/data2/weights/best.pt")

Image_path = "test3.jpg"
image = cv2.imread(Image_path)

# Image Prediction;
results = model(image)[0]
print(results)

# Box drawing;
for box in results.boxes:
    # Cordinates;
    x1,y1,x2,y2 = map(int, box.xyxy[0]) # Cross Cordinates.
    cls_id = int(box.cls[0]) # Classification id.
    confidence = float(box.conf[0]) # Confidence score.
    label = f"model:{model.names[cls_id]} conf:{confidence:.2f}" # Detection label.

    # Box drawing;
    cv2.rectangle(image,(x1,y1,),(x2,y2),(0,255,0),2)
    # Adding label to the box;
    cv2.putText(image,label,(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

cv2.imshow("Prediction",image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("Prediction_results.png",image)




# Finished.