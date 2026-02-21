# Import Libraries;
import cv2
import numpy as np
from tensorflow.keras.models import load_model # CNN Model Loading

# Model Load;
Model = load_model('mnist_cnn_model.h5')

# Camera Initialization;
cap = cv2.VideoCapture(0)
print("You should write number on the paper and show it to the camera.")
print("You can press 'q' to quit the camera preview.")

# Predict the images coming from the camera using CV2;
while True:
    success, frame = cap.read() # Read the camera frame
    if not success:
        break
    
    # Gray Scale;
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # ROI (Region of Interest) Selection;
    h,w = gray.shape
    box_size = 200
    top_left = (w//2-box_size//2,h//2-box_size//2)
    bottom_right = (w//2+box_size//2,h//2+box_size//2)
    cv2.rectangle(frame, top_left, bottom_right, (0,255,0),2)

    # Predict the number with ROI;
    roi = gray[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]]
    roi = cv2.resize(roi, (28,28))
    roi = roi.astype("float32")/255.0
    roi = roi.reshape(1,28,28,1)

    # Prediction;
    prediction = Model.predict(roi, verbose = 0)
    digit = np.argmax(prediction)

    # Write the prediction on the frame;
    cv2.putText(frame, f"Prediction: {digit}", (10,50),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),2)
    
    # Show the frame;
    cv2.imshow("Preview", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# Finished.