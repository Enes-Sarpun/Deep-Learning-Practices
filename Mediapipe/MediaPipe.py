# Import libraries;
import cv2 
import mediapipe as mp
import numpy as np 

# A function that calculates the angle is defined;
def calculate_angle(a, b, c): # Calculates the angle between three points.
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]- b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle >180.0:
        angle = 360-angle
    
    return angle

# Mediapipe;
mp_drawing = mp.solutions.drawing_utils # To draw on a video or image.
mp_pose = mp.solutions.pose # For pose estimation.

# Load video file;
cap = cv2.VideoCapture("squat_test1.avi")

# A rule-based pose estimation class is defined;
counter = 0 # Squat Counter.
stage = None # Down or UP.

def classify_pose(knee_angle):# Pose estimation class.
    if knee_angle<100:
        return "Squatting"
    elif 100<=knee_angle<=160:
        return "Lunging"
    else:
        return "Standing"
    
# print(calculate_angle([0,1],[0,0],[1,0]))
# print(classify_pose(150))

# Pose module is initialized and video frames are processed in a loop.
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():# Video is open.
        ret, frame = cap.read() # Read a frame from the video.
        if not ret:
            print("Video ended.")
            break

        # Convert the image to RGB and make it writeable to improve performance.
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Pose estimation.
        results = pose.process(image)

        # Convert the image back to BGR and make it writeable for drawing.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try: 
            landmarks = results.pose_landmarks.landmark

            # Right hip, knee, and ankle coordinates:
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            
            # Calculate the knee angle;
            angle = calculate_angle(right_hip,right_knee,right_ankle)

            # Classify the pose;
            current_pose = classify_pose(angle)

            # Squat counter logic;
            if angle < 90:
                stage = "down"
            if angle > 160 and stage == 'down':
                stage = "up"    
                counter += 1
            
            # Print the information on the screen;
            cv2.putText(image, f"Diz acisi: {angle}", (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)
            cv2.putText(image,f"Squat Sayaci: {counter}",(10,30), 
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2,cv2.LINE_AA)
            cv2.putText(image,f"Poz: {current_pose}",(10,90), 
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2,cv2.LINE_AA)
        except:
            pass

        # Draw the connections of the key points;
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                      )
        # Show the video with pose estimation;
        cv2.imshow("Mediapipe Pose Tahmini", image)

        if cv2.waitKey(10) &  0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


# Finished.