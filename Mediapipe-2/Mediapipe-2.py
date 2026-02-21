"""
Mediapipe kütüphanesi kullanılarak insan yüz ifadesi tanıma.
"""
# Import necessary libraries;
import cv2
import mediapipe as mp
import numpy as np 

# Inıtialize mediapipe face detection and drawing modules;
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# Initialize camera using opencv module;
cap = cv2.VideoCapture(0) 

# Values;
Left_Eye = [159,145]
Left_brow = [65,158]
Mouth = [69,291]

# Initializing mouth and eyes metric analysis according to face mesh;
def detection_emotion(landmarks,image_width,image_height):
    
    def get_distance(index):
        lm = landmarks[index]
        return np.array([int(lm.x*image_width), int(lm.y*image_height)])

    # Eyes and Brows (Left side);
    brow_point = get_distance(65)
    eye_point = get_distance(159) 
    brow_lift = np.linalg.norm(brow_point - eye_point)

    # Mouth;
    mouth_point_left = get_distance(69)
    mouth_point_right = get_distance(291)
    mouth_width = np.linalg.norm(mouth_point_left - mouth_point_right)

    if brow_lift > 15:
        return "Surprised"
    elif mouth_width > 50:
        return "Happy"
    else:
        return "Neutral"

# Emotions prediction with cv2;
while True:
    ret,frame = cap.read()
    if not ret:
        break
# BGR to RGB conversion;
    frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
# Screen Dimesions;
    h,w,_ = frame.shape 

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Emotion detection;
            emotion = detection_emotion(face_landmarks.landmark,w,h)
            # Write emotion on the Screen;
            cv2.putText(frame, f"Emotion: {emotion}", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
            # Draw face landmarks;
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                landmarks_drawing_spec=None,
                connections_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0), thickness=1)
            )
    cv2.imshow("Face Emotion Recognition",frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()

# Finished.