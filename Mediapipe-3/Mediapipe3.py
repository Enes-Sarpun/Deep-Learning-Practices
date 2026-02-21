"""
Sleeping test for driver.
"""
# Import necessary libraries
import cv2
import mediapipe as mp
import numpy as np
import threading
from playsound import playsound

# Initialize MediaPipe Face Mesh;
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# CV2 video capture;
cap = cv2.VideoCapture(0)

# Values;
Left_Eye = [159,145]
Right_Eye = [386, 374]
Left_brow = [65,158]
Right_brow = [295, 385]

Frame_Threshold = 150
sleep_counter_frame = 5*30 # 5 seconds at 30 FPS
T_DROWSY = 21

closed_eyes_counter = 0
current_state = "Awake"

alarm = "alarm.mp3"
alarm_is_playing = False

# Alarm Settings;
def play_alarm():
    global alarm_is_playing
    if not alarm_is_playing:
        alarm_is_playing = True
        playsound(alarm)
        alarm_is_playing = False

# Solutions; 
def emotion_eyes(landmarks, image_width, image_height, threshold):
    def get_(index):
        lm = landmarks[index]
        return np.array([int(lm.x*image_width),int(lm.y*image_height)])
    # Calculate eyes;
    left_eye_point = get_(159)
    left_brow_point = get_(65)
    right_eye_point = get_(386)
    right_brow_point = get_(295)

    results1 = np.linalg.norm(left_eye_point - left_brow_point)
    results2 = np.linalg.norm(right_eye_point - right_brow_point)
    avg_eyes = (results1 + results2) / 2

    if avg_eyes > threshold:
        return "Drowsy", avg_eyes
    else:
        return "Awake", avg_eyes

# Emotion prediction with cv2;
while True:
    ret,frame = cap.read()
    if not ret:
        break
# BGR to RGB conversion;
    frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
# Screen Dimesions;
    h,w,_=frame.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Emotion detection;
            emotion, distance = emotion_eyes(face_landmarks.landmark, w, h, T_DROWSY)
            # Update state based on emotion;
            if emotion == "Drowsy":
                closed_eyes_counter += 1
                if closed_eyes_counter >= sleep_counter_frame:
                    current_state = "Sleeping"
                    # Play alarm;
                    if not alarm_is_playing:
                        t=threading.Thread(target=play_alarm)
                        t.daemon = True
                        t.start()
                else:
                    current_state = "Drowsy"
            else:
                closed_eyes_counter = 0
                current_state = "Awake"
            
            # Display emotion on frame;
            color = (0,255,0) if current_state == "Awake" else (0,165,255) if current_state == "Drowsy" else (0,0,255)
            cv2.putText(frame,f'State: {current_state}',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,color,2)
            cv2.putText(frame, f'Timer: {closed_eyes_counter}',(30,110),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)
            cv2.putText(frame, f'Distance: {distance:.2f}, T={T_DROWSY}', (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Drawing landmarks;
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0), thickness=1)
            )

    cv2.imshow("Driver Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# Finished.