import cv2
import mediapipe as mp
import numpy as np
import csv
import time

# --- Setup MediaPipe Face Mesh ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmark Indices (MediaPipe)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [61, 291, 13, 14, 78, 308]

def calculate_ear(eye_landmarks, landmarks, w, h):
    pts = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in eye_landmarks])
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    h_dist = np.linalg.norm(pts[0] - pts[3])
    return (v1 + v2) / (2.0 * (h_dist + 1e-6))

def calculate_mar(mouth_landmarks, landmarks, w, h):
    pts = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in mouth_landmarks])
    v_dist = np.linalg.norm(pts[2] - pts[3])
    h_dist = np.linalg.norm(pts[0] - pts[1])
    return v_dist / (h_dist + 1e-6)

# --- CSV Output File ---
csv_file = open("driver_fatigue_dataset.csv", mode="w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["ear_left", "ear_right", "ear_avg", "mar", "label"])

cap = cv2.VideoCapture(0)
current_label = None
recording = False

print("Controls:\n[0] Alert  [1] Microsleep  [2] Yawning  [3] Distracted\n[SPACE] Stop Recording  [ESC] Exit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        mesh = results.multi_face_landmarks[0].landmark
        ear_l = calculate_ear(LEFT_EYE, mesh, w, h)
        ear_r = calculate_ear(RIGHT_EYE, mesh, w, h)
        ear_avg = (ear_l + ear_r) / 2.0
        mar = calculate_mar(MOUTH, mesh, w, h)

        if recording and current_label is not None:
            writer.writerow([round(ear_l, 4), round(ear_r, 4), round(ear_avg, 4), round(mar, 4), current_label])
            cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)  # Red recording indicator

        cv2.putText(frame, f"EAR: {ear_avg:.2f} | MAR: {mar:.2f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Active Label: {current_label if recording else 'IDLE'}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Dataset Collector", frame)
    key = cv2.waitKey(1) & 0xFF

    if key in [ord('0'), ord('1'), ord('2'), ord('3')]:
        current_label = chr(key)
        recording = True
    elif key == ord(' '):  # Spacebar to pause recording
        recording = False
    elif key == 27:        # ESC to exit
        break

cap.release()
csv_file.close()
cv2.destroyAllWindows()