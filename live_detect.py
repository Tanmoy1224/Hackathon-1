import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import os
import threading
import time

# -----------------------------
# 1. Model Architecture
# -----------------------------
class LightweightGRU(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2, num_classes=4, dropout=0.2):
        super(LightweightGRU, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc_block = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc_block(out[:, -1, :])

# -----------------------------
# 2. Audio Engine
# -----------------------------
is_alerting = False

def play_alert():
    global is_alerting
    try:
        import winsound
        winsound.Beep(2800, 300)
    except Exception:
        pass
    is_alerting = False

def trigger_audio():
    global is_alerting
    if not is_alerting:
        is_alerting = True
        threading.Thread(target=play_alert, daemon=True).start()

# -----------------------------
# 3. Model Loading & Landmarks
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = os.path.join("models", "fatigue_gru.pth")

checkpoint = torch.load(model_path, map_location=device)
model = LightweightGRU(input_size=4, num_classes=4).to(device)
if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"])
else:
    model.load_state_dict(checkpoint)
model.eval()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

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
    return v_dist / (h_dist + 1e-6), v_dist, h_dist

# -----------------------------
# 4. Calibration & Buffer Setup
# -----------------------------
BUFFER_SIZE = 30
frame_buffer = deque(maxlen=BUFFER_SIZE)

# Dynamic Eyeglasses / Driver Calibration Variables
calibration_frames = []
CALIBRATION_LIMIT = 60  # ~2 seconds at 30 FPS
is_calibrated = False
baseline_ear = 0.30
ear_threshold = 0.22  # Default fallback

consecutive_closed_frames = 0
eye_closed_start_time = None
CONSECUTIVE_CLOSED_LIMIT = 45

yawn_timestamps = deque()
currently_yawning = False
yawn_consecutive_frames = 0

cap = cv2.VideoCapture(0)
print("[*] Starting backend monitor. Keep eyes open naturally for 2s calibration.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    status_text = "INITIALIZING..."
    status_color = (200, 200, 200)

    # Purge yawns older than 2 mins
    while yawn_timestamps and (current_time - yawn_timestamps[0] > 120):
        yawn_timestamps.popleft()

    if results.multi_face_landmarks:
        mesh = results.multi_face_landmarks[0].landmark
        
        ear_l = calculate_ear(LEFT_EYE, mesh, w, h)
        ear_r = calculate_ear(RIGHT_EYE, mesh, w, h)
        ear_avg = (ear_l + ear_r) / 2.0
        mar, v_dist, h_dist = calculate_mar(MOUTH, mesh, w, h)

        # --- AUTO-CALIBRATION ROUTINE (Compensates for white rims / reflections) ---
        if not is_calibrated:
            calibration_frames.append(ear_avg)
            status_text = f"CALIBRATING GLASSES... {len(calibration_frames)}/{CALIBRATION_LIMIT}"
            status_color = (0, 165, 255)
            
            if len(calibration_frames) >= CALIBRATION_LIMIT:
                # 75th percentile of normal driving gaze
                baseline_ear = float(np.percentile(calibration_frames, 75))
                # When wearing glasses, a closed eye drops to ~70% of open-eye baseline
                ear_threshold = round(baseline_ear * 0.70, 2)
                # Keep within practical bounds
                ear_threshold = max(0.18, min(0.24, ear_threshold))
                is_calibrated = True
                print(f"[CALIBRATED] Driver Baseline: {baseline_ear:.2f} | Adaptive Threshold: {ear_threshold:.2f}")
        else:
            frame_buffer.append([ear_l, ear_r, ear_avg, mar])

            # Microsleep Evaluation using the calibrated threshold
            if ear_avg < ear_threshold:
                consecutive_closed_frames += 1
                if eye_closed_start_time is None:
                    eye_closed_start_time = current_time

                duration = current_time - eye_closed_start_time
                if consecutive_closed_frames >= CONSECUTIVE_CLOSED_LIMIT or duration >= 1.5:
                    status_text = "CRITICAL: MICROSLEEP!"
                    status_color = (0, 0, 255)
                    trigger_audio()
                else:
                    status_text = f"EYES DROOPING ({consecutive_closed_frames}/{CONSECUTIVE_CLOSED_LIMIT})"
                    status_color = (0, 165, 255)
            else:
                consecutive_closed_frames = 0
                eye_closed_start_time = None

            # GRU Inference
            if len(frame_buffer) == BUFFER_SIZE:
                input_tensor = torch.tensor(np.array([list(frame_buffer)]), dtype=torch.float32).to(device)
                with torch.no_grad():
                    logits = model(input_tensor)
                    pred_class = int(torch.argmax(torch.softmax(logits, dim=1), dim=1).item())

                is_yawn = (pred_class == 2) and (mar >= 0.45) and (v_dist >= (0.42 * h_dist))

                if is_yawn:
                    yawn_consecutive_frames += 1
                    status_text = "YAWNING"
                    status_color = (0, 165, 255)
                    if yawn_consecutive_frames >= 20 and not currently_yawning:
                        yawn_timestamps.append(current_time)
                        currently_yawning = True
                else:
                    yawn_consecutive_frames = 0
                    currently_yawning = False
                    if consecutive_closed_frames == 0:
                        status_text = "ALERT / ATTENTIVE"
                        status_color = (0, 255, 0)

                if len(yawn_timestamps) >= 3:
                    status_text = "FATIGUE: >3 YAWNS IN 2 MIN"
                    status_color = (0, 0, 255)
                    trigger_audio()

        # Telemetry overlay
        cv2.putText(frame, f"EAR: {ear_avg:.2f} | Adaptive Limit: {ear_threshold:.2f}", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, f"Closed Frames: {consecutive_closed_frames}/{CONSECUTIVE_CLOSED_LIMIT}", (20, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255) if consecutive_closed_frames > 0 else (255, 255, 255), 2)
    else:
        consecutive_closed_frames = 0
        eye_closed_start_time = None
        status_text = "NO FACE DETECTED"
        status_color = (0, 0, 255)

    cv2.rectangle(frame, (10, h - 60), (w - 10, h - 10), (25, 25, 25), -1)
    cv2.putText(frame, f"STATUS: {status_text}", (25, h - 22), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2)

    cv2.imshow("Glasses Robust DMS Diagnostic", frame)
    if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
        break

cap.release()
cv2.destroyAllWindows()