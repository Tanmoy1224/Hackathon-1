import os
import time
import asyncio
import base64
from collections import deque
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="VIGIL-AI Minimal Testing Bridge")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LightweightGRU(input_size=4, num_classes=4).to(device)

model_path = os.path.join("models", "fatigue_gru.pth")
if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    print(f"[*] Loaded trained GRU model weights onto {device}")
model.eval()

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [61, 291, 13, 14, 78, 308]

def calculate_ear(eye_landmarks, landmarks, w, h):
    pts = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in eye_landmarks])
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    h_dist = np.linalg.norm(pts[0] - pts[3])
    return float((v1 + v2) / (2.0 * (h_dist + 1e-6)))

def calculate_mar(mouth_landmarks, landmarks, w, h):
    pts = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in mouth_landmarks])
    v_dist = np.linalg.norm(pts[2] - pts[3])
    h_dist = np.linalg.norm(pts[0] - pts[1])
    return float(v_dist / (h_dist + 1e-6)), float(v_dist), float(h_dist)

@app.websocket("/ws/telemetry")
async def telemetry_feed(websocket: WebSocket):
    await websocket.accept()
    print("[*] React Frontend connected to /ws/telemetry")

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] Error: Could not open camera.")
        await websocket.close()
        return

    BUFFER_SIZE = 30
    frame_buffer = deque(maxlen=BUFFER_SIZE)

    calibration_buffer = []
    CALIBRATION_LIMIT = 60
    is_calibrated = False
    ear_threshold = 0.22

    consecutive_closed_frames = 0
    eye_closed_start_time = None
    CONSECUTIVE_CLOSED_LIMIT = 45

    # 2-Minute Window Yawn Tracking
    yawn_timestamps = deque()
    currently_yawning = False
    yawn_consecutive_frames = 0
    MIN_YAWN_FRAMES = 20

    # Auto-Reset Yawn Alarm Timer
    yawn_alarm_active = False
    yawn_alarm_start_time = None
    YAWN_ALARM_DURATION = 3.5  # Siren lasts 3.5s then resets counter

    alertness_score = 100

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(0.01)
                continue

            current_time = time.time()
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            # Purge yawns older than 2 minutes (120s)
            while yawn_timestamps and (current_time - yawn_timestamps[0] > 120):
                yawn_timestamps.popleft()

            # Handle Yawn Alarm Timeout and Counter Reset
            if yawn_alarm_active:
                if current_time - yawn_alarm_start_time >= YAWN_ALARM_DURATION:
                    yawn_alarm_active = False
                    yawn_alarm_start_time = None
                    yawn_timestamps.clear()  # Resets back to 0/3!
                    print("[INFO] Yawn alarm duration completed. Counter reset to 0.")

            ear_avg = 0.28
            mar = 0.15
            is_microsleep = False

            if results.multi_face_landmarks:
                mesh = results.multi_face_landmarks[0].landmark

                ear_l = calculate_ear(LEFT_EYE, mesh, w, h)
                ear_r = calculate_ear(RIGHT_EYE, mesh, w, h)
                ear_avg = (ear_l + ear_r) / 2.0
                mar, v_dist, h_dist = calculate_mar(MOUTH, mesh, w, h)

                for idx in LEFT_EYE + RIGHT_EYE:
                    cv2.circle(frame, (int(mesh[idx].x * w), int(mesh[idx].y * h)), 1, (0, 255, 0), -1)
                for idx in MOUTH:
                    cv2.circle(frame, (int(mesh[idx].x * w), int(mesh[idx].y * h)), 1, (0, 180, 255), -1)

                # Eyeglasses dynamic calibration
                if not is_calibrated:
                    calibration_buffer.append(ear_avg)
                    cv2.putText(frame, f"CALIBRATING... {len(calibration_buffer)}/{CALIBRATION_LIMIT}", 
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    if len(calibration_buffer) >= CALIBRATION_LIMIT:
                        baseline_ear = float(np.percentile(calibration_buffer, 75))
                        ear_threshold = round(baseline_ear * 0.70, 2)
                        ear_threshold = max(0.18, min(0.25, ear_threshold))
                        is_calibrated = True
                else:
                    frame_buffer.append([ear_l, ear_r, ear_avg, mar])

                    # Microsleep (45 frames / 1.5s)
                    if ear_avg < ear_threshold:
                        consecutive_closed_frames += 1
                        if eye_closed_start_time is None:
                            eye_closed_start_time = current_time

                        duration = current_time - eye_closed_start_time
                        if consecutive_closed_frames >= CONSECUTIVE_CLOSED_LIMIT or duration >= 1.5:
                            is_microsleep = True
                            alertness_score = max(10, alertness_score - 4)
                        else:
                            alertness_score = max(45, alertness_score - 1)
                    else:
                        consecutive_closed_frames = 0
                        eye_closed_start_time = None
                        alertness_score = min(100, alertness_score + 1)

                    # Model Inference & Yawn Counter
                    if not is_microsleep and len(frame_buffer) == BUFFER_SIZE:
                        input_tensor = torch.tensor(np.array([list(frame_buffer)]), dtype=torch.float32).to(device)
                        with torch.no_grad():
                            logits = model(input_tensor)
                            pred_class = int(torch.argmax(torch.softmax(logits, dim=1), dim=1).item())

                        is_yawn = (pred_class == 2) and (mar >= 0.45) and (v_dist >= (0.42 * h_dist))
                        if is_yawn:
                            yawn_consecutive_frames += 1
                            if yawn_consecutive_frames >= MIN_YAWN_FRAMES and not currently_yawning:
                                yawn_timestamps.append(current_time)
                                currently_yawning = True
                                print(f"[EVENT] Yawn recorded: {len(yawn_timestamps)}/3")
                        else:
                            yawn_consecutive_frames = 0
                            currently_yawning = False

                    # Trigger alarm on 3 yawns within 2 minutes
                    if len(yawn_timestamps) >= 3 and not yawn_alarm_active:
                        yawn_alarm_active = True
                        yawn_alarm_start_time = current_time
                        alertness_score = min(alertness_score, 30)

            else:
                consecutive_closed_frames = 0
                eye_closed_start_time = None

            if is_microsleep or yawn_alarm_active:
                cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 10)

            display_frame = cv2.resize(frame, (640, 360))
            _, buffer = cv2.imencode('.jpg', display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')

            telemetry_payload = {
                "ear": round(float(ear_avg), 3),
                "mar": round(float(mar), 3),
                "alertness": int(alertness_score),
                "consecutive_frames": int(consecutive_closed_frames),
                "microsleep": bool(is_microsleep),
                "yawn_count": int(len(yawn_timestamps)),
                "yawn_alert": bool(yawn_alarm_active),
                "frame": f"data:image/jpeg;base64,{jpg_as_text}"
            }

            await websocket.send_json(telemetry_payload)
            await asyncio.sleep(0.03)

    except WebSocketDisconnect:
        print("[-] Client disconnected.")
    except Exception as e:
        print(f"[!] Server exception: {e}")
    finally:
        cap.release()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)