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

app = FastAPI(title="VIGIL-AI Clean Telemetry Bridge")

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

FOREHEAD = 10
NOSE_TIP = 1
CHIN = 152

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

def calculate_head_pitch(landmarks, w, h):
    forehead_y = landmarks[FOREHEAD].y * h
    nose_y = landmarks[NOSE_TIP].y * h
    chin_y = landmarks[CHIN].y * h

    face_height = chin_y - forehead_y
    if face_height <= 0:
        return 0.4, False

    vertical_ratio = (chin_y - nose_y) / face_height
    is_bowed = (vertical_ratio < 0.23) or ((chin_y - nose_y) < (0.16 * face_height))
    return vertical_ratio, is_bowed

@app.websocket("/ws/telemetry")
async def telemetry_feed(websocket: WebSocket):
    await websocket.accept()
    print("[*] Client connected to /ws/telemetry (Clean Stream)")

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    BUFFER_SIZE = 30
    frame_buffer = deque(maxlen=BUFFER_SIZE)

    calibration_buffer = []
    CALIBRATION_LIMIT = 45
    is_calibrated = False
    ear_threshold = 0.22

    consecutive_closed_frames = 0
    eye_closed_start_time = None
    CONSECUTIVE_CLOSED_LIMIT = 40

    yawn_timestamps = deque()
    currently_yawning = False
    yawn_consecutive_frames = 0

    yawn_alarm_active = False
    yawn_alarm_start_time = None
    YAWN_ALARM_DURATION = 3.5

    alertness_score = 100

    try:
        while True:
            client_data = await websocket.receive_json()
            image_data = client_data.get("image")

            if not image_data:
                continue

            if "," in image_data:
                image_data = image_data.split(",")[1]

            img_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            current_time = time.time()
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            while yawn_timestamps and (current_time - yawn_timestamps[0] > 120):
                yawn_timestamps.popleft()

            if yawn_alarm_active:
                if current_time - yawn_alarm_start_time >= YAWN_ALARM_DURATION:
                    yawn_alarm_active = False
                    yawn_alarm_start_time = None
                    yawn_timestamps.clear()

            ear_avg = 0.28
            mar = 0.15
            is_microsleep = False
            is_head_bowed = False

            if results.multi_face_landmarks:
                mesh = results.multi_face_landmarks[0].landmark

                ear_l = calculate_ear(LEFT_EYE, mesh, w, h)
                ear_r = calculate_ear(RIGHT_EYE, mesh, w, h)
                ear_avg = (ear_l + ear_r) / 2.0
                mar, v_dist, h_dist = calculate_mar(MOUTH, mesh, w, h)

                _, is_head_bowed = calculate_head_pitch(mesh, w, h)

                if not is_calibrated:
                    calibration_buffer.append(ear_avg)
                    if len(calibration_buffer) >= CALIBRATION_LIMIT:
                        baseline_ear = float(np.percentile(calibration_buffer, 75))
                        ear_threshold = round(baseline_ear * 0.70, 2)
                        ear_threshold = max(0.18, min(0.25, ear_threshold))
                        is_calibrated = True
                else:
                    frame_buffer.append([ear_l, ear_r, ear_avg, mar])

                    if is_head_bowed:
                        consecutive_closed_frames = 0
                        eye_closed_start_time = None
                        alertness_score = max(50, alertness_score - 1)

                    elif ear_avg < ear_threshold:
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

                    # --- SENSITIVE & NATURAL YAWN DETECTION ---
                    if not is_microsleep:
                        # Lowered from 0.38 to 0.28 and ratio from 0.32 to 0.22
                        geometric_yawn = (mar >= 0.28) and (v_dist >= (0.22 * h_dist))
                        
                        model_yawn = False
                        if len(frame_buffer) == BUFFER_SIZE:
                            input_tensor = torch.tensor(np.array([list(frame_buffer)]), dtype=torch.float32).to(device)
                            with torch.no_grad():
                                logits = model(input_tensor)
                                pred_class = int(torch.argmax(torch.softmax(logits, dim=1), dim=1).item())
                            model_yawn = (pred_class == 2)

                        # Triggers on comfortable mouth opening
                        if geometric_yawn or (model_yawn and mar >= 0.25):
                            yawn_consecutive_frames += 1
                            if yawn_consecutive_frames >= 4 and not currently_yawning:
                                yawn_timestamps.append(current_time)
                                currently_yawning = True
                                print(f"[EVENT] Yawn recorded: {len(yawn_timestamps)}/3 (MAR: {mar:.3f})")
                        else:
                            yawn_consecutive_frames = 0
                            # Reset flag once mouth closes back below 0.24
                            if mar < 0.24:
                                currently_yawning = False

                    if len(yawn_timestamps) >= 3 and not yawn_alarm_active:
                        yawn_alarm_active = True
                        yawn_alarm_start_time = current_time
                        alertness_score = min(alertness_score, 30)

            else:
                consecutive_closed_frames = 0
                eye_closed_start_time = None

            # --- AUTONOMOUS BRAKE CAN-BUS ESCALATION PIPELINE ---
            unresponsive_time = 0.0
            brake_pressure = 0
            vehicle_speed = 80
            can_command = "0x018 [SYS_STANDBY_NOMINAL]"
            intervention_stage = "CRUISE"
            hazard_active = False

            if is_microsleep and eye_closed_start_time is not None:
                unresponsive_time = current_time - eye_closed_start_time

                if unresponsive_time >= 20.0:
                    intervention_stage = "SAFE_STOP"
                    elapsed_stop = unresponsive_time - 20.0
                    brake_pressure = min(100, int(elapsed_stop * 18) + 40)
                    vehicle_speed = max(0, 56 - int(elapsed_stop * 12))
                    can_command = "0x028 [CMD: EMERGENCY_SAFE_STOP_FULL]"
                    hazard_active = True
                elif unresponsive_time >= 10.0:
                    intervention_stage = "HAPTIC_JOLT"
                    elapsed_jolt = unresponsive_time - 10.0
                    cycle_pos = elapsed_jolt % 3.0
                    pulse_num = min(3, int(elapsed_jolt // 3.0) + 1)
                    
                    if cycle_pos < 0.85 and pulse_num <= 3:
                        brake_pressure = 85
                    else:
                        brake_pressure = 0

                    vehicle_speed = max(50, 80 - (pulse_num * 8))
                    can_command = f"0x130 [CMD: BRAKE_JOLT_PULSE_{pulse_num}/3]"
                    hazard_active = False
                else:
                    intervention_stage = "PRE_WARN"
                    brake_pressure = 0
                    vehicle_speed = 80
                    can_command = "0x110 [STATUS: DRIVER_INATTENTIVE]"
                    hazard_active = False
            else:
                intervention_stage = "CRUISE"
                brake_pressure = 0
                vehicle_speed = 80
                can_command = "0x018 [SYS_STANDBY_NOMINAL]"
                hazard_active = False

            telemetry_payload = {
                "ear": round(float(ear_avg), 3),
                "mar": round(float(mar), 3),
                "alertness": int(alertness_score),
                "consecutive_frames": int(consecutive_closed_frames),
                "microsleep": bool(is_microsleep),
                "yawn_count": int(len(yawn_timestamps)),
                "yawn_warning": bool(yawn_alarm_active),
                "head_down": bool(is_head_bowed),
                "unresponsive_time": round(float(unresponsive_time), 1),
                "brake_pressure": int(brake_pressure),
                "vehicle_speed": int(vehicle_speed),
                "can_command": str(can_command),
                "intervention_stage": str(intervention_stage),
                "hazard_active": bool(hazard_active)
            }

            await websocket.send_json(telemetry_payload)

    except WebSocketDisconnect:
        print("[-] Client disconnected.")
    except Exception as e:
        print(f"[!] Server exception: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)