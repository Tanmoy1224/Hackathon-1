import base64
import os
import time
from collections import deque
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# -----------------------------
# 1. FastAPI App Initialization
# -----------------------------
app = FastAPI(title="VIGIL-AI Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust to your React app's origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# 2. GRU Model Architecture
# -----------------------------
class LightweightGRU(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2, num_classes=4, dropout=0.2):
        super(LightweightGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_block = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc_block(out[:, -1, :])

# -----------------------------
# 3. Model Loading & Geometry
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join("models", "fatigue_gru.pth")

model = LightweightGRU(input_size=4, num_classes=4).to(DEVICE)
if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    print(f"[*] Loaded model weights successfully onto {DEVICE}")
else:
    print(f"[!] Warning: '{MODEL_PATH}' not found. Initialized with raw weights.")
model.eval()

# MediaPipe Setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

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

# -----------------------------
# 4. Session State Tracker Class
# -----------------------------
class DriverSessionState:
    def __init__(self):
        self.frame_buffer = deque(maxlen=30)
        self.consecutive_closed_frames = 0
        self.eye_closed_start_time = None
        self.yawn_timestamps = deque()
        self.yawn_consecutive_frames = 0
        self.currently_yawning = False

    def reset_closure(self):
        self.consecutive_closed_frames = 0
        self.eye_closed_start_time = None

# -----------------------------
# 5. WebSocket Route
# -----------------------------
@app.websocket("/ws/dms")
async def dms_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session = DriverSessionState()
    print("[*] Driver client connected via WebSocket")

    EAR_CLOSURE_THRESHOLD = 0.21
    CONSECUTIVE_CLOSED_LIMIT = 45
    YAWN_WINDOW_SECONDS = 120
    YAWN_ALERT_THRESHOLD = 3
    MIN_YAWN_FRAMES = 20

    try:
        while True:
            # Expecting base64 encoded JPEG string or JSON with 'image' field
            data = await websocket.receive_text()
            if "," in data:
                data = data.split(",")[1]  # Strip data:image/jpeg;base64 prefix if present

            # Decode image bytes
            img_bytes = base64.b64decode(data)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                await websocket.send_json({"error": "Invalid frame"})
                continue

            current_time = time.time()
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            # Purge yawns older than 2 minutes
            while session.yawn_timestamps and (current_time - session.yawn_timestamps[0] > YAWN_WINDOW_SECONDS):
                session.yawn_timestamps.popleft()

            # Default payload structure
            response = {
                "face_detected": False,
                "status": "NO_FACE",
                "alert_level": "WARNING",  # NORMAL | WARNING | CRITICAL
                "trigger_siren": False,
                "ear": 0.0,
                "mar": 0.0,
                "closed_frames": session.consecutive_closed_frames,
                "max_closed_frames": CONSECUTIVE_CLOSED_LIMIT,
                "yawns_in_window": len(session.yawn_timestamps),
                "max_yawns_allowed": YAWN_ALERT_THRESHOLD,
            }

            if results.multi_face_landmarks:
                response["face_detected"] = True
                mesh = results.multi_face_landmarks[0].landmark

                ear_l = calculate_ear(LEFT_EYE, mesh, w, h)
                ear_r = calculate_ear(RIGHT_EYE, mesh, w, h)
                ear_avg = (ear_l + ear_r) / 2.0
                mar, v_dist, h_dist = calculate_mar(MOUTH, mesh, w, h)

                session.frame_buffer.append([ear_l, ear_r, ear_avg, mar])
                response["ear"] = round(ear_avg, 3)
                response["mar"] = round(mar, 3)

                microsleep_triggered = False

                # 1. Microsleep Evaluation (45 frames / 1.5s)
                if ear_avg < EAR_CLOSURE_THRESHOLD:
                    session.consecutive_closed_frames += 1
                    if session.eye_closed_start_time is None:
                        session.eye_closed_start_time = current_time

                    duration = current_time - session.eye_closed_start_time
                    if session.consecutive_closed_frames >= CONSECUTIVE_CLOSED_LIMIT or duration >= 1.5:
                        microsleep_triggered = True
                        response["status"] = "MICROSLEEP"
                        response["alert_level"] = "CRITICAL"
                        response["trigger_siren"] = True
                    else:
                        response["status"] = "DROWSY"
                        response["alert_level"] = "WARNING"
                else:
                    session.reset_closure()

                response["closed_frames"] = session.consecutive_closed_frames

                # 2. GRU Model Classification (when not in microsleep)
                if not microsleep_triggered and len(session.frame_buffer) == 30:
                    input_tensor = torch.tensor(
                        np.array([list(session.frame_buffer)]), dtype=torch.float32
                    ).to(DEVICE)

                    with torch.no_grad():
                        logits = model(input_tensor)
                        pred_class = int(torch.argmax(torch.softmax(logits, dim=1), dim=1).item())

                    # Yawn vs Smile check
                    is_yawn = (pred_class == 2) and (mar >= 0.45) and (v_dist >= (0.42 * h_dist))

                    if is_yawn:
                        session.yawn_consecutive_frames += 1
                        response["status"] = "YAWNING"
                        response["alert_level"] = "WARNING"
                        if session.yawn_consecutive_frames >= MIN_YAWN_FRAMES and not session.currently_yawning:
                            session.yawn_timestamps.append(current_time)
                            session.currently_yawning = True
                    else:
                        session.yawn_consecutive_frames = 0
                        session.currently_yawning = False
                        if session.consecutive_closed_frames == 0:
                            response["status"] = "ATTENTIVE"
                            response["alert_level"] = "NORMAL"

                    # 3. Yawn Frequency Alert
                    if len(session.yawn_timestamps) >= YAWN_ALERT_THRESHOLD:
                        response["status"] = "FREQUENT_YAWNS"
                        response["alert_level"] = "CRITICAL"
                        response["trigger_siren"] = True

                response["yawns_in_window"] = len(session.yawn_timestamps)

            else:
                session.reset_closure()

            # Return real-time inference JSON back to React
            await websocket.send_json(response)

    except WebSocketDisconnect:
        print("[-] Driver client disconnected")
    except Exception as e:
        print(f"[!] WebSocket loop error: {e}")
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)