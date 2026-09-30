import os
import time
from collections import deque
import cv2
import mediapipe as mp
import numpy as np
import pygame
import torch
import torch.nn as nn

# -----------------------------
# 1. Reliable Hardware Audio Synthesizer (85dB Dual-Tone Siren)
# -----------------------------
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

def generate_siren_sound(freq1=3200, freq2=2200, duration_per_tone=0.18):
    sample_rate = 44100
    n_samples = int(sample_rate * duration_per_tone)

    t = np.linspace(0, duration_per_tone, n_samples, False)
    tone1 = np.sin(2 * np.pi * freq1 * t) * 0.9
    tone2 = np.sin(2 * np.pi * freq2 * t) * 0.9

    siren_wave = np.concatenate([tone1, tone2])
    siren_stereo = np.column_stack([siren_wave, siren_wave])
    siren_bytes = (siren_stereo * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(siren_bytes)

EMERGENCY_SOUND = generate_siren_sound()
is_alarming = False

def start_alarm():
    global is_alarming
    if not is_alarming:
        is_alarming = True
        EMERGENCY_SOUND.play(loops=-1)

def stop_alarm():
    global is_alarming
    if is_alarming:
        is_alarming = False
        EMERGENCY_SOUND.stop()

# -----------------------------
# 2. Model Architecture
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
# 3. Model Loading
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = os.path.join("models", "fatigue_gru.pth")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found!")

checkpoint = torch.load(model_path, map_location=device)
if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    input_size = checkpoint.get("input_size", 4)
    num_classes = checkpoint.get("num_classes", 4)
    model = LightweightGRU(input_size=input_size, num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["state_dict"])
else:
    model = LightweightGRU(input_size=4, num_classes=4).to(device)
    model.load_state_dict(checkpoint)
model.eval()

# -----------------------------
# 4. MediaPipe & Geometry Setup
# -----------------------------
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
# 5. HUD Renderer Function
# -----------------------------
def render_driver_display(state_text, sub_text, alert_level, ear, mar, yawns_count, closed_frames, dynamic_thresh, max_closed=45):
    canvas = np.zeros((480, 800, 3), dtype=np.uint8)
    canvas[:] = (20, 20, 24)

    colors = {
        'NORMAL': (60, 220, 80),     # Green
        'WARNING': (0, 165, 255),    # Amber
        'CRITICAL': (30, 30, 240)    # Red
    }
    active_color = colors.get(alert_level, (180, 180, 180))

    # Header Bar
    cv2.rectangle(canvas, (0, 0), (800, 60), (32, 32, 38), -1)
    cv2.putText(canvas, "AI DRIVER MONITORING SYSTEM // ADAS CLUSTER", (25, 40),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (200, 200, 200), 1)

    # Flashing Edge on Critical Alert
    if alert_level == 'CRITICAL':
        if int(time.time() * 6) % 2 == 0:
            cv2.rectangle(canvas, (0, 0), (800, 480), (0, 0, 255), 14)

    # Center Primary State Card
    cv2.rectangle(canvas, (50, 90), (750, 250), (28, 28, 34), -1)
    cv2.rectangle(canvas, (50, 90), (750, 250), active_color, 2)

    cv2.putText(canvas, state_text, (80, 165),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, active_color, 3)
    cv2.putText(canvas, sub_text, (80, 215),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 220), 1)

    # Telemetry Gauge 1: Eye Closure Progress
    cv2.putText(canvas, "EYE CLOSURE DURATION", (50, 295),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (160, 160, 160), 1)
    cv2.rectangle(canvas, (50, 310), (450, 335), (40, 40, 48), -1)
    closure_ratio = min(1.0, closed_frames / float(max_closed))
    bar_width = int(400 * closure_ratio)
    bar_color = (60, 220, 80) if closure_ratio < 0.5 else ((0, 165, 255) if closure_ratio < 0.9 else (30, 30, 240))
    if bar_width > 0:
        cv2.rectangle(canvas, (50, 310), (50 + bar_width, 335), bar_color, -1)
    cv2.putText(canvas, f"{closed_frames}/{max_closed} f", (465, 330),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Telemetry Gauge 2: Yawn Frequency Boxes (2 Min Window)
    cv2.putText(canvas, "YAWN FREQUENCY (2 MIN WINDOW)", (50, 385),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (160, 160, 160), 1)
    for i in range(3):
        box_x = 50 + (i * 70)
        box_color = (0, 165, 255) if i < yawns_count else (40, 40, 48)
        cv2.rectangle(canvas, (box_x, 400), (box_x + 55, 435), box_color, -1)
        cv2.rectangle(canvas, (box_x, 400), (box_x + 55, 435), (80, 80, 90), 1)
        cv2.putText(canvas, f"#{i+1}", (box_x + 15, 424), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Right Side Numerical Card (Shows Adaptive Threshold)
    cv2.rectangle(canvas, (540, 280), (750, 440), (28, 28, 34), -1)
    cv2.rectangle(canvas, (540, 280), (750, 440), (45, 45, 55), 1)
    cv2.putText(canvas, "ADAPTIVE SENSORS", (555, 305),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(canvas, f"EAR: {ear:.2f}", (555, 345),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(canvas, f"MAR: {mar:.2f}", (555, 385),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(canvas, f"Limit: {dynamic_thresh:.2f}", (555, 422),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 1)

    return canvas

# -----------------------------
# 6. Counters & Buffer Setup
# -----------------------------
BUFFER_SIZE = 30
frame_buffer = deque(maxlen=BUFFER_SIZE)

# Eyeglasses Dynamic Calibration Setup
calibration_buffer = []
CALIBRATION_LIMIT = 60  # ~2 seconds at 30 FPS
is_calibrated = False
baseline_ear = 0.30
ear_threshold = 0.22  # Fallback default

CONSECUTIVE_CLOSED_LIMIT = 45
consecutive_closed_frames = 0
eye_closed_start_time = None

yawn_timestamps = deque()
YAWN_WINDOW_SECONDS = 120
YAWN_ALERT_THRESHOLD = 3
currently_yawning = False
yawn_consecutive_frames = 0
MIN_YAWN_FRAMES = 20

yawn_alarm_active = False
yawn_alarm_start_time = None
YAWN_ALARM_DURATION = 3.0

cap = cv2.VideoCapture(0)
print("Cockpit display running. Keep eyes open naturally for 2s calibration.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    state_text = "SYSTEM ACTIVE"
    sub_text = "Driver attentive and eyes on road"
    alert_level = 'NORMAL'
    ear_avg = 0.0
    mar = 0.0
    microsleep_active = False

    # Purge yawns older than 2 minutes
    while yawn_timestamps and (current_time - yawn_timestamps[0] > YAWN_WINDOW_SECONDS):
        yawn_timestamps.popleft()

    # Manage Active Yawn Alarm Countdown & Auto-Reset
    if yawn_alarm_active:
        if current_time - yawn_alarm_start_time >= YAWN_ALARM_DURATION:
            yawn_alarm_active = False
            yawn_alarm_start_time = None
            yawn_timestamps.clear()
            print("[INFO] Yawn siren complete. Counter reset to 0/3.")
        else:
            state_text = "FATIGUE ALERT: REST REQUIRED"
            sub_text = f"Excessive yawning! Siren active ({YAWN_ALARM_DURATION - (current_time - yawn_alarm_start_time):.1f}s)"
            alert_level = 'CRITICAL'

    if results.multi_face_landmarks:
        mesh = results.multi_face_landmarks[0].landmark

        ear_l = calculate_ear(LEFT_EYE, mesh, w, h)
        ear_r = calculate_ear(RIGHT_EYE, mesh, w, h)
        ear_avg = (ear_l + ear_r) / 2.0
        mar, v_dist, h_dist = calculate_mar(MOUTH, mesh, w, h)

        # Draw Landmarks on Diagnostic Feed
        for idx in LEFT_EYE + RIGHT_EYE:
            cv2.circle(frame, (int(mesh[idx].x * w), int(mesh[idx].y * h)), 1, (0, 255, 0), -1)

        # -------------------------------------------------------------
        # STEP 1: DYNAMIC EYEGLASSES CALIBRATION (First ~2 seconds)
        # -------------------------------------------------------------
        if not is_calibrated:
            calibration_buffer.append(ear_avg)
            state_text = f"CALIBRATING SENSORS ({len(calibration_buffer)}/{CALIBRATION_LIMIT})"
            sub_text = "Adapting to eyeglasses and driver eye geometry..."
            alert_level = 'WARNING'

            if len(calibration_buffer) >= CALIBRATION_LIMIT:
                # Use 75th percentile to establish driver's normal open-eye baseline
                baseline_ear = float(np.percentile(calibration_buffer, 75))
                # Set dynamic threshold to 70% of open-eye baseline
                ear_threshold = round(baseline_ear * 0.70, 2)
                # Keep safely constrained
                ear_threshold = max(0.18, min(0.25, ear_threshold))
                is_calibrated = True
                print(f"[CALIBRATED] Driver Baseline: {baseline_ear:.2f} | Adaptive Limit: {ear_threshold:.2f}")

        else:
            frame_buffer.append([ear_l, ear_r, ear_avg, mar])

            # -------------------------------------------------------------
            # STEP 2: MICROSLEEP EVALUATION (Using Adaptive Threshold)
            # -------------------------------------------------------------
            if ear_avg < ear_threshold:
                consecutive_closed_frames += 1
                if eye_closed_start_time is None:
                    eye_closed_start_time = current_time

                duration = current_time - eye_closed_start_time
                if consecutive_closed_frames >= CONSECUTIVE_CLOSED_LIMIT or duration >= 1.5:
                    microsleep_active = True
                    state_text = "EMERGENCY: WAKE UP!"
                    sub_text = f"Microsleep detected ({duration:.1f}s) - Pull over safely"
                    alert_level = 'CRITICAL'
                else:
                    if not yawn_alarm_active:
                        state_text = "WARNING: DROWSINESS"
                        sub_text = f"Eyelids drooping ({consecutive_closed_frames}/{CONSECUTIVE_CLOSED_LIMIT} frames)"
                        alert_level = 'WARNING'
            else:
                consecutive_closed_frames = 0
                eye_closed_start_time = None

            # -------------------------------------------------------------
            # STEP 3: MODEL CLASSIFICATION & YAWN DETECTION
            # -------------------------------------------------------------
            if not microsleep_active and not yawn_alarm_active and len(frame_buffer) == BUFFER_SIZE:
                input_tensor = torch.tensor(np.array([list(frame_buffer)]), dtype=torch.float32).to(device)
                with torch.no_grad():
                    logits = model(input_tensor)
                    pred_class = int(torch.argmax(torch.softmax(logits, dim=1), dim=1).item())

                # Distinguish smile from yawn
                is_yawn = (pred_class == 2) and (mar >= 0.45) and (v_dist >= (0.42 * h_dist))

                if is_yawn:
                    yawn_consecutive_frames += 1
                    state_text = "CAUTION: YAWN DETECTED"
                    sub_text = "Fatigue symptoms registered"
                    alert_level = 'WARNING'
                    if yawn_consecutive_frames >= MIN_YAWN_FRAMES and not currently_yawning:
                        yawn_timestamps.append(current_time)
                        currently_yawning = True
                        print(f"[EVENT] Yawn registered! Count in 2 min: {len(yawn_timestamps)}/3")
                else:
                    yawn_consecutive_frames = 0
                    currently_yawning = False
                    if consecutive_closed_frames == 0:
                        state_text = "SYSTEM ACTIVE"
                        sub_text = "Driver attentive and eyes on road"
                        alert_level = 'NORMAL'

                # Trigger Yawn Siren on 3rd Yawn
                if len(yawn_timestamps) >= YAWN_ALERT_THRESHOLD and not yawn_alarm_active:
                    yawn_alarm_active = True
                    yawn_alarm_start_time = current_time
                    state_text = "FATIGUE ALERT: REST REQUIRED"
                    sub_text = "3 yawns in 2 mins. High accident risk!"
                    alert_level = 'CRITICAL'

    else:
        consecutive_closed_frames = 0
        eye_closed_start_time = None
        if not yawn_alarm_active:
            state_text = "WARNING: DRIVER UNSEEN"
            sub_text = "Face obstructed or looking away from road"
            alert_level = 'WARNING'

    # Master Siren Dispatcher
    if microsleep_active or yawn_alarm_active:
        start_alarm()
    else:
        stop_alarm()

    # Render Cockpit HUD
    hud_canvas = render_driver_display(
        state_text=state_text,
        sub_text=sub_text,
        alert_level=alert_level,
        ear=ear_avg,
        mar=mar,
        yawns_count=len(yawn_timestamps),
        closed_frames=consecutive_closed_frames,
        dynamic_thresh=ear_threshold,
        max_closed=CONSECUTIVE_CLOSED_LIMIT
    )

    cv2.imshow("Driver Cockpit Display (HUD)", hud_canvas)
    cv2.imshow("Camera Diagnostic Feed", frame)

    # Press 'r' to recalibrate anytime while running
    key = cv2.waitKey(1) & 0xFF
    if key in [27, ord('q')]:
        break
    elif key == ord('r'):
        calibration_buffer.clear()
        is_calibrated = False
        print("[*] Manual recalibration triggered.")

stop_alarm()
cap.release()
cv2.destroyAllWindows()