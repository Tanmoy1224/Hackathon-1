// Example React hook / snippet
import { useEffect, useRef, useState } from 'react';

export default function DriverMonitor() {
  const videoRef = useRef(null);
  const wsRef = useRef(null);
  const [telemetry, setTelemetry] = useState(null);
  const sirenAudio = useRef(new Audio('/siren.mp3')); // Place siren.mp3 in public/

  useEffect(() => {
    // 1. Setup Camera
    navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
      .then((stream) => { videoRef.current.srcObject = stream; });

    // 2. Setup WebSocket
    const ws = new WebSocket('ws://localhost:8000/ws/dms');
    wsRef.current = ws;

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setTelemetry(data);

      // Trigger Browser Siren on CRITICAL alert
      if (data.trigger_siren) {
        sirenAudio.current.play().catch(() => {});
      } else {
        sirenAudio.current.pause();
        sirenAudio.current.currentTime = 0;
      }
    };

    // 3. Frame Dispatcher Loop (~25 FPS)
    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 480;
    const ctx = canvas.getContext('2d');

    const interval = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN && videoRef.current) {
        ctx.drawImage(videoRef.current, 0, 0, 640, 480);
        const base64Data = canvas.toDataURL('image/jpeg', 0.6); // 60% JPEG quality keeps bandwidth low
        ws.send(base64Data);
      }
    }, 40);

    return () => {
      clearInterval(interval);
      ws.close();
    };
  }, []);

  return (
    <div>
      <video ref={videoRef} autoPlay playsInline muted width="640" height="480" />
      {telemetry && (
        <div style={{ color: telemetry.alert_level === 'CRITICAL' ? 'red' : 'green' }}>
          <h2>Status: {telemetry.status}</h2>
          <p>EAR: {telemetry.ear} | Closed: {telemetry.closed_frames}/{telemetry.max_closed_frames}</p>
          <p>Yawns: {telemetry.yawns_in_window}/{telemetry.max_yawns_allowed}</p>
        </div>
      )}
    </div>
  );
}