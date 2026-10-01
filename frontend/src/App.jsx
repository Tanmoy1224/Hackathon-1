import React, { useState, useEffect, useRef } from 'react';
import { 
  ShieldAlert, Eye, Activity, Zap, Terminal, AlertTriangle, 
  Volume2, VolumeX, Flame, BellRing, Disc, Gauge, AlertOctagon, TrendingUp,
  ChevronDown, ArrowUp, Camera
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';

export default function App() {
  const [wsConnected, setWsConnected] = useState(false);
  const [ear, setEar] = useState(0.29);
  const [mar, setMar] = useState(0.18);
  const [alertnessScore, setAlertnessScore] = useState(98);
  const [consecutiveLowEarFrames, setConsecutiveLowEarFrames] = useState(0);
  const [isMicrosleep, setIsMicrosleep] = useState(false);
  const [isYawnWarning, setIsYawnWarning] = useState(false);
  const [isHeadDown, setIsHeadDown] = useState(false);
  const [yawnCount, setYawnCount] = useState(0);
  const [audioMuted, setAudioMuted] = useState(false);
  const [audioUnlocked, setAudioUnlocked] = useState(false);
  const [telemetryHistory, setTelemetryHistory] = useState([]);

  // Autonomous Braking States
  const [unresponsiveTime, setUnresponsiveTime] = useState(0.0);
  const [brakePressure, setBrakePressure] = useState(0);
  const [vehicleSpeed, setVehicleSpeed] = useState(80);
  const [canCommand, setCanCommand] = useState("0x018 [SYS_STANDBY_NOMINAL]");
  const [interventionStage, setInterventionStage] = useState("CRUISE");
  const [hazardActive, setHazardActive] = useState(false);

  const earThreshold = 0.22;
  const marThreshold = 0.28;
  const frameThreshold = 40;

  const audioCtxRef = useRef(null);
  const osc1Ref = useRef(null);
  const osc2Ref = useRef(null);
  const gainNodeRef = useRef(null);
  const isPlayingRef = useRef(false);

  const localVideoRef = useRef(null);
  const captureCanvasRef = useRef(null);
  const socketRef = useRef(null);
  const cameraSectionRef = useRef(null);

  const initAudio = () => {
    try {
      if (!audioCtxRef.current) {
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        audioCtxRef.current = new AudioContext();
      }
      if (audioCtxRef.current.state === 'suspended') {
        audioCtxRef.current.resume();
      }
      setAudioUnlocked(true);
    } catch (e) {
      console.error("Audio unlock failed:", e);
    }
  };

  const startAlarmSound = () => {
    initAudio();
    if (!audioCtxRef.current || isPlayingRef.current || audioMuted) return;

    try {
      const ctx = audioCtxRef.current;
      if (ctx.state === 'suspended') ctx.resume();

      const masterGain = ctx.createGain();
      masterGain.gain.setValueAtTime(0.35, ctx.currentTime);
      masterGain.connect(ctx.destination);
      gainNodeRef.current = masterGain;

      const osc1 = ctx.createOscillator();
      osc1.type = 'sawtooth';
      osc1.frequency.setValueAtTime(880, ctx.currentTime);
      osc1.connect(masterGain);

      const osc2 = ctx.createOscillator();
      osc2.type = 'square';
      osc2.frequency.setValueAtTime(1760, ctx.currentTime);
      osc2.connect(masterGain);

      osc1.start();
      osc2.start();

      osc1Ref.current = osc1;
      osc2Ref.current = osc2;
      isPlayingRef.current = true;
    } catch (err) {
      console.warn("Alarm play error:", err);
    }
  };

  const stopAlarmSound = () => {
    try {
      if (osc1Ref.current) {
        osc1Ref.current.stop();
        osc1Ref.current.disconnect();
        osc1Ref.current = null;
      }
      if (osc2Ref.current) {
        osc2Ref.current.stop();
        osc2Ref.current.disconnect();
        osc2Ref.current = null;
      }
      if (gainNodeRef.current) {
        gainNodeRef.current.disconnect();
        gainNodeRef.current = null;
      }
    } catch (e) {}
    isPlayingRef.current = false;
  };

  useEffect(() => {
    const shouldAlarm = (isMicrosleep || isYawnWarning || alertnessScore < 35) && !audioMuted;
    if (shouldAlarm) {
      startAlarmSound();
    } else {
      stopAlarmSound();
    }
    return () => stopAlarmSound();
  }, [isMicrosleep, isYawnWarning, alertnessScore, audioMuted]);

  // Request HD webcam from client browser
  useEffect(() => {
    let stream = null;
    async function startClientCamera() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
          audio: false
        });
        if (localVideoRef.current) {
          localVideoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error("Camera access error:", err);
      }
    }
    startClientCamera();

    return () => {
      if (stream) stream.getTracks().forEach(track => track.stop());
    };
  }, []);

  // Frame streaming loop
  useEffect(() => {
    let socket;
    let isWaitingForResponse = false;
    let sendLoop = null;

    const sendNextFrame = () => {
      if (
        socket &&
        socket.readyState === WebSocket.OPEN &&
        !isWaitingForResponse &&
        localVideoRef.current &&
        captureCanvasRef.current
      ) {
        const video = localVideoRef.current;
        const canvas = captureCanvasRef.current;

        if (video.videoWidth > 0 && video.videoHeight > 0) {
          canvas.width = 480;
          canvas.height = 360;
          const ctx = canvas.getContext('2d');
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

          const dataUrl = canvas.toDataURL('image/jpeg', 0.55);
          isWaitingForResponse = true;
          socket.send(JSON.stringify({ image: dataUrl }));
        }
      }
    };

    const connectWs = () => {
      socket = new WebSocket('wss://guidable-imprecise-canine.ngrok-free.dev/ws/telemetry');
      socketRef.current = socket;

      socket.onopen = () => {
        setWsConnected(true);
        isWaitingForResponse = false;
        sendLoop = setInterval(sendNextFrame, 35);
      };

      socket.onmessage = (event) => {
        isWaitingForResponse = false;

        try {
          const data = JSON.parse(event.data);
          const currentEar = data.ear !== undefined ? data.ear : 0.28;
          const currentMar = data.mar !== undefined ? data.mar : 0.15;

          if (data.ear !== undefined) setEar(data.ear);
          if (data.mar !== undefined) setMar(data.mar);
          if (data.alertness !== undefined) setAlertnessScore(data.alertness);
          if (data.consecutive_frames !== undefined) setConsecutiveLowEarFrames(data.consecutive_frames);
          if (data.microsleep !== undefined) setIsMicrosleep(data.microsleep);
          if (data.yawn_warning !== undefined) setIsYawnWarning(data.yawn_warning);
          if (data.yawn_count !== undefined) setYawnCount(data.yawn_count);
          if (data.head_down !== undefined) setIsHeadDown(data.head_down);

          if (data.unresponsive_time !== undefined) setUnresponsiveTime(data.unresponsive_time);
          if (data.brake_pressure !== undefined) setBrakePressure(data.brake_pressure);
          if (data.vehicle_speed !== undefined) setVehicleSpeed(data.vehicle_speed);
          if (data.can_command !== undefined) setCanCommand(data.can_command);
          if (data.intervention_stage !== undefined) setInterventionStage(data.intervention_stage);
          if (data.hazard_active !== undefined) setHazardActive(data.hazard_active);

          const timeLabel = new Date().toLocaleTimeString().split(' ')[0];
          setTelemetryHistory(prev => [
            ...prev.slice(-24), 
            {
              time: timeLabel,
              EAR: currentEar,
              MAR: currentMar,
            }
          ]);
        } catch (err) {
          console.error("Payload error:", err);
        }
      };

      socket.onclose = () => {
        setWsConnected(false);
        isWaitingForResponse = false;
        if (sendLoop) clearInterval(sendLoop);
        setTimeout(connectWs, 2000);
      };

      socket.onerror = () => {
        setWsConnected(false);
        isWaitingForResponse = false;
        if (sendLoop) clearInterval(sendLoop);
      };
    };

    connectWs();
    return () => {
      if (sendLoop) clearInterval(sendLoop);
      if (socket) socket.close();
    };
  }, []);

  const scrollToCamera = () => {
    if (cameraSectionRef.current) {
      cameraSectionRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div 
      onClick={initAudio} 
      className="bg-[#070A10] text-slate-100 font-sans select-none min-h-screen"
    >
      <canvas ref={captureCanvasRef} style={{ display: 'none' }} />

      {/* ========================================================= */}
      {/* SCREEN 1: DASHBOARD & TELEMETRY CLUSTER (100vh Front Page)*/}
      {/* ========================================================= */}
      <section className="min-h-screen w-full p-4 md:p-6 flex flex-col justify-between box-border">
        
        {/* BANNER NOTIFICATIONS */}
        <div>
          {isMicrosleep && (
            <div className={`py-2 px-4 rounded-xl border flex items-center justify-between animate-pulse mb-3 shadow-xl ${
              interventionStage === 'SAFE_STOP'
                ? 'bg-red-700/95 border-red-300 shadow-[0_0_25px_rgba(255,0,0,0.8)]'
                : (interventionStage === 'HAPTIC_JOLT'
                    ? 'bg-amber-600/95 border-amber-300 shadow-[0_0_25px_rgba(245,158,11,0.8)]'
                    : 'bg-red-600/90 border-red-400 shadow-[0_0_25px_rgba(255,0,0,0.5)]')
            }`}>
              <div className="flex items-center space-x-3">
                <AlertTriangle className="w-5 h-5 text-yellow-300 animate-bounce flex-shrink-0" />
                <div>
                  <span className="font-extrabold tracking-wider text-sm md:text-base block">
                    {interventionStage === 'SAFE_STOP' && "🛑 CRITICAL: SAFE STOP EXECUTED — 100% BRAKE FORCE"}
                    {interventionStage === 'HAPTIC_JOLT' && "⚡ INTERVENTION: 3X HAPTIC BRAKE JOLT WARNING ENGAGED"}
                    {interventionStage === 'PRE_WARN' && "CRITICAL WARNING: MICROSLEEP DETECTED!"}
                  </span>
                  <span className="text-xs text-red-100">
                    Unresponsive: {unresponsiveTime.toFixed(1)}s • {canCommand}
                  </span>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                {hazardActive && (
                  <span className="text-xs bg-amber-500 text-black font-extrabold px-2 py-0.5 rounded animate-ping">
                    HAZARDS ON
                  </span>
                )}
                <span className="text-xs bg-black/40 px-2.5 py-1 rounded border border-red-300">
                  {consecutiveLowEarFrames} / {frameThreshold} F
                </span>
              </div>
            </div>
          )}

          {isYawnWarning && !isMicrosleep && (
            <div className="bg-amber-600/90 text-white py-2 px-4 rounded-xl border border-amber-400 flex items-center justify-between animate-pulse mb-3 shadow-[0_0_25px_rgba(245,158,11,0.5)]">
              <div className="flex items-center space-x-3">
                <Flame className="w-5 h-5 text-yellow-200 animate-bounce flex-shrink-0" />
                <span className="font-extrabold tracking-wider text-sm md:text-base">
                  FATIGUE ALERT: REST REQUIRED (3 YAWNS IN 2 MINS)
                </span>
              </div>
              <span className="text-xs bg-black/40 px-3 py-1 rounded border border-amber-300">
                HIGH RISK
              </span>
            </div>
          )}

          {isHeadDown && !isMicrosleep && (
            <div className="bg-orange-600/90 text-white py-1.5 px-4 rounded-xl border border-orange-400 flex items-center justify-between mb-3 shadow-[0_0_20px_rgba(234,88,12,0.4)]">
              <span className="font-bold text-xs md:text-sm tracking-wide">
                ⚠ ATTENTION: HEAD BOWED DOWN / ROAD VIEW OCCLUDED
              </span>
              <span className="text-xs font-semibold bg-black/40 px-2 py-0.5 rounded">
                DISTRACTION
              </span>
            </div>
          )}

          {/* HEADER BAR */}
          <header className="flex items-center justify-between border-b border-cyan-900/40 pb-3 mb-3">
            <div className="flex items-center space-x-3">
              <div className="p-2 rounded-lg bg-cyan-950 border border-cyan-500/30 text-cyan-400">
                <ShieldAlert className="w-5 h-5" />
              </div>
              <div>
                <h1 className="font-bold text-lg md:text-xl tracking-tight bg-gradient-to-r from-cyan-400 to-emerald-400 bg-clip-text text-transparent">
                  DRIVEGUARD AI
                </h1>
                <p className="text-[11px] text-slate-400">Driver Fatigue & Autonomous Braking Monitor</p>
              </div>
            </div>

            <div className="flex items-center space-x-2 md:space-x-3">
              <button
                onMouseDown={() => startAlarmSound()}
                onMouseUp={() => stopAlarmSound()}
                onTouchStart={() => startAlarmSound()}
                onTouchEnd={() => stopAlarmSound()}
                className="px-2.5 py-1.5 rounded-lg bg-red-950/80 hover:bg-red-900 text-red-300 border border-red-700/60 text-xs font-bold flex items-center space-x-1 transition active:scale-95"
                title="Press and hold to test siren"
              >
                <BellRing className="w-3.5 h-3.5 text-red-400" />
                <span>HOLD TO TEST SIREN</span>
              </button>

              <div className="flex items-center space-x-1.5 bg-slate-900 px-2.5 py-1.5 rounded-lg border border-slate-800 text-xs">
                <span className={`w-2 h-2 rounded-full ${wsConnected ? 'bg-emerald-400 animate-ping' : 'bg-red-500'}`} />
                <span className={wsConnected ? 'text-emerald-400 font-bold' : 'text-red-400 font-bold'}>
                  {wsConnected ? 'LIVE FEED CONNECTED' : 'DISCONNECTED'}
                </span>
              </div>

              <button
                onClick={() => setAudioMuted(!audioMuted)}
                className={`p-1.5 md:px-2.5 md:py-1.5 rounded-lg border text-xs flex items-center space-x-1 transition ${
                  audioMuted 
                    ? 'bg-slate-900 text-slate-400 border-slate-800' 
                    : 'bg-cyan-950 text-cyan-400 border-cyan-800'
                }`}
              >
                {audioMuted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
                <span className="hidden md:inline">{audioMuted ? "MUTED" : "ALARM ON"}</span>
              </button>
            </div>
          </header>

          {!audioUnlocked && (
            <div 
              onClick={initAudio}
              className="mb-3 bg-cyan-950/60 border border-cyan-500/40 rounded-lg py-1.5 px-3 text-center text-xs text-cyan-300 cursor-pointer hover:bg-cyan-900/50 transition"
            >
              🔊 <strong>Click anywhere on screen</strong> once to initialize audio alarm permissions.
            </div>
          )}
        </div>

        {/* MAIN COCKPIT DASHBOARD (LEFT 6 COLS: VITALS | RIGHT 6 COLS: ACTUATOR & GRAPH) */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 flex-1 items-stretch">
          
          {/* LEFT TELEMETRY CARDS (6 COLUMNS) */}
          <div className="lg:col-span-6 grid grid-cols-1 sm:grid-cols-2 gap-3 flex-1">
            
            {/* ALERTNESS INDEX */}
            <div className="p-3.5 rounded-xl border border-slate-800 bg-[#0D131F] shadow-lg flex flex-col justify-between">
              <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
                <span className="font-semibold uppercase tracking-wider flex items-center gap-1.5">
                  <Zap className="w-3.5 h-3.5 text-emerald-400" /> Alertness Index
                </span>
                <span className="text-[10px] text-slate-400">0 - 100%</span>
              </div>
              <div className="flex items-baseline justify-between my-1">
                <span className={`text-2xl font-extrabold tracking-tight ${
                  alertnessScore > 75 ? 'text-emerald-400' : (alertnessScore > 45 ? 'text-amber-400' : 'text-red-500')
                }`}>
                  {alertnessScore}%
                </span>
                <span className={`text-[10px] font-semibold px-2 py-0.5 rounded ${
                  alertnessScore > 75 
                    ? 'bg-emerald-950 text-emerald-300 border border-emerald-800' 
                    : (alertnessScore > 45 ? 'bg-amber-950 text-amber-300 border border-amber-800' : 'bg-red-950 text-red-300 border border-red-800')
                }`}>
                  {alertnessScore > 75 ? 'OPTIMAL' : (alertnessScore > 45 ? 'FATIGUED' : 'CRITICAL')}
                </span>
              </div>
              <div className="w-full bg-slate-900 h-2 rounded-full overflow-hidden border border-slate-800">
                <div 
                  className={`h-full transition-all duration-300 ${
                    alertnessScore > 75 ? 'bg-emerald-400' : (alertnessScore > 45 ? 'bg-amber-400' : 'bg-red-500')
                  }`}
                  style={{ width: `${alertnessScore}%` }}
                />
              </div>
            </div>

            {/* YAWN FREQUENCY */}
            <div className={`p-3.5 rounded-xl border transition-all shadow-lg flex flex-col justify-between ${
              yawnCount >= 3 
                ? 'bg-red-950/30 border-red-600/60 shadow-[0_0_20px_rgba(255,0,85,0.25)]' 
                : 'bg-[#0D131F] border-slate-800'
            }`}>
              <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
                <span className="font-semibold uppercase tracking-wider flex items-center gap-1.5">
                  <Flame className="w-3.5 h-3.5 text-amber-400" /> Yawns (2m Window)
                </span>
                <span className="text-[10px] text-slate-400">Limit: 3 / 2m</span>
              </div>
              <div className="flex items-baseline justify-between my-1">
                <span className={`text-2xl font-extrabold tracking-tight ${yawnCount >= 3 ? 'text-red-400' : 'text-amber-400'}`}>
                  {yawnCount} <span className="text-sm text-slate-500 font-normal">/ 3</span>
                </span>
                <div className="flex space-x-1">
                  {[1, 2, 3].map((num) => (
                    <span
                      key={num}
                      className={`w-5 h-5 rounded flex items-center justify-center text-[10px] font-bold border transition-colors ${
                        num <= yawnCount 
                          ? 'bg-amber-500 text-slate-950 border-amber-400 shadow-[0_0_8px_rgba(245,158,11,0.5)]' 
                          : 'bg-slate-900 text-slate-500 border-slate-800'
                      }`}
                    >
                      #{num}
                    </span>
                  ))}
                </div>
              </div>
              <div className="w-full bg-slate-900 h-2 rounded-full overflow-hidden border border-slate-800">
                <div 
                  className={`h-full transition-all duration-300 ${yawnCount >= 3 ? 'bg-red-500' : 'bg-amber-400'}`}
                  style={{ width: `${Math.min(100, (yawnCount / 3) * 100)}%` }}
                />
              </div>
            </div>

            {/* EAR METRIC */}
            <div className={`p-3.5 rounded-xl border transition-all shadow-lg flex flex-col justify-between ${
              ear < earThreshold 
                ? 'bg-red-950/30 border-red-600/60 shadow-[0_0_20px_rgba(255,0,85,0.25)]' 
                : 'bg-[#0D131F] border-slate-800'
            }`}>
              <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
                <span className="font-semibold uppercase tracking-wider flex items-center gap-1.5">
                  <Eye className="w-3.5 h-3.5 text-cyan-400" /> EAR Metric
                </span>
                <span className="text-[10px] bg-slate-800 text-slate-300 px-1.5 py-0.5 rounded">
                  &lt; {earThreshold}
                </span>
              </div>
              <div className="flex items-baseline justify-between my-1">
                <span className={`text-2xl font-extrabold tracking-tight ${ear < earThreshold ? 'text-red-400' : 'text-cyan-400'}`}>
                  {ear.toFixed(3)}
                </span>
                <span className={`text-[10px] font-semibold px-2 py-0.5 rounded ${
                  ear < earThreshold ? 'bg-red-900 text-red-200' : 'bg-emerald-950 text-emerald-400 border border-emerald-800'
                }`}>
                  {ear < earThreshold ? 'CLOSED' : 'OPEN'}
                </span>
              </div>
              <div className="w-full bg-slate-900 h-2 rounded-full overflow-hidden border border-slate-800">
                <div 
                  className={`h-full transition-all duration-300 ${ear < earThreshold ? 'bg-red-500' : 'bg-cyan-400'}`}
                  style={{ width: `${Math.min(100, (ear / 0.4) * 100)}%` }}
                />
              </div>
            </div>

            {/* MAR METRIC */}
            <div className={`p-3.5 rounded-xl border transition-all shadow-lg flex flex-col justify-between ${
              mar > marThreshold 
                ? 'bg-amber-950/30 border-amber-600/60 shadow-[0_0_20px_rgba(255,184,0,0.2)]' 
                : 'bg-[#0D131F] border-slate-800'
            }`}>
              <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
                <span className="font-semibold uppercase tracking-wider flex items-center gap-1.5">
                  <Activity className="w-3.5 h-3.5 text-amber-400" /> MAR Metric
                </span>
                <span className="text-[10px] bg-slate-800 text-slate-300 px-1.5 py-0.5 rounded">
                  &ge; {marThreshold}
                </span>
              </div>
              <div className="flex items-baseline justify-between my-1">
                <span className={`text-2xl font-extrabold tracking-tight ${mar > marThreshold ? 'text-amber-400' : 'text-slate-200'}`}>
                  {mar.toFixed(3)}
                </span>
                <span className={`text-[10px] font-semibold px-2 py-0.5 rounded ${
                  mar > marThreshold ? 'bg-amber-950 text-amber-300 border border-amber-800' : 'bg-slate-900 text-slate-400'
                }`}>
                  {mar > marThreshold ? 'YAWNING' : 'NORMAL'}
                </span>
              </div>
              <div className="w-full bg-slate-900 h-2 rounded-full overflow-hidden border border-slate-800">
                <div 
                  className={`h-full transition-all duration-300 ${mar > marThreshold ? 'bg-amber-400' : 'bg-emerald-500'}`}
                  style={{ width: `${Math.min(100, (mar / 0.8) * 100)}%` }}
                />
              </div>
            </div>

            {/* CONSECUTIVE CLOSED FRAMES */}
            <div className="sm:col-span-2 p-3.5 rounded-xl border border-slate-800 bg-[#0D131F] shadow-lg flex flex-col justify-between">
              <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
                <span className="font-semibold uppercase tracking-wider flex items-center gap-1.5">
                  <Terminal className="w-3.5 h-3.5 text-purple-400" /> Consecutive Closed Frames
                </span>
                <span className="text-[10px] text-slate-400">Limit: {frameThreshold} f (~1.5s)</span>
              </div>
              <div className="flex items-baseline justify-between my-1">
                <span className={`text-2xl font-extrabold tracking-tight ${
                  consecutiveLowEarFrames >= frameThreshold ? 'text-red-500' : 'text-purple-300'
                }`}>
                  {consecutiveLowEarFrames} <span className="text-xs text-slate-500 font-normal">/ {frameThreshold}</span>
                </span>
                <span className={`text-[10px] font-semibold px-2 py-0.5 rounded ${
                  consecutiveLowEarFrames >= frameThreshold ? 'bg-red-950 text-red-300 border border-red-800' : 'bg-slate-900 text-slate-400'
                }`}>
                  {consecutiveLowEarFrames >= frameThreshold ? 'TRIGGERED' : 'MONITORING'}
                </span>
              </div>
              <div className="w-full bg-slate-900 h-2 rounded-full overflow-hidden border border-slate-800">
                <div 
                  className={`h-full transition-all duration-200 ${
                    consecutiveLowEarFrames >= frameThreshold ? 'bg-red-600 animate-pulse' : 'bg-purple-500'
                  }`}
                  style={{ width: `${Math.min(100, (consecutiveLowEarFrames / frameThreshold) * 100)}%` }}
                />
              </div>
            </div>

          </div>

          {/* RIGHT COLUMN: AUTONOMOUS BRAKE ACTUATOR & LIVE GRAPH (6 COLUMNS) */}
          <div className="lg:col-span-6 flex flex-col justify-between space-y-3">
            
            {/* AUTONOMOUS BRAKE ACTUATOR (CAN-BUS) */}
            <div className={`p-4 rounded-xl border transition-all shadow-xl ${
              interventionStage === 'SAFE_STOP'
                ? 'bg-red-950/40 border-red-600 shadow-[0_0_20px_rgba(255,0,0,0.3)]'
                : (interventionStage === 'HAPTIC_JOLT'
                    ? 'bg-amber-950/40 border-amber-500 shadow-[0_0_20px_rgba(245,158,11,0.25)]'
                    : 'bg-[#0D131F] border-slate-800')
            }`}>
              <div className="flex items-center justify-between text-slate-400 text-xs mb-2">
                <span className="font-semibold uppercase tracking-wider flex items-center gap-1.5 text-slate-300">
                  <Disc className={`w-4 h-4 ${brakePressure > 0 ? 'text-red-400 animate-spin' : 'text-cyan-400'}`} />
                  ECU Autonomous Brake Actuator (CAN-Bus)
                </span>
                <span className="text-[10px] bg-cyan-950 text-cyan-300 px-2 py-0.5 rounded border border-cyan-800 font-mono">
                  {canCommand}
                </span>
              </div>

              <div className="grid grid-cols-2 gap-3 my-2">
                <div className="bg-black/40 p-2.5 rounded-lg border border-slate-800 flex items-center justify-between">
                  <div>
                    <span className="text-[10px] text-slate-400 block uppercase">Vehicle Speed</span>
                    <span className={`text-2xl font-black ${vehicleSpeed === 0 ? 'text-red-500' : 'text-cyan-400'}`}>
                      {vehicleSpeed} <span className="text-xs text-slate-500 font-normal">KM/H</span>
                    </span>
                  </div>
                  <Gauge className={`w-6 h-6 ${vehicleSpeed === 0 ? 'text-red-500' : 'text-slate-600'}`} />
                </div>

                <div className="bg-black/40 p-2.5 rounded-lg border border-slate-800 flex items-center justify-between">
                  <div>
                    <span className="text-[10px] text-slate-400 block uppercase">Hydraulic Pressure</span>
                    <span className={`text-2xl font-black ${brakePressure > 50 ? 'text-red-400' : 'text-slate-200'}`}>
                      {brakePressure}% <span className="text-xs text-slate-500 font-normal">APPLIED</span>
                    </span>
                  </div>
                  <AlertOctagon className={`w-6 h-6 ${brakePressure > 0 ? 'text-red-400' : 'text-slate-600'}`} />
                </div>
              </div>

              <div className="w-full bg-slate-900 h-2.5 rounded-full overflow-hidden border border-slate-800 mt-2">
                <div 
                  className={`h-full transition-all duration-150 ${brakePressure > 50 ? 'bg-red-500' : (brakePressure > 0 ? 'bg-amber-400' : 'bg-cyan-500')}`}
                  style={{ width: `${brakePressure}%` }}
                />
              </div>
            </div>

            {/* REAL-TIME DUAL-STREAM TELEMETRY GRAPH (EAR vs MAR) */}
            <div className="bg-[#0D131F] border border-cyan-900/40 rounded-xl p-4 shadow-xl flex flex-col justify-between flex-1">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center space-x-2">
                  <TrendingUp className="w-4 h-4 text-cyan-400" />
                  <span className="text-xs uppercase font-bold tracking-wider text-slate-200">
                    Fatigue Telemetry Stream (EAR vs MAR)
                  </span>
                </div>

                <div className="flex items-center space-x-3 text-[10px]">
                  <div className="flex items-center space-x-1.5">
                    <span className="w-2 h-2 rounded-full bg-cyan-400 inline-block" />
                    <span className="text-cyan-300 font-bold">EAR</span>
                  </div>
                  <div className="flex items-center space-x-1.5">
                    <span className="w-2 h-2 rounded-full bg-amber-400 inline-block" />
                    <span className="text-amber-300 font-bold">MAR</span>
                  </div>
                </div>
              </div>

              <div className="h-[180px] w-full pt-1">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={telemetryHistory} margin={{ top: 5, right: 10, left: -25, bottom: 0 }}>
                    <XAxis dataKey="time" stroke="#334155" fontSize={9} tickLine={false} />
                    <YAxis domain={[0, 0.65]} stroke="#334155" fontSize={9} tickLine={false} />
                    
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: 'rgba(11, 15, 25, 0.95)', 
                        borderColor: '#00F0FF33', 
                        borderRadius: '8px', 
                        fontSize: '10px'
                      }}
                      labelStyle={{ color: '#94A3B8' }}
                    />

                    <ReferenceLine y={earThreshold} stroke="#FF0055" strokeDasharray="3 3" />
                    <ReferenceLine y={marThreshold} stroke="#FFB800" strokeDasharray="3 3" />

                    <Line type="monotone" dataKey="EAR" stroke="#00F0FF" strokeWidth={1.8} dot={false} isAnimationActive={false} />
                    <Line type="monotone" dataKey="MAR" stroke="#FFB800" strokeWidth={1.8} dot={false} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              <div className="mt-2 pt-2 border-t border-slate-800 flex items-center justify-between text-[10px] text-slate-500 font-mono">
                <span>ROLLING BUFFER: 24 SAMPLES</span>
                <span className="text-cyan-400">CAN-BUS TELEMETRY SYNCED</span>
              </div>
            </div>

          </div>

        </div>

        {/* BOTTOM NOTCH SCROLL-DOWN INDICATOR */}
        <div className="pt-3 mt-2 border-t border-slate-900 flex items-center justify-between text-xs text-slate-500">
          <span>VIGIL-AI EDGE COCKPIT</span>

          <button 
            onClick={scrollToCamera}
            className="flex items-center space-x-1.5 text-cyan-400 hover:text-cyan-300 font-bold bg-cyan-950/70 hover:bg-cyan-900/60 px-4 py-1.5 rounded-full border border-cyan-800/80 transition shadow-lg active:scale-95"
          >
            <Camera className="w-3.5 h-3.5" />
            <span>SCROLL DOWN FOR DRIVER CAMERA FEED</span>
            <ChevronDown className="w-4 h-4 animate-bounce" />
          </button>

          <span className="text-cyan-400 font-bold">&lt; 50MS EDGE LATENCY</span>
        </div>

      </section>

      {/* ========================================================= */}
      {/* SCREEN 2: DEDICATED WEBCAM VIEW (Scrolled Down Page)       */}
      {/* ========================================================= */}
      <section 
        ref={cameraSectionRef}
        className="min-h-screen w-full p-6 md:p-10 flex flex-col justify-center items-center bg-[#05070D] border-t-2 border-cyan-950 relative"
      >
        <div className="w-full max-w-4xl bg-[#0D131F] border border-cyan-900/50 rounded-2xl p-6 shadow-2xl flex flex-col justify-between">
          
          <div className="flex items-center justify-between mb-4 pb-3 border-b border-slate-800">
            <div className="flex items-center space-x-2.5">
              <span className="w-3 h-3 rounded-full bg-emerald-400 animate-pulse" />
              <div>
                <h2 className="text-base font-bold uppercase tracking-wider text-slate-200">
                  Driver Diagnostic Camera Feed (Hardware HD)
                </h2>
                <p className="text-[11px] text-slate-400">
                  Real-time direct hardware stream used for facial landmark classification
                </p>
              </div>
            </div>

            <div className="flex items-center space-x-2">
              <span className="text-xs bg-cyan-950 text-cyan-400 font-mono px-3 py-1 rounded-lg border border-cyan-800">
                ZERO-LATENCY FEED
              </span>
              <button 
                onClick={scrollToTop}
                className="flex items-center space-x-1 text-xs text-slate-300 hover:text-cyan-300 bg-slate-900 hover:bg-slate-800 px-3 py-1 rounded-lg border border-slate-700 transition"
              >
                <ArrowUp className="w-3.5 h-3.5" />
                <span>Top</span>
              </button>
            </div>
          </div>

          {/* CLEAN VIDEO ELEMENT - NATIVE 60FPS WITHOUT OVERLAY DOTS */}
          <div className={`w-full aspect-video bg-black rounded-xl overflow-hidden border transition-all flex items-center justify-center relative shadow-inner ${
            isMicrosleep ? 'border-red-500 shadow-[0_0_30px_rgba(255,0,0,0.7)]' : 'border-slate-800'
          }`}>
            <video 
              ref={localVideoRef} 
              autoPlay 
              playsInline 
              muted 
              className="w-full h-full object-cover" 
            />
          </div>

          <div className="mt-4 pt-3 border-t border-slate-800/80 flex items-center justify-between text-xs text-slate-400">
            <span>CLIENT HARDWARE CAMERA CAPTURE VIA WEBRTC / MEDIA-DEVICES</span>
            <button 
              onClick={scrollToTop}
              className="text-cyan-400 hover:underline font-bold flex items-center space-x-1"
            >
              <span>↑ Return to Cockpit Cluster</span>
            </button>
          </div>

        </div>
      </section>

    </div>
  );
}