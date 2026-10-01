import React, { useState, useEffect, useRef } from 'react';
import { 
  ShieldAlert, Eye, Activity, Zap, Terminal, AlertTriangle, 
  Volume2, VolumeX, Flame, BellRing, Radio, Compass, TrendingUp, ChevronDown, Camera
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
  const [camFrame, setCamFrame] = useState(null);
  const [audioMuted, setAudioMuted] = useState(false);
  const [audioUnlocked, setAudioUnlocked] = useState(false);
  const [fps, setFps] = useState(25.0);
  const [telemetryHistory, setTelemetryHistory] = useState([]);
  const [cameraActive, setCameraActive] = useState(false);

  const earThreshold = 0.22;
  const marThreshold = 0.45;
  const frameThreshold = 45;

  const audioCtxRef = useRef(null);
  const osc1Ref = useRef(null);
  const osc2Ref = useRef(null);
  const gainNodeRef = useRef(null);
  const isPlayingRef = useRef(false);

  // Hidden video and canvas elements for client camera capture
  const localVideoRef = useRef(null);
  const hiddenCanvasRef = useRef(null);
  const socketRef = useRef(null);

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
      masterGain.gain.setValueAtTime(0.4, ctx.currentTime);
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

  // Request browser camera on whoever's computer visits the link
  useEffect(() => {
    async function startClientCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 480 }, height: { ideal: 360 }, facingMode: "user" },
          audio: false
        });
        if (localVideoRef.current) {
          localVideoRef.current.srcObject = stream;
          setCameraActive(true);
        }
      } catch (err) {
        console.error("Camera access error:", err);
        alert("Camera access denied or unavailable. Please allow camera permissions in your browser.");
      }
    }
    startClientCamera();

    return () => {
      if (localVideoRef.current && localVideoRef.current.srcObject) {
        localVideoRef.current.srcObject.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  // WebSocket Connection & Frame Streaming Loop
  useEffect(() => {
    let socket;
    let frameTimes = [];
    let frameInterval = null;

    const connectWs = () => {
      // Points to your active ngrok tunnel
      socket = new WebSocket('wss://guidable-imprecise-canine.ngrok-free.dev/ws/telemetry');
      socketRef.current = socket;

      socket.onopen = () => {
        setWsConnected(true);

        // Start sending client's browser video frames every 50ms (~20 FPS)
        frameInterval = setInterval(() => {
          if (socket.readyState === WebSocket.OPEN && localVideoRef.current && hiddenCanvasRef.current) {
            const video = localVideoRef.current;
            const canvas = hiddenCanvasRef.current;
            if (video.videoWidth > 0 && video.videoHeight > 0) {
              canvas.width = 480;
              canvas.height = 360;
              const ctx = canvas.getContext('2d');
              ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
              const dataUrl = canvas.toDataURL('image/jpeg', 0.6);
              socket.send(JSON.stringify({ image: dataUrl }));
            }
          }
        }, 50);
      };

      socket.onmessage = (event) => {
        try {
          const now = performance.now();
          frameTimes.push(now);
          if (frameTimes.length > 8) {
            frameTimes.shift();
            const avgDelta = (frameTimes[frameTimes.length - 1] - frameTimes[0]) / (frameTimes.length - 1);
            if (avgDelta > 0) setFps((1000 / avgDelta).toFixed(1));
          }

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
          if (data.frame) setCamFrame(data.frame);

          const timeLabel = new Date().toLocaleTimeString().split(' ')[0];
          setTelemetryHistory(prev => [
            ...prev.slice(-29), 
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
        if (frameInterval) clearInterval(frameInterval);
        setTimeout(connectWs, 2000);
      };

      socket.onerror = () => {
        setWsConnected(false);
        if (frameInterval) clearInterval(frameInterval);
      };
    };

    connectWs();
    return () => {
      if (frameInterval) clearInterval(frameInterval);
      if (socket) socket.close();
    };
  }, []);

  const getSystemStatus = () => {
    if (isMicrosleep) return { text: "CRITICAL: MICROSLEEP", color: "text-red-500", border: "border-red-500", bg: "bg-red-950/60" };
    if (isYawnWarning) return { text: "FATIGUE: FREQUENT YAWN", color: "text-amber-500", border: "border-amber-500", bg: "bg-amber-950/60" };
    if (isHeadDown) return { text: "DISTRACTION: HEAD BOWED", color: "text-orange-500", border: "border-orange-500", bg: "bg-orange-950/60" };
    if (consecutiveLowEarFrames > 15) return { text: "WARNING: DROWSINESS", color: "text-amber-400", border: "border-amber-400", bg: "bg-amber-950/40" };
    return { text: "NOMINAL: ATTENTIVE", color: "text-cyan-400", border: "border-cyan-500/40", bg: "bg-cyan-950/30" };
  };

  const status = getSystemStatus();

  const radius = 56;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (alertnessScore / 100) * circumference;

  return (
    <div 
      onClick={initAudio} 
      className="min-h-screen w-full bg-[#05070D] text-slate-100 font-mono select-none relative overflow-x-hidden"
    >
      {/* Hidden elements capturing current client device video */}
      <video ref={localVideoRef} autoPlay playsInline muted className="hidden" />
      <canvas ref={hiddenCanvasRef} className="hidden" />

      {/* Cyber Grid Background */}
      <div 
        className="fixed inset-0 pointer-events-none opacity-15"
        style={{
          backgroundImage: `linear-gradient(#0ff 1px, transparent 1px), linear-gradient(90deg, #0ff 1px, transparent 1px)`,
          backgroundSize: '35px 35px'
        }}
      />

      {/* SECTION 1: FRONT SCREEN COCKPIT (Zero Scroll) */}
      <div className="h-screen w-full p-4 flex flex-col justify-between box-border relative z-10">
        
        {/* POPUP ALERT BANNER */}
        {isMicrosleep && (
          <div className="absolute top-2 left-6 right-6 z-50 bg-red-600/95 text-white py-2 px-5 rounded-2xl border-2 border-red-300 flex items-center justify-between animate-pulse shadow-[0_0_35px_rgba(255,0,0,0.8)] backdrop-blur-md">
            <div className="flex items-center space-x-3">
              <AlertTriangle className="w-6 h-6 text-yellow-300 animate-bounce" />
              <div>
                <span className="font-extrabold text-base tracking-wider block font-sans">
                  EMERGENCY: DRIVER MICROSLEEP DETECTED!
                </span>
                <span className="text-[11px] text-red-100">
                  Acoustic Alarm Active • Frame Duration: {(consecutiveLowEarFrames / 30).toFixed(1)}s (Limit {frameThreshold}f)
                </span>
              </div>
            </div>
            <span className="text-xs font-bold bg-black/60 px-3 py-1 rounded-xl border border-red-300">
              {consecutiveLowEarFrames}/{frameThreshold} F
            </span>
          </div>
        )}

        {/* HEADER BAR */}
        <header className="flex items-center justify-between border border-cyan-900/40 bg-[#090D16]/90 backdrop-blur-md rounded-2xl px-4 py-2 shadow-[0_2px_20px_rgba(0,255,255,0.05)]">
          <div className="flex items-center space-x-3">
            <div className="p-2 rounded-xl bg-cyan-950 border border-cyan-500/40 text-cyan-400">
              <ShieldAlert className="w-5 h-5 animate-pulse" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h1 className="font-extrabold text-base tracking-tight bg-gradient-to-r from-cyan-400 to-emerald-400 bg-clip-text text-transparent font-sans">
                  DRIVEGUARD AI COCKPIT
                </h1>
                <span className="text-[9px] uppercase font-bold tracking-widest bg-cyan-950/80 text-cyan-300 px-1.5 py-0.5 rounded border border-cyan-800">
                  ADAS HUD
                </span>
              </div>
            </div>
          </div>

          <div className={`px-4 py-1.5 rounded-xl border ${status.border} ${status.bg} flex items-center space-x-2 shadow-inner`}>
            <span className="w-2.5 h-2.5 rounded-full bg-current animate-ping" />
            <span className={`text-xs font-black tracking-wider uppercase font-sans ${status.color}`}>
              {status.text}
            </span>
          </div>

          <div className="flex items-center space-x-3">
            <button
              onMouseDown={() => startAlarmSound()}
              onMouseUp={() => stopAlarmSound()}
              onTouchStart={() => startAlarmSound()}
              onTouchEnd={() => stopAlarmSound()}
              className="px-3 py-1 rounded-xl bg-red-950/70 hover:bg-red-900 text-red-300 border border-red-700/60 text-xs font-bold flex items-center space-x-1.5 transition active:scale-95"
              title="Hold to test siren"
            >
              <BellRing className="w-3.5 h-3.5 text-red-400" />
              <span>TEST SIREN</span>
            </button>

            <div className="flex items-center space-x-1.5 bg-slate-900 px-2.5 py-1 rounded-xl border border-slate-800 text-xs">
              <Radio className={`w-3 h-3 ${wsConnected ? 'text-emerald-400 animate-spin' : 'text-red-500'}`} />
              <span className={wsConnected ? 'text-emerald-400 font-bold' : 'text-red-400 font-bold'}>
                {wsConnected ? 'SYNCED' : 'CONNECTING'}
              </span>
            </div>

            <button
              onClick={(e) => {
                e.stopPropagation();
                initAudio();
                setAudioMuted(!audioMuted);
              }}
              className={`p-1.5 rounded-xl border text-xs flex items-center space-x-1 transition ${
                audioMuted ? 'bg-slate-900 text-slate-400 border-slate-800' : 'bg-cyan-950 text-cyan-400 border-cyan-800'
              }`}
            >
              {audioMuted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
            </button>
          </div>
        </header>

        {/* MAIN HUD CLUSTER */}
        <div className="grid grid-cols-12 gap-4 flex-1 my-3 overflow-hidden items-stretch">
          
          {/* LEFT TELEMETRY TILES (6 COLUMNS) */}
          <div className="col-span-6 flex flex-col justify-between space-y-3">
            
            {/* ROW 1: ALERTNESS SPEEDOMETER + YAWN PIP TILE */}
            <div className="grid grid-cols-2 gap-3 flex-1">
              
              {/* SPEEDOMETER */}
              <div className="p-3 rounded-2xl border border-cyan-900/40 bg-[#0B0F19]/90 backdrop-blur-md flex flex-col items-center justify-between shadow-xl">
                <span className="text-slate-400 font-sans uppercase font-bold text-[11px] flex items-center gap-1.5">
                  <Zap className="w-3.5 h-3.5 text-cyan-400" /> Alertness Index
                </span>
                <div className="relative flex items-center justify-center my-1">
                  <svg className="w-28 h-28 transform -rotate-90">
                    <circle
                      cx="56"
                      cy="56"
                      r={radius}
                      stroke="rgba(30, 41, 59, 0.6)"
                      strokeWidth="8"
                      fill="transparent"
                    />
                    <circle
                      cx="56"
                      cy="56"
                      r={radius}
                      stroke={alertnessScore > 70 ? "#00F0FF" : (alertnessScore > 40 ? "#FFB800" : "#FF0055")}
                      strokeWidth="8"
                      strokeDasharray={circumference}
                      strokeDashoffset={strokeDashoffset}
                      strokeLinecap="round"
                      fill="transparent"
                      className="transition-all duration-300"
                    />
                  </svg>
                  <div className="absolute flex flex-col items-center">
                    <span className={`text-2xl font-black font-sans ${alertnessScore > 70 ? "text-cyan-400" : (alertnessScore > 40 ? "text-amber-400" : "text-red-500")}`}>
                      {alertnessScore}%
                    </span>
                    <span className="text-[9px] text-slate-400 uppercase tracking-widest font-bold">
                      {alertnessScore > 70 ? "ATTENTIVE" : (alertnessScore > 40 ? "DROWSY" : "CRITICAL")}
                    </span>
                  </div>
                </div>
                <span className="text-[10px] text-cyan-400/80 font-mono">EDGE LOGGING NOMINAL</span>
              </div>

              {/* YAWN COUNTER */}
              <div className={`p-3 rounded-2xl border flex flex-col justify-between transition-all shadow-xl ${
                yawnCount >= 3 ? 'bg-red-950/40 border-red-500/70 shadow-[0_0_20px_rgba(255,0,0,0.3)]' : 'bg-[#0B0F19]/90 border-cyan-900/40'
              }`}>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-400 font-sans uppercase font-bold flex items-center gap-1 text-[11px]">
                    <Flame className="w-3.5 h-3.5 text-amber-400" /> Yawns (2m Window)
                  </span>
                  <span className="text-[9px] text-amber-400 bg-amber-950/80 px-1.5 py-0.5 rounded border border-amber-900 font-mono">
                    FR-5
                  </span>
                </div>

                <div className="flex items-baseline justify-between my-1">
                  <span className={`text-3xl font-black font-sans ${yawnCount >= 3 ? 'text-red-400 animate-pulse' : 'text-amber-400'}`}>
                    {yawnCount} <span className="text-sm text-slate-500 font-normal">/ 3</span>
                  </span>
                  <span className="text-[10px] text-slate-400">Limit: 3/120s</span>
                </div>

                <div className="grid grid-cols-3 gap-2">
                  {[1, 2, 3].map((num) => (
                    <div
                      key={num}
                      className={`h-9 rounded-xl flex flex-col items-center justify-center font-sans font-bold text-xs border transition-all ${
                        num <= yawnCount 
                          ? 'bg-amber-500/20 text-amber-300 border-amber-400 shadow-[0_0_10px_rgba(245,158,11,0.5)]' 
                          : 'bg-slate-950 text-slate-600 border-slate-800'
                      }`}
                    >
                      <span className="text-[8px] text-slate-400">YAWN</span>
                      <span>#{num}</span>
                    </div>
                  ))}
                </div>

                <span className="text-[10px] text-slate-500 font-mono flex justify-between">
                  <span>Auto-Reset Timer</span>
                  <span>120s Window</span>
                </span>
              </div>

            </div>

            {/* ROW 2: EAR & MAR */}
            <div className="grid grid-cols-2 gap-3 flex-1">
              
              {/* EAR CARD */}
              <div className={`p-3 rounded-2xl border flex flex-col justify-between shadow-xl ${
                ear < earThreshold ? 'bg-red-950/40 border-red-500 shadow-[0_0_20px_rgba(255,0,85,0.3)]' : 'bg-[#0B0F19]/90 border-cyan-900/40'
              }`}>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-400 font-sans uppercase font-bold flex items-center gap-1.5 text-[11px]">
                    <Eye className="w-3.5 h-3.5 text-cyan-400" /> EAR Metric
                  </span>
                  <span className="text-[10px] bg-slate-900 text-slate-400 px-1.5 py-0.5 rounded border border-slate-800">
                    &lt; {earThreshold}
                  </span>
                </div>
                <div className="flex items-baseline justify-between my-1">
                  <span className={`text-3xl font-black font-sans ${ear < earThreshold ? 'text-red-400' : 'text-cyan-400'}`}>
                    {ear.toFixed(3)}
                  </span>
                  <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
                    ear < earThreshold ? 'bg-red-900 text-red-200 animate-pulse' : 'bg-cyan-950 text-cyan-300 border border-cyan-800'
                  }`}>
                    {ear < earThreshold ? 'CLOSED' : 'OPEN'}
                  </span>
                </div>
                <div className="w-full bg-slate-950 h-2 rounded-full overflow-hidden border border-slate-800">
                  <div 
                    className={`h-full transition-all duration-200 ${ear < earThreshold ? 'bg-red-500 shadow-[0_0_10px_#f00]' : 'bg-cyan-400'}`}
                    style={{ width: `${Math.min(100, (ear / 0.4) * 100)}%` }}
                  />
                </div>
              </div>

              {/* MAR CARD */}
              <div className={`p-3 rounded-2xl border flex flex-col justify-between shadow-xl ${
                mar > marThreshold ? 'bg-amber-950/40 border-amber-500 shadow-[0_0_20px_rgba(245,158,11,0.3)]' : 'bg-[#0B0F19]/90 border-cyan-900/40'
              }`}>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-400 font-sans uppercase font-bold flex items-center gap-1.5 text-[11px]">
                    <Activity className="w-3.5 h-3.5 text-amber-400" /> MAR Metric
                  </span>
                  <span className="text-[10px] bg-slate-900 text-slate-400 px-1.5 py-0.5 rounded border border-slate-800">
                    &gt; {marThreshold}
                  </span>
                </div>
                <div className="flex items-baseline justify-between my-1">
                  <span className={`text-3xl font-black font-sans ${mar > marThreshold ? 'text-amber-400' : 'text-slate-200'}`}>
                    {mar.toFixed(3)}
                  </span>
                  <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
                    mar > marThreshold ? 'bg-amber-950 text-amber-300 border border-amber-800 animate-pulse' : 'bg-slate-900 text-slate-400'
                  }`}>
                    {mar > marThreshold ? 'YAWN' : 'NORMAL'}
                  </span>
                </div>
                <div className="w-full bg-slate-950 h-2 rounded-full overflow-hidden border border-slate-800">
                  <div 
                    className={`h-full transition-all duration-200 ${mar > marThreshold ? 'bg-amber-400 shadow-[0_0_10px_#fa0]' : 'bg-emerald-500'}`}
                    style={{ width: `${Math.min(100, (mar / 0.8) * 100)}%` }}
                  />
                </div>
              </div>

            </div>

            {/* ROW 3: CONSECUTIVE CLOSED FRAMES */}
            <div className="p-3 rounded-2xl border border-cyan-900/40 bg-[#0B0F19]/90 backdrop-blur-md shadow-xl">
              <div className="flex items-center justify-between text-xs mb-1">
                <span className="text-slate-400 font-sans uppercase font-bold flex items-center gap-1.5 text-[11px]">
                  <Terminal className="w-3.5 h-3.5 text-purple-400" /> Eyelid Closure Progress (FR-4)
                </span>
                <span className="text-[10px] text-purple-400 font-bold bg-purple-950/60 px-2 py-0.5 rounded border border-purple-900">
                  {frameThreshold} FRAMES = 1.5s
                </span>
              </div>
              <div className="flex items-baseline justify-between my-1">
                <span className={`text-2xl font-black font-sans ${consecutiveLowEarFrames >= frameThreshold ? 'text-red-500 animate-bounce' : 'text-purple-300'}`}>
                  {consecutiveLowEarFrames} <span className="text-xs text-slate-500 font-normal">/ {frameThreshold} f</span>
                </span>
                <span className="text-[11px] text-slate-400">
                  {((consecutiveLowEarFrames / 30)).toFixed(1)}s Elapsed
                </span>
              </div>
              <div className="w-full bg-slate-950 h-2.5 rounded-full overflow-hidden border border-slate-800 flex">
                <div 
                  className={`h-full rounded-full transition-all duration-100 ${
                    consecutiveLowEarFrames >= frameThreshold ? 'bg-red-600 animate-pulse' : (consecutiveLowEarFrames > 20 ? 'bg-amber-400' : 'bg-purple-500')
                  }`}
                  style={{ width: `${Math.min(100, (consecutiveLowEarFrames / frameThreshold) * 100)}%` }}
                />
              </div>
            </div>

          </div>

          {/* RIGHT LIVE CHART (6 COLUMNS) */}
          <div className="col-span-6 bg-[#0B0F19]/90 border border-cyan-900/50 rounded-2xl p-4 shadow-xl backdrop-blur-md flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center space-x-2">
                  <TrendingUp className="w-4 h-4 text-cyan-400" />
                  <span className="text-xs uppercase font-extrabold tracking-wider text-slate-200 font-sans">
                    Real-Time Fatigue Telemetry (EAR vs MAR)
                  </span>
                </div>

                <div className="flex items-center space-x-3 text-[11px]">
                  <div className="flex items-center space-x-1.5">
                    <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 inline-block shadow-[0_0_8px_#00F0FF]" />
                    <span className="text-cyan-300 font-bold">EAR</span>
                  </div>
                  <div className="flex items-center space-x-1.5">
                    <span className="w-2.5 h-2.5 rounded-full bg-amber-400 inline-block shadow-[0_0_8px_#FFB800]" />
                    <span className="text-amber-300 font-bold">MAR</span>
                  </div>
                </div>
              </div>
              <p className="text-[11px] text-slate-500 font-sans">
                Continuous eye aspect ratio vs mouth aspect ratio telemetry streams logged at ~25Hz.
              </p>
            </div>

            <div className="h-[280px] w-full my-2">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={telemetryHistory} margin={{ top: 8, right: 12, left: -25, bottom: 0 }}>
                  <XAxis dataKey="time" stroke="#334155" fontSize={9} tickLine={false} />
                  <YAxis domain={[0, 0.65]} stroke="#334155" fontSize={9} tickLine={false} />
                  
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: 'rgba(11, 15, 25, 0.95)', 
                      borderColor: '#00F0FF33', 
                      borderRadius: '10px', 
                      fontSize: '11px',
                      boxShadow: '0 0 15px rgba(0,0,0,0.8)' 
                    }}
                    labelStyle={{ color: '#94A3B8' }}
                  />

                  <ReferenceLine 
                    y={earThreshold} 
                    stroke="#FF0055" 
                    strokeDasharray="3 3" 
                    label={{ value: 'EAR Limit', fill: '#FF0055', fontSize: 9, position: 'right' }} 
                  />

                  <ReferenceLine 
                    y={marThreshold} 
                    stroke="#FFB800" 
                    strokeDasharray="3 3" 
                    label={{ value: 'Yawn Limit', fill: '#FFB800', fontSize: 9, position: 'right' }} 
                  />

                  <Line type="monotone" dataKey="EAR" stroke="#00F0FF" strokeWidth={2} dot={false} isAnimationActive={false} />
                  <Line type="monotone" dataKey="MAR" stroke="#FFB800" strokeWidth={2} dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="pt-2 border-t border-slate-800/80 flex items-center justify-between text-[11px] text-slate-500 font-mono">
              <span>STREAM BUFFER: 30 SAMPLES</span>
              <span className="text-cyan-400 font-bold">CLIENT CAMERA TELEMETRY ACTIVE</span>
            </div>
          </div>

        </div>

        {/* BOTTOM NOTCH SCROLL-DOWN INDICATOR */}
        <div className="border-t border-cyan-900/40 pt-2 flex items-center justify-between text-[11px] text-slate-500">
          <span>VIGIL-AI AUTOMOTIVE TELEMETRY CLUSTER</span>
          
          <button 
            onClick={() => window.scrollTo({ top: window.innerHeight, behavior: 'smooth' })}
            className="flex items-center space-x-1.5 text-cyan-400 hover:text-cyan-300 font-bold bg-cyan-950/60 px-3 py-1 rounded-full border border-cyan-800 transition"
          >
            <span>SCROLL DOWN FOR PROCESSED CAMERA TELECAST</span>
            <ChevronDown className="w-4 h-4 animate-bounce" />
          </button>

          <span className="text-cyan-400 font-bold">FR-4 • FR-5 • FR-6</span>
        </div>

      </div>

      {/* SECTION 2: LIVE CAMERA TELECAST (Scrolled view) */}
      <div className="min-h-screen w-full p-6 flex flex-col justify-center items-center relative z-10 border-t-2 border-cyan-900/40 bg-[#070A12]/95 backdrop-blur-lg">
        <div className="w-full max-w-4xl bg-[#0B0F19]/90 border border-cyan-900/60 rounded-3xl p-6 shadow-2xl flex flex-col justify-between">
          
          <div className="w-full flex items-center justify-between mb-4">
            <div className="flex items-center space-x-2">
              <Camera className="w-4 h-4 text-cyan-400" />
              <h2 className="text-sm uppercase font-extrabold tracking-wider text-slate-200 font-sans">
                Driver Diagnostic Camera Telecast (Local Device Stream)
              </h2>
            </div>

            <div className="flex items-center space-x-3 text-xs text-cyan-400 font-mono">
              <span className="bg-cyan-950 px-2.5 py-1 rounded-lg border border-cyan-800">
                FPS: {fps}
              </span>
              <span className="bg-cyan-950 px-2.5 py-1 rounded-lg border border-cyan-800">
                {cameraActive ? "WEBCAM: ACTIVE" : "WEBCAM: WAITING"}
              </span>
            </div>
          </div>

          <div className="relative w-full aspect-video bg-black rounded-2xl overflow-hidden border border-cyan-900/60 flex items-center justify-center shadow-2xl">
            {camFrame ? (
              <img 
                src={camFrame} 
                alt="Driver Diagnostic Stream" 
                className="w-full h-full object-contain" 
              />
            ) : (
              <div className="text-cyan-500 text-sm flex flex-col items-center space-y-3">
                <Compass className="w-8 h-8 animate-spin" />
                <span className="tracking-widest">AWAITING BROWSER CAMERA STREAM...</span>
              </div>
            )}

            <div className="absolute top-4 left-4 w-8 h-8 border-t-2 border-l-2 border-cyan-400 pointer-events-none" />
            <div className="absolute top-4 right-4 w-8 h-8 border-t-2 border-r-2 border-cyan-400 pointer-events-none" />
            <div className="absolute bottom-4 left-4 w-8 h-8 border-b-2 border-l-2 border-cyan-400 pointer-events-none" />
            <div className="absolute bottom-4 right-4 w-8 h-8 border-b-2 border-r-2 border-cyan-400 pointer-events-none" />
          </div>

          <div className="mt-4 pt-3 border-t border-slate-800 flex items-center justify-between text-xs text-slate-400">
            <span>CLIENT HARDWARE CAMERA CAPTURE VIA WEBRTC / MEDIA-DEVICES</span>
            <button 
              onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
              className="text-cyan-400 hover:underline font-bold"
            >
              ↑ Back to Cockpit Dashboard
            </button>
          </div>
        </div>
      </div>

    </div>
  );
}