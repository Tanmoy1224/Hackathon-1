import React, { useState, useEffect } from 'react';
import { ShieldAlert, Eye, Activity, Zap, Terminal, AlertTriangle, Volume2, VolumeX, Flame } from 'lucide-react';

let audioCtx = null;
let alarmOscillator = null;
let alarmGain = null;

const start85dBAlarm = () => {
  try {
    if (!audioCtx) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioCtx.state === 'suspended') {
      audioCtx.resume();
    }
    if (alarmOscillator) return;

    alarmOscillator = audioCtx.createOscillator();
    alarmGain = audioCtx.createGain();

    alarmOscillator.type = 'sawtooth';
    alarmOscillator.frequency.setValueAtTime(880, audioCtx.currentTime);
    alarmGain.gain.setValueAtTime(0.35, audioCtx.currentTime);

    alarmOscillator.connect(alarmGain);
    alarmGain.connect(audioCtx.destination);
    alarmOscillator.start();
  } catch (err) {
    console.warn("Audio synthesis error:", err);
  }
};

const stop85dBAlarm = () => {
  if (alarmOscillator) {
    try {
      alarmOscillator.stop();
      alarmOscillator.disconnect();
    } catch(e) {}
    alarmOscillator = null;
  }
};

export default function App() {
  const [wsConnected, setWsConnected] = useState(false);
  const [ear, setEar] = useState(0.29);
  const [mar, setMar] = useState(0.18);
  const [alertnessScore, setAlertnessScore] = useState(98);
  const [consecutiveLowEarFrames, setConsecutiveLowEarFrames] = useState(0);
  const [isMicrosleep, setIsMicrosleep] = useState(false);
  const [isYawnWarning, setIsYawnWarning] = useState(false);
  const [yawnCount, setYawnCount] = useState(0);
  const [camFrame, setCamFrame] = useState(null);
  const [audioMuted, setAudioMuted] = useState(false);

  const earThreshold = 0.22;
  const marThreshold = 0.45;
  const frameThreshold = 45;

  // Trigger continuous siren on microsleep or 3-yawn fatigue alert
  useEffect(() => {
    if ((isMicrosleep || isYawnWarning || alertnessScore < 35) && !audioMuted) {
      start85dBAlarm();
    } else {
      stop85dBAlarm();
    }
    return () => stop85dBAlarm();
  }, [isMicrosleep, isYawnWarning, alertnessScore, audioMuted]);

  useEffect(() => {
    let socket;
    const connectWs = () => {
      socket = new WebSocket('ws://localhost:8000/ws/telemetry');
      socket.onopen = () => setWsConnected(true);
      socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.ear !== undefined) setEar(data.ear);
          if (data.mar !== undefined) setMar(data.mar);
          if (data.alertness !== undefined) setAlertnessScore(data.alertness);
          if (data.consecutive_frames !== undefined) setConsecutiveLowEarFrames(data.consecutive_frames);
          if (data.microsleep !== undefined) setIsMicrosleep(data.microsleep);
          if (data.yawn_warning !== undefined) setIsYawnWarning(data.yawn_warning);
          if (data.yawn_count !== undefined) setYawnCount(data.yawn_count);
          if (data.frame) setCamFrame(data.frame);
        } catch (err) {
          console.error("Payload parse error:", err);
        }
      };
      socket.onclose = () => {
        setWsConnected(false);
        setTimeout(connectWs, 2000);
      };
      socket.onerror = () => setWsConnected(false);
    };

    connectWs();
    return () => socket && socket.close();
  }, []);

  return (
    <div className="min-h-screen bg-[#070A10] text-slate-100 font-sans p-4 md:p-6 flex flex-col justify-between">
      
      {/* 1. MICROSLEEP ALERT BANNER */}
      {isMicrosleep && (
        <div className="bg-red-600/90 text-white py-2.5 px-4 rounded-xl border border-red-400 flex items-center justify-between animate-pulse mb-4 shadow-[0_0_25px_rgba(255,0,0,0.5)]">
          <div className="flex items-center space-x-3">
            <AlertTriangle className="w-6 h-6 text-yellow-300 animate-bounce flex-shrink-0" />
            <span className="font-extrabold tracking-wider text-base md:text-lg">
              CRITICAL WARNING: MICROSLEEP DETECTED!
            </span>
          </div>
          <span className="text-xs bg-black/40 px-3 py-1 rounded border border-red-300">
            {consecutiveLowEarFrames} / {frameThreshold} FRAMES
          </span>
        </div>
      )}

      {/* 2. EXCESSIVE YAWNING ALERT BANNER */}
      {isYawnWarning && !isMicrosleep && (
        <div className="bg-amber-600/90 text-white py-2.5 px-4 rounded-xl border border-amber-400 flex items-center justify-between animate-pulse mb-4 shadow-[0_0_25px_rgba(245,158,11,0.5)]">
          <div className="flex items-center space-x-3">
            <Flame className="w-6 h-6 text-yellow-200 animate-bounce flex-shrink-0" />
            <span className="font-extrabold tracking-wider text-base md:text-lg">
              FATIGUE ALERT: REST REQUIRED (3 YAWNS IN 2 MINS)
            </span>
          </div>
          <span className="text-xs bg-black/40 px-3 py-1 rounded border border-amber-300">
            HIGH ACCIDENT RISK
          </span>
        </div>
      )}

      {/* HEADER BAR */}
      <header className="flex items-center justify-between border-b border-cyan-900/40 pb-4 mb-4">
        <div className="flex items-center space-x-3">
          <div className="p-2 rounded-lg bg-cyan-950 border border-cyan-500/30 text-cyan-400">
            <ShieldAlert className="w-6 h-6" />
          </div>
          <div>
            <h1 className="font-bold text-xl tracking-tight bg-gradient-to-r from-cyan-400 to-emerald-400 bg-clip-text text-transparent">
              DRIVEGUARD AI
            </h1>
            <p className="text-xs text-slate-400">Driver Fatigue & Microsleep Monitor</p>
          </div>
        </div>

        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2 bg-slate-900 px-3 py-1.5 rounded-lg border border-slate-800 text-xs">
            <span className={`w-2 h-2 rounded-full ${wsConnected ? 'bg-emerald-400 animate-ping' : 'bg-red-500'}`} />
            <span className={wsConnected ? 'text-emerald-400 font-bold' : 'text-red-400 font-bold'}>
              {wsConnected ? 'LIVE FEED CONNECTED' : 'DISCONNECTED'}
            </span>
          </div>

          <button
            onClick={() => setAudioMuted(!audioMuted)}
            className={`p-2 rounded-lg border text-xs flex items-center space-x-1.5 transition ${
              audioMuted 
                ? 'bg-slate-900 text-slate-400 border-slate-800' 
                : 'bg-cyan-950 text-cyan-400 border-cyan-800'
            }`}
          >
            {audioMuted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
            <span>{audioMuted ? "MUTED" : "ALARM ON"}</span>
          </button>
        </div>
      </header>

      {/* CENTER WORKSPACE: 5 METRIC TILES + CAMERA */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 flex-1 items-start">
        
        {/* TELEMETRY GAUGES (LEFT 6 COLUMNS) */}
        <div className="lg:col-span-6 grid grid-cols-1 sm:grid-cols-2 gap-4">
          
          {/* DRIVER ALERTNESS INDEX */}
          <div className="p-4 rounded-xl border border-slate-800 bg-[#0D131F] shadow-xl">
            <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
              <span className="font-semibold uppercase tracking-wider flex items-center gap-1.5">
                <Zap className="w-4 h-4 text-emerald-400" /> Alertness Index
              </span>
              <span className="text-[10px] text-slate-400">0 - 100%</span>
            </div>
            <div className="flex items-baseline justify-between mt-2">
              <span className={`text-3xl font-extrabold tracking-tight ${
                alertnessScore > 75 ? 'text-emerald-400' : (alertnessScore > 45 ? 'text-amber-400' : 'text-red-500')
              }`}>
                {alertnessScore}%
              </span>
              <span className={`text-xs font-semibold px-2 py-0.5 rounded ${
                alertnessScore > 75 
                  ? 'bg-emerald-950 text-emerald-300 border border-emerald-800' 
                  : (alertnessScore > 45 ? 'bg-amber-950 text-amber-300 border border-amber-800' : 'bg-red-950 text-red-300 border border-red-800')
              }`}>
                {alertnessScore > 75 ? 'OPTIMAL' : (alertnessScore > 45 ? 'FATIGUED' : 'CRITICAL')}
              </span>
            </div>
            <div className="w-full bg-slate-900 h-2 rounded-full mt-3 overflow-hidden border border-slate-800">
              <div 
                className={`h-full transition-all duration-300 ${
                  alertnessScore > 75 ? 'bg-emerald-400' : (alertnessScore > 45 ? 'bg-amber-400' : 'bg-red-500')
                }`}
                style={{ width: `${alertnessScore}%` }}
              />
            </div>
          </div>

          {/* YAWN FREQUENCY COUNTER (FR-5) */}
          <div className={`p-4 rounded-xl border transition-all shadow-xl ${
            yawnCount >= 3 
              ? 'bg-red-950/30 border-red-600/60 shadow-[0_0_20px_rgba(255,0,85,0.25)]' 
              : 'bg-[#0D131F] border-slate-800'
          }`}>
            <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
              <span className="font-semibold uppercase tracking-wider flex items-center gap-1.5">
                <Flame className="w-4 h-4 text-amber-400" /> Yawns (2m Window)
              </span>
              <span className="text-[10px] text-slate-400">Limit: 3 / 2min</span>
            </div>
            <div className="flex items-baseline justify-between mt-2">
              <span className={`text-3xl font-extrabold tracking-tight ${yawnCount >= 3 ? 'text-red-400' : 'text-amber-400'}`}>
                {yawnCount} <span className="text-base text-slate-500 font-normal">/ 3</span>
              </span>
              <div className="flex space-x-1">
                {[1, 2, 3].map((num) => (
                  <span
                    key={num}
                    className={`w-6 h-6 rounded flex items-center justify-center text-xs font-bold border ${
                      num <= yawnCount 
                        ? 'bg-amber-500 text-slate-950 border-amber-400' 
                        : 'bg-slate-900 text-slate-500 border-slate-800'
                    }`}
                  >
                    #{num}
                  </span>
                ))}
              </div>
            </div>
            <div className="w-full bg-slate-900 h-2 rounded-full mt-3 overflow-hidden border border-slate-800">
              <div 
                className={`h-full transition-all duration-300 ${yawnCount >= 3 ? 'bg-red-500' : 'bg-amber-400'}`}
                style={{ width: `${Math.min(100, (yawnCount / 3) * 100)}%` }}
              />
            </div>
          </div>

          {/* EYE ASPECT RATIO (EAR) */}
          <div className={`p-4 rounded-xl border transition-all shadow-xl ${
            ear < earThreshold 
              ? 'bg-red-950/30 border-red-600/60 shadow-[0_0_20px_rgba(255,0,85,0.25)]' 
              : 'bg-[#0D131F] border-slate-800'
          }`}>
            <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
              <span className="font-semibold uppercase tracking-wider flex items-center gap-1.5">
                <Eye className="w-4 h-4 text-cyan-400" /> EAR Metric
              </span>
              <span className="text-[10px] bg-slate-800 text-slate-300 px-1.5 py-0.5 rounded">
                Thresh: &lt; {earThreshold}
              </span>
            </div>
            <div className="flex items-baseline justify-between mt-2">
              <span className={`text-3xl font-extrabold tracking-tight ${ear < earThreshold ? 'text-red-400' : 'text-cyan-400'}`}>
                {ear.toFixed(3)}
              </span>
              <span className={`text-xs font-semibold px-2 py-0.5 rounded ${
                ear < earThreshold ? 'bg-red-900 text-red-200' : 'bg-emerald-950 text-emerald-400 border border-emerald-800'
              }`}>
                {ear < earThreshold ? 'CLOSED' : 'OPEN'}
              </span>
            </div>
            <div className="w-full bg-slate-900 h-2 rounded-full mt-3 overflow-hidden border border-slate-800">
              <div 
                className={`h-full transition-all duration-300 ${ear < earThreshold ? 'bg-red-500' : 'bg-cyan-400'}`}
                style={{ width: `${Math.min(100, (ear / 0.4) * 100)}%` }}
              />
            </div>
          </div>

          {/* MOUTH ASPECT RATIO (MAR) */}
          <div className={`p-4 rounded-xl border transition-all shadow-xl ${
            mar > marThreshold 
              ? 'bg-amber-950/30 border-amber-600/60 shadow-[0_0_20px_rgba(255,184,0,0.2)]' 
              : 'bg-[#0D131F] border-slate-800'
          }`}>
            <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
              <span className="font-semibold uppercase tracking-wider flex items-center gap-1.5">
                <Activity className="w-4 h-4 text-amber-400" /> MAR Metric
              </span>
              <span className="text-[10px] bg-slate-800 text-slate-300 px-1.5 py-0.5 rounded">
                Limit: &ge; {marThreshold}
              </span>
            </div>
            <div className="flex items-baseline justify-between mt-2">
              <span className={`text-3xl font-extrabold tracking-tight ${mar > marThreshold ? 'text-amber-400' : 'text-slate-200'}`}>
                {mar.toFixed(3)}
              </span>
              <span className={`text-xs font-semibold px-2 py-0.5 rounded ${
                mar > marThreshold ? 'bg-amber-950 text-amber-300 border border-amber-800' : 'bg-slate-900 text-slate-400'
              }`}>
                {mar > marThreshold ? 'YAWNING' : 'NORMAL'}
              </span>
            </div>
            <div className="w-full bg-slate-900 h-2 rounded-full mt-3 overflow-hidden border border-slate-800">
              <div 
                className={`h-full transition-all duration-300 ${mar > marThreshold ? 'bg-amber-400' : 'bg-emerald-500'}`}
                style={{ width: `${Math.min(100, (mar / 0.8) * 100)}%` }}
              />
            </div>
          </div>

          {/* DROWSY FRAME COUNTER (SPANNING FULL ROW) */}
          <div className="sm:col-span-2 p-4 rounded-xl border border-slate-800 bg-[#0D131F] shadow-xl">
            <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
              <span className="font-semibold uppercase tracking-wider flex items-center gap-1.5">
                <Terminal className="w-4 h-4 text-purple-400" /> Consecutive Closed Frames
              </span>
              <span className="text-[10px] text-slate-400">Limit: {frameThreshold} f (~1.5s)</span>
            </div>
            <div className="flex items-baseline justify-between mt-2">
              <span className={`text-3xl font-extrabold tracking-tight ${
                consecutiveLowEarFrames >= frameThreshold ? 'text-red-500' : 'text-purple-300'
              }`}>
                {consecutiveLowEarFrames} <span className="text-base text-slate-500 font-normal">/ {frameThreshold}</span>
              </span>
              <span className={`text-xs font-semibold px-2 py-0.5 rounded ${
                consecutiveLowEarFrames >= frameThreshold ? 'bg-red-950 text-red-300 border border-red-800' : 'bg-slate-900 text-slate-400'
              }`}>
                {consecutiveLowEarFrames >= frameThreshold ? 'TRIGGERED' : 'MONITORING'}
              </span>
            </div>
            <div className="w-full bg-slate-900 h-2 rounded-full mt-3 overflow-hidden border border-slate-800">
              <div 
                className={`h-full transition-all duration-200 ${
                  consecutiveLowEarFrames >= frameThreshold ? 'bg-red-600 animate-pulse' : 'bg-purple-500'
                }`}
                style={{ width: `${Math.min(100, (consecutiveLowEarFrames / frameThreshold) * 100)}%` }}
              />
            </div>
          </div>

        </div>

        {/* LIVE CAMERA FEED (RIGHT 6 COLUMNS) */}
        <div className="lg:col-span-6 bg-[#0D131F] border border-cyan-900/40 rounded-xl p-4 shadow-xl flex flex-col items-center justify-center">
          <div className="w-full flex items-center justify-between mb-3 px-1">
            <div className="flex items-center space-x-2">
              <span className="w-2.5 h-2.5 rounded-full bg-red-500 animate-pulse" />
              <span className="text-xs uppercase font-bold tracking-wider text-slate-300">
                Driver Diagnostic Camera Feed
              </span>
            </div>
            <span className="text-[10px] font-mono bg-cyan-950 text-cyan-400 px-2 py-0.5 rounded border border-cyan-900">
              MEDIA PIPE 3D MESH
            </span>
          </div>

          <div className="w-full aspect-video bg-black rounded-lg overflow-hidden border border-slate-800 flex items-center justify-center relative">
            {camFrame ? (
              <img 
                src={camFrame} 
                alt="Driver Camera Diagnostic" 
                className="w-full h-full object-cover" 
              />
            ) : (
              <div className="text-slate-600 text-xs font-mono animate-pulse">
                CONNECTING TO CAMERA...
              </div>
            )}
          </div>
        </div>

      </div>

      <footer className="mt-4 pt-3 border-t border-slate-900 text-center text-xs text-slate-600 font-mono">
        VIGIL-AI EDGE MONITORING CLUSTER • RUNNING AT &lt; 50MS LATENCY
      </footer>
    </div>
  );
}