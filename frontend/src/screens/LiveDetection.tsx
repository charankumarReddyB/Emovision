import { useState, useEffect, useRef, useCallback } from 'react'
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts'
import { EMOTIONS, EMOTION_COLORS, EMOTION_ICONS } from '../data'
import type { Emotion, PersonDetection } from '../types'
import { apiService } from '../services/api'
import { DetectionWebSocket, type WebSocketStatus } from '../services/websocket'

import ImageUploadAnalysis from './ImageUploadAnalysis'
import VideoUploadAnalysis from './VideoUploadAnalysis'

export default function LiveDetection() {
  const [activeMode, setActiveMode] = useState<'webcam' | 'image' | 'video'>('webcam')
  const [cameraOn, setCameraOn] = useState(false)
  const [sessionActive, setSessionActive] = useState(false)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [persons, setPersons] = useState<PersonDetection[]>([])
  const [fps, setFps] = useState(0)
  const [avgConfidence, setAvgConfidence] = useState(0)
  const [dominantExpression, setDominantExpression] = useState<string>('Neutral')
  const [duration, setDuration] = useState(0)
  const [wsStatus, setWsStatus] = useState<WebSocketStatus>('disconnected')
  const [errorMsg, setErrorMsg] = useState<string | null>(null)
  const [endedNotice, setEndedNotice] = useState<{ sessionId: string } | null>(null)

  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const wsRef = useRef<DetectionWebSocket | null>(null)
  const videoRef = useRef<HTMLVideoElement | null>(null)

  useEffect(() => {
    let stream: MediaStream | null = null
    if (cameraOn) {
      navigator.mediaDevices
        ?.getUserMedia({ video: { width: 1280, height: 720 } })
        .then((mediaStream) => {
          stream = mediaStream
          if (videoRef.current) {
            videoRef.current.srcObject = mediaStream
          }
        })
        .catch((err) => {
          console.warn('Webcam stream access:', err)
        })
    }

    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop())
      }
    }
  }, [cameraOn])

  // Stream live webcam frames to backend over WebSocket for real OpenCV face detection
  useEffect(() => {
    let interval: ReturnType<typeof setInterval> | null = null
    const canvas = document.createElement('canvas')
    canvas.width = 320
    canvas.height = 240
    const ctx = canvas.getContext('2d')

    if (cameraOn && sessionActive && wsStatus === 'connected') {
      interval = setInterval(() => {
        if (videoRef.current && wsRef.current && videoRef.current.readyState >= 2) {
          ctx?.drawImage(videoRef.current, 0, 0, 320, 240)
          const base64Img = canvas.toDataURL('image/jpeg', 0.35)
          wsRef.current.sendFrame(base64Img)
        }
      }, 100)
    }

    return () => {
      if (interval) clearInterval(interval)
    }
  }, [cameraOn, sessionActive, wsStatus])

  const handleMessage = useCallback((payload: any) => {
    if (payload.people) {
      setPersons(payload.people)
    }
    if (payload.fps !== undefined) {
      setFps(payload.fps)
    }
    if (payload.average_confidence !== undefined) {
      setAvgConfidence(payload.average_confidence)
    }
    if (payload.dominant_expression) {
      setDominantExpression(payload.dominant_expression)
    }
  }, [])

  const startSession = useCallback(async () => {
    try {
      setErrorMsg(null)
      const res = await apiService.startSession('Live Webcam Session', 'webcam')
      const sid = res.session_id
      setSessionId(sid)
      setSessionActive(true)
      setCameraOn(true)
      setDuration(0)

      if (timerRef.current) clearInterval(timerRef.current)
      timerRef.current = setInterval(() => setDuration((d) => d + 1), 1000)

      // Initialize WebSocket connection
      if (wsRef.current) {
        wsRef.current.disconnect()
      }

      const ws = new DetectionWebSocket({
        sessionId: sid,
        onMessage: handleMessage,
        onStatusChange: (status) => {
          setWsStatus(status)
          if (status === 'connected') {
            setErrorMsg(null)
          }
        },
        onError: (errEvent) => {
          console.error('[LiveDetection] WebSocket onError:', errEvent)
          setErrorMsg('WebSocket connection error (check browser console for details)')
        },
      })
      wsRef.current = ws
      ws.connect()
    } catch (err: any) {
      setErrorMsg(err.message || 'Failed to start detection session')
    }
  }, [handleMessage])

  const endSession = useCallback(() => {
    const finishedId = sessionId || 'sess_active'

    // 1. Immediately trigger popup notice overlay in 0ms!
    setEndedNotice({ sessionId: finishedId })

    // 2. Immediately reset live stream states & disconnect WebSocket
    setSessionActive(false)
    setSessionId(null)
    setPersons([])
    setFps(0)
    setAvgConfidence(0)
    setDuration(0)
    setWsStatus('disconnected')

    if (wsRef.current) {
      wsRef.current.disconnect()
      wsRef.current = null
    }

    if (timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }

    // 3. Asynchronously close session on backend in background without delaying UI popup
    if (finishedId && finishedId !== 'sess_active') {
      apiService
        .endSession(finishedId)
        .catch((err) => console.warn('Background session close notice:', err))
    }
  }, [sessionId])

  const startCamera = useCallback(() => {
    setCameraOn(true)
  }, [])

  const stopCamera = useCallback(() => {
    setCameraOn(false)
  }, [])

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
      if (wsRef.current) wsRef.current.disconnect()
    }
  }, [])

  // Emotion count breakdown
  const emotionCounts: Record<string, number> = {
    Happy: 0,
    Sad: 0,
    Angry: 0,
    Fear: 0,
    Surprise: 0,
    Disgust: 0,
    Neutral: 0,
  }

  persons.forEach((p) => {
    const emo = p.expression || 'Neutral'
    emotionCounts[emo] = (emotionCounts[emo] || 0) + 1
  })

  const durationStr = (() => {
    const h = Math.floor(duration / 3600)
    const m = Math.floor((duration % 3600) / 60)
    const s = duration % 60
    return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
  })()

  const donutData = EMOTIONS.map((e) => ({ name: e, value: emotionCounts[e] || 0 })).filter(
    (d) => d.value > 0
  )

  return (
    <div className="flex-1 flex flex-col overflow-hidden animate-fade-in">
      {/* Top bar */}
      <div
        className="px-5 py-3 flex items-center justify-between border-b"
        style={{ borderColor: 'rgba(0,212,255,0.08)', background: '#09101d' }}
      >
        <div className="flex items-center gap-3">
          {/* Mode Selector Tabs */}
          <div className="flex items-center gap-1 p-1 rounded-xl bg-slate-900 border border-slate-800">
            <button
              onClick={() => setActiveMode('webcam')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                activeMode === 'webcam'
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              📹 Live Webcam
            </button>
            <button
              onClick={() => setActiveMode('image')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                activeMode === 'image'
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              📷 Upload Image
            </button>
            <button
              onClick={() => setActiveMode('video')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                activeMode === 'video'
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              🎬 Upload Video
            </button>
          </div>

          {sessionActive && (
            <div
              className="flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full font-mono"
              style={{
                background: 'rgba(239,68,68,0.12)',
                border: '1px solid rgba(239,68,68,0.25)',
                color: '#f87171',
              }}
            >
              <span className="w-1.5 h-1.5 rounded-full bg-red-400 animate-pulse" />
              REC · {sessionId}
            </div>
          )}
        </div>

        {/* WebSocket Connection Status */}
        <div
          className="flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full font-mono"
          style={{
            background:
              wsStatus === 'connected'
                ? 'rgba(34,197,94,0.12)'
                : wsStatus === 'connecting' || wsStatus === 'reconnecting'
                ? 'rgba(234,179,8,0.12)'
                : 'rgba(100,116,139,0.12)',
            border:
              wsStatus === 'connected'
                ? '1px solid rgba(34,197,94,0.3)'
                : wsStatus === 'connecting' || wsStatus === 'reconnecting'
                ? '1px solid rgba(234,179,8,0.3)'
                : '1px solid rgba(100,116,139,0.3)',
            color:
              wsStatus === 'connected'
                ? '#4ade80'
                : wsStatus === 'connecting' || wsStatus === 'reconnecting'
                ? '#fde047'
                : '#94a3b8',
          }}
        >
          <span
            className={`w-1.5 h-1.5 rounded-full ${
              wsStatus === 'connected' ? 'bg-green-400' : 'bg-yellow-400 animate-pulse'
            }`}
          />
          WS: {wsStatus.toUpperCase()}
        </div>

        <div className="ml-auto flex items-center gap-2">
          {!sessionActive ? (
            <button
              onClick={startSession}
              className="px-3.5 py-1.5 rounded text-xs font-semibold transition-all"
              style={{
                background: 'linear-gradient(135deg,#00d4ff,#0098cc)',
                color: '#080c14',
              }}
            >
              Start Session
            </button>
          ) : (
            <>
              <button
                onClick={cameraOn ? stopCamera : startCamera}
                className="px-3 py-1.5 rounded text-xs font-medium transition-all"
                style={{
                  background: cameraOn ? 'rgba(239,68,68,0.1)' : 'rgba(0,212,255,0.15)',
                  color: cameraOn ? '#f87171' : '#00d4ff',
                  border: cameraOn
                    ? '1px solid rgba(239,68,68,0.2)'
                    : '1px solid rgba(0,212,255,0.2)',
                }}
              >
                {cameraOn ? 'Stop Camera' : 'Start Camera'}
              </button>
              <button
                onClick={endSession}
                className="px-3.5 py-1.5 rounded text-xs font-semibold transition-all"
                style={{
                  background: 'rgba(168,85,247,0.15)',
                  color: '#c084fc',
                  border: '1px solid rgba(168,85,247,0.3)',
                }}
              >
                End Session
              </button>
            </>
          )}
        </div>
      </div>

      {errorMsg && (
        <div
          className="mx-5 mt-2 px-3 py-1.5 rounded text-xs flex items-center gap-2"
          style={{ background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.2)', color: '#f87171' }}
        >
          Backend WS Error: {errorMsg}
        </div>
      )}

      {activeMode === 'image' && <ImageUploadAnalysis />}
      {activeMode === 'video' && <VideoUploadAnalysis />}
      {activeMode === 'webcam' && (
        <div className="flex-1 flex gap-0 overflow-hidden">
        {/* Camera feed viewport */}
        <div className="flex-1 p-4 flex flex-col gap-3">
          <div
            className="relative flex-1 rounded-xl overflow-hidden"
            style={{ background: '#040810', border: '1px solid rgba(0,212,255,0.12)', minHeight: 0 }}
          >
            {/* Live Webcam Stream */}
            {cameraOn && (
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="absolute inset-0 w-full h-full object-cover"
                style={{ opacity: 0.85 }}
              />
            )}

            {/* Cyber Grid overlay */}
            <div
              className="absolute inset-0"
              style={{
                backgroundImage:
                  'linear-gradient(rgba(0,212,255,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(0,212,255,0.03) 1px, transparent 1px)',
                backgroundSize: '40px 40px',
              }}
            />

            {/* Corner Markers */}
            {[
              ['top-3 left-3', 'border-t border-l'],
              ['top-3 right-3', 'border-t border-r'],
              ['bottom-3 left-3', 'border-b border-l'],
              ['bottom-3 right-3', 'border-b border-r'],
            ].map(([pos, brd], i) => (
              <div
                key={i}
                className={`absolute ${pos} w-6 h-6 ${brd}`}
                style={{ borderColor: 'rgba(0,212,255,0.4)' }}
              />
            ))}

            {!cameraOn && !sessionActive && (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-3">
                <svg
                  width="48"
                  height="48"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="rgba(0,212,255,0.3)"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="M15 10l4.553-2.069A1 1 0 0121 8.845v6.31a1 1 0 01-1.447.894L15 14" />
                  <rect x="3" y="6" width="12" height="12" rx="2" />
                </svg>
                <p className="text-xs" style={{ color: '#64748b' }}>
                  Click <strong className="text-cyan-400">Start Session</strong> to launch WebSocket detection pipeline
                </p>
              </div>
            )}

            {/* Dynamic N-Person Bounding Box Overlays */}
            {cameraOn &&
              persons.map((p) => {
                const emo = p.expression as Emotion
                const color = EMOTION_COLORS[emo] || '#00d4ff'
                const confPct =
                  p.confidence > 1 ? Math.round(p.confidence) : Math.round(p.confidence * 100)

                // Normalize bounding box coordinates
                const bbox = p.bounding_box || { x: 100, y: 100, width: 140, height: 140 }
                const frameW = (bbox as any).frame_width || 640
                const frameH = (bbox as any).frame_height || 480
                let left = (bbox.x / frameW) * 100
                let top = (bbox.y / frameH) * 100
                let width = (bbox.width / frameW) * 100
                let height = (bbox.height / frameH) * 100

                return (
                  <div
                    key={p.person_id}
                    className="absolute transition-all duration-300"
                    style={{
                      left: `${left}%`,
                      top: `${top}%`,
                      width: `${width}%`,
                      height: `${height}%`,
                      border: `2px solid ${color}`,
                      borderRadius: 4,
                      boxShadow: `0 0 14px ${color}40`,
                    }}
                  >
                    {/* Person Label Badge */}
                    <div
                      className="absolute -top-7 left-0 flex items-center gap-1.5 px-2 py-0.5 rounded text-xs font-mono whitespace-nowrap"
                      style={{
                        background: '#09101d',
                        border: `1px solid ${color}`,
                        color: color,
                      }}
                    >
                      <span>Face {p.face_index || p.person_id || index + 1}</span>
                      <span style={{ color: '#94a3b8' }}>—</span>
                      <span>{p.expression}</span>
                      <span style={{ color: '#94a3b8' }}>—</span>
                      <span>{confPct}%</span>
                    </div>

                    {/* Corner Dots */}
                    {[
                      '-top-1 -left-1',
                      '-top-1 -right-1',
                      '-bottom-1 -left-1',
                      '-bottom-1 -right-1',
                    ].map((pos, i) => (
                      <span
                        key={i}
                        className={`absolute ${pos} w-2 h-2 rounded-full`}
                        style={{ background: color }}
                      />
                    ))}
                  </div>
                )
              })}

            {/* Scan Line Animation */}
            {sessionActive && (
              <div
                className="absolute inset-x-0 h-px pointer-events-none"
                style={{
                  background:
                    'linear-gradient(90deg,transparent,rgba(0,212,255,0.6),transparent)',
                  animation: 'scan-line 2.5s linear infinite',
                  top: 0,
                }}
              />
            )}

            {/* Live Stats Overlay Badge */}
            {cameraOn && (
              <div
                className="absolute top-3 left-1/2 -translate-x-1/2 flex items-center gap-3 px-3 py-1 rounded-full text-xs font-mono"
                style={{
                  background: 'rgba(8,12,20,0.85)',
                  border: '1px solid rgba(0,212,255,0.15)',
                  color: '#64748b',
                }}
              >
                {sessionActive && (
                  <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
                )}
                <span>FPS: <strong className="text-cyan-400">{fps.toFixed(1)}</strong></span>
                <span>Dominant: <strong className="text-purple-400">{dominantExpression}</strong></span>
              </div>
            )}
          </div>
        </div>

        {/* Live Metrics Sidebar */}
        <div className="w-64 shrink-0 p-4 pl-0 flex flex-col gap-3 overflow-y-auto">
          <div className="text-xs font-semibold tracking-wider uppercase" style={{ color: '#475569' }}>
            Live Stream Metrics
          </div>

          {/* People Count */}
          <div
            className="rounded-xl p-3"
            style={{ background: '#0d1424', border: '1px solid rgba(0,212,255,0.1)' }}
          >
            <div className="text-xs mb-1" style={{ color: '#64748b' }}>
              Faces Detected (N)
            </div>
            <div className="text-3xl font-bold font-mono" style={{ color: '#00d4ff' }}>
              {persons.length}
            </div>
          </div>

          {/* Emotion Breakdown */}
          <div
            className="rounded-xl p-3 space-y-1.5"
            style={{ background: '#0d1424', border: '1px solid rgba(0,212,255,0.1)' }}
          >
            <div className="text-xs mb-2" style={{ color: '#64748b' }}>
              Emotion Breakdown
            </div>
            {EMOTIONS.map((e) => (
              <div key={e} className="flex items-center gap-2 text-xs">
                <span>{EMOTION_ICONS[e]}</span>
                <span style={{ color: '#94a3b8', flex: 1 }}>
                  {e}
                </span>
                <div
                  className="w-16 h-1.5 rounded-full overflow-hidden"
                  style={{ background: '#0a1120' }}
                >
                  <div
                    className="h-full rounded-full transition-all duration-500"
                    style={{
                      width: persons.length
                        ? `${((emotionCounts[e] || 0) / persons.length) * 100}%`
                        : '0%',
                      background: EMOTION_COLORS[e],
                    }}
                  />
                </div>
                <span className="font-mono w-4 text-right" style={{ color: EMOTION_COLORS[e] }}>
                  {emotionCounts[e] || 0}
                </span>
              </div>
            ))}
          </div>

          {/* Key Indicators */}
          <div
            className="rounded-xl p-3 space-y-2"
            style={{ background: '#0d1424', border: '1px solid rgba(0,212,255,0.1)' }}
          >
            {[
              {
                label: 'Avg Confidence',
                value: `${avgConfidence > 1 ? avgConfidence.toFixed(1) : (avgConfidence * 100).toFixed(1)}%`,
                color: '#22c55e',
              },
              { label: 'Current FPS', value: sessionActive ? fps.toFixed(1) : '—', color: '#00d4ff' },
              { label: 'Session Duration', value: sessionActive ? durationStr : '—', color: '#a855f7' },
              { label: 'Dominant Expression', value: dominantExpression, color: '#f97316' },
            ].map((m) => (
              <div key={m.label} className="flex items-center justify-between text-xs">
                <span style={{ color: '#64748b' }}>{m.label}</span>
                <span className="font-mono font-bold" style={{ color: m.color }}>
                  {m.value}
                </span>
              </div>
            ))}
          </div>

          {/* Mini Donut Chart */}
          {persons.length > 0 && (
            <div
              className="rounded-xl p-3"
              style={{ background: '#0d1424', border: '1px solid rgba(0,212,255,0.1)' }}
            >
              <div className="text-xs mb-2" style={{ color: '#64748b' }}>
                Distribution
              </div>
              <ResponsiveContainer width="100%" height={100}>
                <PieChart>
                  <Pie
                    data={donutData}
                    cx="50%"
                    cy="50%"
                    innerRadius={28}
                    outerRadius={45}
                    paddingAngle={2}
                    dataKey="value"
                  >
                    {donutData.map((entry) => (
                      <Cell key={entry.name} fill={EMOTION_COLORS[entry.name as Emotion]} />
                    ))}
                  </Pie>
                  <Tooltip
                    formatter={(v: any, n: any) => [v, n]}
                    contentStyle={{
                      background: '#131e30',
                      border: '1px solid rgba(0,212,255,0.15)',
                      color: '#e2e8f0',
                      fontSize: 11,
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </div>
      )}

      {/* Session Ended Toast Notification Popup */}
      {endedNotice && (
        <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/85 backdrop-blur-md p-4 animate-fade-in">
          <div className="bg-slate-950 border-2 border-cyan-400 p-6 rounded-2xl max-w-md w-full shadow-2xl text-center space-y-4 relative">
            <div className="w-16 h-16 bg-emerald-500/20 text-emerald-400 rounded-full flex items-center justify-center mx-auto text-3xl font-bold border border-emerald-500/40">
              ✓
            </div>
            <div>
              <h3 className="text-xl font-extrabold text-white tracking-tight">
                Session Saved to History!
              </h3>
              <p className="text-xs text-cyan-400 font-mono mt-1">
                ID: {endedNotice.sessionId}
              </p>
            </div>
            <p className="text-xs text-slate-300 leading-relaxed">
              The detection session has been logged with all Person ID face photo cards, expressions, and telemetry into database history.
            </p>
            <div className="flex items-center justify-center gap-3 pt-2">
              <button
                onClick={() => setEndedNotice(null)}
                className="px-6 py-2.5 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 text-xs font-bold rounded-xl shadow-lg shadow-cyan-500/30 transition transform active:scale-95"
              >
                Close & Continue
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
