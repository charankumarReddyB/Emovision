import { useState, useEffect, useRef, useCallback } from 'react'
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts'
import { EMOTIONS, EMOTION_COLORS, EMOTION_ICONS } from '../data'
import type { Emotion, DetectedPerson } from '../types'

const EMOTION_LIST: Emotion[] = ['Happy', 'Sad', 'Angry', 'Fear', 'Surprise', 'Disgust', 'Neutral']

function randomEmotion(): Emotion {
  return EMOTION_LIST[Math.floor(Math.random() * EMOTION_LIST.length)]
}
function randomConf(min = 70, max = 98) {
  return Math.round(min + Math.random() * (max - min))
}

function makePersons(count: number): DetectedPerson[] {
  const cols = Math.ceil(Math.sqrt(count))
  return Array.from({ length: count }, (_, i) => {
    const col = i % cols
    const row = Math.floor(i / cols)
    const cellW = 100 / cols
    const cellH = 100 / Math.ceil(count / cols)
    return {
      id: i + 1,
      emotion: randomEmotion(),
      confidence: randomConf(),
      x: col * cellW + cellW * 0.1,
      y: row * cellH + cellH * 0.05,
      w: cellW * 0.75,
      h: cellH * 0.85,
    }
  })
}

export default function LiveDetection() {
  const [cameraOn, setCameraOn] = useState(false)
  const [analysisOn, setAnalysisOn] = useState(false)
  const [sessionActive, setSessionActive] = useState(false)
  const [persons, setPersons] = useState<DetectedPerson[]>([])
  const [fps, setFps] = useState(0)
  const [duration, setDuration] = useState(0)
  const [personCount, setPersonCount] = useState(3)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const fpsRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const startCamera = useCallback(() => {
    setCameraOn(true)
    const p = makePersons(personCount)
    setPersons(p)
  }, [personCount])

  const stopCamera = useCallback(() => {
    setCameraOn(false)
    setAnalysisOn(false)
    setPersons([])
    if (intervalRef.current) clearInterval(intervalRef.current)
  }, [])

  const startAnalysis = useCallback(() => {
    if (!cameraOn) return
    setAnalysisOn(true)
    setSessionActive(true)
    setDuration(0)

    timerRef.current = setInterval(() => setDuration(d => d + 1), 1000)
    fpsRef.current = setInterval(() => setFps(Math.round(27 + Math.random() * 6)), 800)

    intervalRef.current = setInterval(() => {
      setPersons(prev => {
        const count = Math.max(1, prev.length + (Math.random() > 0.7 ? Math.round(Math.random() * 2 - 1) : 0))
        const clamped = Math.min(8, Math.max(1, count))
        return makePersons(clamped)
      })
    }, 1800)
  }, [cameraOn])

  const endSession = useCallback(() => {
    setSessionActive(false)
    setAnalysisOn(false)
    setCameraOn(false)
    setPersons([])
    setFps(0)
    setDuration(0)
    if (intervalRef.current) clearInterval(intervalRef.current)
    if (timerRef.current) clearInterval(timerRef.current)
    if (fpsRef.current) clearInterval(fpsRef.current)
  }, [])

  useEffect(() => () => {
    if (intervalRef.current) clearInterval(intervalRef.current)
    if (timerRef.current) clearInterval(timerRef.current)
    if (fpsRef.current) clearInterval(fpsRef.current)
  }, [])

  const emotionCounts: Record<Emotion, number> = { Happy: 0, Sad: 0, Angry: 0, Fear: 0, Surprise: 0, Disgust: 0, Neutral: 0 }
  persons.forEach(p => { emotionCounts[p.emotion]++ })
  const avgConf = persons.length ? Math.round(persons.reduce((a, p) => a + p.confidence, 0) / persons.length) : 0

  const durationStr = (() => {
    const h = Math.floor(duration / 3600)
    const m = Math.floor((duration % 3600) / 60)
    const s = duration % 60
    return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
  })()

  const donutData = EMOTION_LIST.map(e => ({ name: e, value: emotionCounts[e] })).filter(d => d.value > 0)

  return (
    <div className="flex-1 flex flex-col overflow-hidden animate-fade-in">
      {/* Top bar */}
      <div className="px-5 py-3 flex items-center gap-4 border-b" style={{ borderColor: 'rgba(0,212,255,0.08)', background: '#09101d' }}>
        <h2 className="text-sm font-bold tracking-wide" style={{ color: '#e2e8f0' }}>Live Detection</h2>
        {sessionActive && (
          <div className="flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full" style={{ background: 'rgba(239,68,68,0.12)', border: '1px solid rgba(239,68,68,0.25)', color: '#f87171' }}>
            <span className="w-1.5 h-1.5 rounded-full bg-red-400" style={{ animation: 'pulse-dot 1s ease-in-out infinite' }} />
            REC
          </div>
        )}
        <div className="ml-auto flex items-center gap-2">
          <select
            value={personCount}
            onChange={e => setPersonCount(Number(e.target.value))}
            disabled={cameraOn}
            className="text-xs px-2 py-1 rounded"
            style={{ background: '#0d1424', color: '#94a3b8', border: '1px solid rgba(0,212,255,0.15)' }}
          >
            {[1, 2, 3, 4, 5, 6, 7, 8].map(n => (
              <option key={n} value={n}>{n} person{n > 1 ? 's' : ''}</option>
            ))}
          </select>
          <button
            onClick={startCamera}
            disabled={cameraOn}
            className="px-3 py-1.5 rounded text-xs font-medium transition-all"
            style={{ background: cameraOn ? 'rgba(0,212,255,0.08)' : 'rgba(0,212,255,0.15)', color: cameraOn ? '#475569' : '#00d4ff', border: '1px solid rgba(0,212,255,0.2)' }}
          >Start Camera</button>
          <button
            onClick={stopCamera}
            disabled={!cameraOn}
            className="px-3 py-1.5 rounded text-xs font-medium transition-all"
            style={{ background: !cameraOn ? 'rgba(100,116,139,0.05)' : 'rgba(239,68,68,0.1)', color: !cameraOn ? '#334155' : '#f87171', border: '1px solid rgba(239,68,68,0.15)' }}
          >Stop Camera</button>
          <button
            onClick={startAnalysis}
            disabled={!cameraOn || analysisOn}
            className="px-3 py-1.5 rounded text-xs font-medium transition-all"
            style={{ background: (!cameraOn || analysisOn) ? 'rgba(34,197,94,0.05)' : 'rgba(34,197,94,0.15)', color: (!cameraOn || analysisOn) ? '#334155' : '#4ade80', border: '1px solid rgba(34,197,94,0.2)' }}
          >Start Analysis</button>
          <button
            onClick={endSession}
            disabled={!sessionActive}
            className="px-3 py-1.5 rounded text-xs font-medium transition-all"
            style={{ background: !sessionActive ? 'rgba(168,85,247,0.05)' : 'rgba(168,85,247,0.15)', color: !sessionActive ? '#334155' : '#c084fc', border: '1px solid rgba(168,85,247,0.2)' }}
          >End Session</button>
        </div>
      </div>

      <div className="flex-1 flex gap-0 overflow-hidden">
        {/* Camera feed */}
        <div className="flex-1 p-4 flex flex-col gap-3">
          <div
            className="relative flex-1 rounded-xl overflow-hidden"
            style={{ background: '#040810', border: '1px solid rgba(0,212,255,0.12)', minHeight: 0 }}
          >
            {/* Grid overlay */}
            <div className="absolute inset-0" style={{
              backgroundImage: 'linear-gradient(rgba(0,212,255,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(0,212,255,0.03) 1px, transparent 1px)',
              backgroundSize: '40px 40px',
            }} />

            {/* Corner markers */}
            {[['top-3 left-3', 'border-t border-l'], ['top-3 right-3', 'border-t border-r'], ['bottom-3 left-3', 'border-b border-l'], ['bottom-3 right-3', 'border-b border-r']].map(([pos, brd], i) => (
              <div key={i} className={`absolute ${pos} w-6 h-6 ${brd}`} style={{ borderColor: 'rgba(0,212,255,0.4)' }} />
            ))}

            {!cameraOn && (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-3">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="rgba(0,212,255,0.3)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M15 10l4.553-2.069A1 1 0 0121 8.845v6.31a1 1 0 01-1.447.894L15 14" />
                  <rect x="3" y="6" width="12" height="12" rx="2" />
                </svg>
                <p className="text-xs" style={{ color: '#334155' }}>Camera inactive — click Start Camera to begin</p>
              </div>
            )}

            {/* Persons */}
            {cameraOn && persons.map(p => (
              <div
                key={p.id}
                className="absolute animate-bbox"
                style={{
                  left: `${p.x}%`, top: `${p.y}%`,
                  width: `${p.w}%`, height: `${p.h}%`,
                  border: `1.5px solid ${EMOTION_COLORS[p.emotion]}`,
                  borderRadius: 4,
                  boxShadow: `0 0 12px ${EMOTION_COLORS[p.emotion]}30`,
                }}
              >
                {/* Person label */}
                <div
                  className="absolute -top-6 left-0 flex items-center gap-1.5 px-2 py-0.5 rounded text-xs font-mono whitespace-nowrap"
                  style={{ background: `${EMOTION_COLORS[p.emotion]}22`, border: `1px solid ${EMOTION_COLORS[p.emotion]}55`, color: EMOTION_COLORS[p.emotion] }}
                >
                  <span>P{p.id}</span>
                  <span style={{ color: '#94a3b8' }}>·</span>
                  <span>{p.emotion}</span>
                  <span style={{ color: '#64748b' }}>{p.confidence}%</span>
                </div>
                {/* Corner dots */}
                {['-top-0.5 -left-0.5', '-top-0.5 -right-0.5', '-bottom-0.5 -left-0.5', '-bottom-0.5 -right-0.5'].map((pos, i) => (
                  <span key={i} className={`absolute ${pos} w-1.5 h-1.5 rounded-full`} style={{ background: EMOTION_COLORS[p.emotion] }} />
                ))}
              </div>
            ))}

            {/* Scan line when active */}
            {analysisOn && (
              <div
                className="absolute inset-x-0 h-px pointer-events-none"
                style={{
                  background: 'linear-gradient(90deg,transparent,rgba(0,212,255,0.6),transparent)',
                  animation: 'scan-line 2.5s linear infinite',
                  top: 0,
                }}
              />
            )}

            {/* FPS badge */}
            {cameraOn && (
              <div className="absolute top-3 left-1/2 -translate-x-1/2 flex items-center gap-2 px-3 py-1 rounded-full text-xs font-mono" style={{ background: 'rgba(8,12,20,0.85)', border: '1px solid rgba(0,212,255,0.15)', color: '#64748b' }}>
                {analysisOn && <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse-dot" />}
                {analysisOn ? `${fps} FPS` : 'Camera feed active'}
              </div>
            )}
          </div>
        </div>

        {/* Stats panel */}
        <div className="w-64 shrink-0 p-4 pl-0 flex flex-col gap-3 overflow-y-auto">
          <div className="text-xs font-semibold tracking-wider uppercase" style={{ color: '#475569' }}>Live Statistics</div>

          {/* People count */}
          <div className="rounded-xl p-3" style={{ background: '#0d1424', border: '1px solid rgba(0,212,255,0.1)' }}>
            <div className="text-xs mb-1" style={{ color: '#64748b' }}>People Detected</div>
            <div className="text-3xl font-bold font-mono" style={{ color: '#00d4ff' }}>{persons.length}</div>
          </div>

          {/* Emotion counts */}
          <div className="rounded-xl p-3 space-y-1.5" style={{ background: '#0d1424', border: '1px solid rgba(0,212,255,0.1)' }}>
            <div className="text-xs mb-2" style={{ color: '#64748b' }}>Emotion Breakdown</div>
            {EMOTION_LIST.map(e => (
              <div key={e} className="flex items-center gap-2 text-xs">
                <span>{EMOTION_ICONS[e]}</span>
                <span style={{ color: '#94a3b8', flex: 1 }}>{e}</span>
                <div className="w-16 h-1 rounded-full overflow-hidden" style={{ background: '#0a1120' }}>
                  <div className="h-full rounded-full transition-all duration-500" style={{ width: persons.length ? `${(emotionCounts[e] / persons.length) * 100}%` : '0%', background: EMOTION_COLORS[e] }} />
                </div>
                <span className="font-mono w-4 text-right" style={{ color: EMOTION_COLORS[e] }}>{emotionCounts[e]}</span>
              </div>
            ))}
          </div>

          {/* Metrics */}
          <div className="rounded-xl p-3 space-y-2" style={{ background: '#0d1424', border: '1px solid rgba(0,212,255,0.1)' }}>
            {[
              { label: 'Avg Confidence', value: `${avgConf}%`, color: '#22c55e' },
              { label: 'Current FPS', value: analysisOn ? `${fps}` : '—', color: '#00d4ff' },
              { label: 'Session Duration', value: sessionActive ? durationStr : '—', color: '#a855f7' },
            ].map(m => (
              <div key={m.label} className="flex items-center justify-between text-xs">
                <span style={{ color: '#64748b' }}>{m.label}</span>
                <span className="font-mono font-bold" style={{ color: m.color }}>{m.value}</span>
              </div>
            ))}
          </div>

          {/* Mini donut */}
          {persons.length > 0 && (
            <div className="rounded-xl p-3" style={{ background: '#0d1424', border: '1px solid rgba(0,212,255,0.1)' }}>
              <div className="text-xs mb-2" style={{ color: '#64748b' }}>Distribution</div>
              <ResponsiveContainer width="100%" height={100}>
                <PieChart>
                  <Pie data={donutData} cx="50%" cy="50%" innerRadius={28} outerRadius={45} paddingAngle={2} dataKey="value">
                    {donutData.map((entry) => (
                      <Cell key={entry.name} fill={EMOTION_COLORS[entry.name as Emotion]} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(v: any, n: any) => [v, n]} contentStyle={{ background: '#131e30', border: '1px solid rgba(0,212,255,0.15)', color: '#e2e8f0', fontSize: 11 }} />
                </PieChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Person IDs */}
          {persons.length > 0 && (
            <div className="rounded-xl p-3" style={{ background: '#0d1424', border: '1px solid rgba(0,212,255,0.1)' }}>
              <div className="text-xs mb-2" style={{ color: '#64748b' }}>Active Tracking IDs</div>
              <div className="flex flex-wrap gap-1.5">
                {persons.map(p => (
                  <span key={p.id} className="text-xs px-2 py-0.5 rounded font-mono" style={{ background: `${EMOTION_COLORS[p.emotion]}15`, border: `1px solid ${EMOTION_COLORS[p.emotion]}40`, color: EMOTION_COLORS[p.emotion] }}>
                    P{p.id}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
