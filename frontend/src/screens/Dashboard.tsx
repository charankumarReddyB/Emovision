import { useState, useEffect } from 'react'
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'
import { EMOTIONS, EMOTION_COLORS, EMOTION_ICONS } from '../data'
import type { Emotion, Screen, SessionSummary } from '../types'
import { apiService } from '../services/api'
import { formatLocalDateTime } from '../utils/date'

interface Props {
  onNavigate: (s: Screen) => void
}

function StatCard({
  icon,
  label,
  value,
  sub,
}: {
  icon: React.ReactNode
  label: string
  value: string | number
  sub?: string
}) {
  return (
    <div
      className="rounded-xl p-4 flex flex-col gap-2"
      style={{ background: '#0d1424', border: '1px solid rgba(0,212,255,0.1)' }}
    >
      <div className="flex items-center gap-2">
        <div
          className="w-8 h-8 rounded-lg flex items-center justify-center"
          style={{ background: 'rgba(0,212,255,0.08)' }}
        >
          {icon}
        </div>
        <span className="text-xs" style={{ color: '#64748b' }}>
          {label}
        </span>
      </div>
      <div
        className="text-2xl font-bold tracking-tight"
        style={{ color: '#e2e8f0', fontVariantNumeric: 'tabular-nums' }}
      >
        {value}
      </div>
      {sub && (
        <div className="text-xs" style={{ color: '#475569' }}>
          {sub}
        </div>
      )}
    </div>
  )
}

const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload?.length) {
    return (
      <div
        className="px-3 py-2 rounded-lg text-xs"
        style={{
          background: '#131e30',
          border: '1px solid rgba(0,212,255,0.15)',
          color: '#e2e8f0',
        }}
      >
        {payload[0].name}: <strong>{payload[0].value}</strong>
      </div>
    )
  }
  return null
}

export default function Dashboard({ onNavigate }: Props) {
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [backendStatus, setBackendStatus] = useState<string>('checking')

  useEffect(() => {
    let isMounted = true

    async function loadData() {
      try {
        setLoading(true)
        setError(null)
        const [healthRes, sessionsRes] = await Promise.all([
          apiService.getHealth().catch(() => null),
          apiService.getSessions(1, 50).catch(() => ({ total: 0, page: 1, limit: 50, sessions: [] })),
        ])

        if (isMounted) {
          if (healthRes) {
            setBackendStatus(healthRes.status === 'ok' ? 'Online' : 'Warning')
          } else {
            setBackendStatus('Offline')
          }
          setSessions(sessionsRes.sessions || [])
        }
      } catch (err: any) {
        if (isMounted) {
          setError(err.message || 'Failed to load backend data')
        }
      } finally {
        if (isMounted) setLoading(false)
      }
    }

    loadData()
    return () => {
      isMounted = false
    }
  }, [])

  // Calculate statistics from backend session data
  const totalSessions = sessions.length
  const totalFaces = sessions.reduce((acc, s) => acc + (s.people_count || 0), 0)
  const avgConf = totalSessions
    ? (sessions.reduce((acc, s) => acc + (s.average_confidence || 0), 0) / totalSessions).toFixed(1)
    : '0.0'

  const dominantCounts: Record<string, number> = {}
  sessions.forEach((s) => {
    if (s.dominant_expression) {
      dominantCounts[s.dominant_expression] = (dominantCounts[s.dominant_expression] || 0) + 1
    }
  })
  const mostDetectedEntry = Object.entries(dominantCounts).sort((a, b) => b[1] - a[1])[0]
  const mostDetected = mostDetectedEntry ? mostDetectedEntry[0] : 'None'

  // Combine expression counts across sessions
  const combinedEmotions: Record<string, number> = {
    Happy: 0,
    Sad: 0,
    Angry: 0,
    Fear: 0,
    Surprise: 0,
    Disgust: 0,
    Neutral: 0,
  }
  sessions.forEach((s) => {
    if (s.dominant_expression && combinedEmotions[s.dominant_expression] !== undefined) {
      combinedEmotions[s.dominant_expression] += 1
    }
  })

  const donutData = EMOTIONS.map((e) => ({ name: e, value: combinedEmotions[e] }))
  const barData = EMOTIONS.map((e) => ({
    name: e.slice(0, 3),
    value: combinedEmotions[e],
    color: EMOTION_COLORS[e as Emotion],
  }))

  const recentSessions = sessions.slice(0, 4)

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-xl font-bold tracking-tight" style={{ color: '#e2e8f0' }}>
            Computer Vision — Emotion Recognition Dashboard
          </h1>
          <p className="text-sm mt-0.5" style={{ color: '#64748b' }}>
            Real-Time Multi-Person Facial Expression Recognition
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium"
            style={{
              background:
                backendStatus === 'Online'
                  ? 'rgba(34,197,94,0.1)'
                  : 'rgba(239,68,68,0.1)',
              border:
                backendStatus === 'Online'
                  ? '1px solid rgba(34,197,94,0.2)'
                  : '1px solid rgba(239,68,68,0.2)',
              color: backendStatus === 'Online' ? '#4ade80' : '#f87171',
            }}
          >
            <span
              className={`w-2 h-2 rounded-full ${backendStatus === 'Online' ? 'bg-green-400 animate-pulse' : 'bg-red-400'
                }`}
            />
            Backend: {backendStatus}
          </div>
          <button
            onClick={() => onNavigate('live')}
            className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all"
            style={{
              background: 'linear-gradient(135deg,#00d4ff,#0098cc)',
              color: '#080c14',
            }}
          >
            <svg
              width="14"
              height="14"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M15 10l4.553-2.069A1 1 0 0121 8.845v6.31a1 1 0 01-1.447.894L15 14" />
              <rect x="3" y="6" width="12" height="12" rx="2" />
            </svg>
            Start Live Detection
          </button>
        </div>
      </div>

      {error && (
        <div
          className="p-3 rounded-lg text-xs flex items-center gap-2"
          style={{ background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.2)', color: '#f87171' }}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10" /><line x1="12" y1="8" x2="12" y2="12" /><line x1="12" y1="16" x2="12.01" y2="16" /></svg>
          Backend communication error: {error}
        </div>
      )}

      {/* Stat cards */}
      <div className="grid grid-cols-4 gap-4">
        <StatCard
          icon={
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#00d4ff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M17 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2" />
              <circle cx="9" cy="7" r="4" />
              <path d="M23 21v-2a4 4 0 00-3-3.87" />
              <path d="M16 3.13a4 4 0 010 7.75" />
            </svg>
          }
          label="Total Sessions"
          value={totalSessions}
          sub="Recorded in SQLite DB"
        />
        <StatCard
          icon={
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#a855f7" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="8" r="5" />
              <path d="M3 21a9 9 0 0118 0" />
            </svg>
          }
          label="Total Faces Analyzed"
          value={totalFaces}
          sub="Across all sessions"
        />
        <StatCard
          icon={
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#22c55e" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
            </svg>
          }
          label="Average Confidence"
          value={`${avgConf}%`}
          sub="Model accuracy"
        />
        <StatCard
          icon={
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#f97316" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
            </svg>
          }
          label="Most Detected"
          value={mostDetected}
          sub={`${EMOTION_ICONS[mostDetected as Emotion] || '😐'} dominant expression`}
        />
      </div>

      {/* Empty State vs Real Charts & Sessions */}
      {loading ? (
        <div className="rounded-xl p-8 flex items-center justify-center" style={{ background: '#0d1424', border: '1px solid rgba(0,212,255,0.1)' }}>
          <div className="text-xs text-slate-400 animate-pulse">Loading backend analytics data...</div>
        </div>
      ) : totalSessions === 0 ? (
        <div
          className="rounded-xl p-8 flex flex-col items-center justify-center text-center gap-3"
          style={{ background: '#0d1424', border: '1px dashed rgba(0,212,255,0.2)' }}
        >
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="rgba(0,212,255,0.4)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M15 10l4.553-2.069A1 1 0 0121 8.845v6.31a1 1 0 01-1.447.894L15 14" />
            <rect x="3" y="6" width="12" height="12" rx="2" />
          </svg>
          <h3 className="text-sm font-bold text-slate-200">No Session Data Found</h3>
          <p className="text-xs text-slate-400 max-w-sm">
            There are currently no recorded detection sessions in the SQLite backend database. Click below to start live multi-person emotion recognition.
          </p>
          <button
            onClick={() => onNavigate('live')}
            className="mt-2 px-4 py-2 rounded-lg text-xs font-medium"
            style={{ background: '#00d4ff', color: '#080c14' }}
          >
            Launch Live Detection
          </button>
        </div>
      ) : (
        <>
          {/* Charts row */}
          <div className="grid grid-cols-5 gap-4">
            {/* Donut */}
            <div
              className="col-span-2 rounded-xl p-4"
              style={{ background: '#0d1424', border: '1px solid rgba(0,212,255,0.1)' }}
            >
              <div className="text-sm font-semibold mb-4" style={{ color: '#94a3b8' }}>
                Expression Distribution
              </div>
              <div className="flex items-center gap-4">
                <ResponsiveContainer width={140} height={140}>
                  <PieChart>
                    <Pie
                      data={donutData}
                      cx="50%"
                      cy="50%"
                      innerRadius={42}
                      outerRadius={65}
                      paddingAngle={2}
                      dataKey="value"
                    >
                      {donutData.map((entry) => (
                        <Cell key={entry.name} fill={EMOTION_COLORS[entry.name as Emotion]} />
                      ))}
                    </Pie>
                    <Tooltip content={<CustomTooltip />} />
                  </PieChart>
                </ResponsiveContainer>
                <div className="flex flex-col gap-1.5 flex-1">
                  {EMOTIONS.map((e) => (
                    <div key={e} className="flex items-center gap-2 text-xs">
                      <span
                        className="w-2 h-2 rounded-full shrink-0"
                        style={{ background: EMOTION_COLORS[e] }}
                      />
                      <span style={{ color: '#94a3b8' }}>{e}</span>
                      <span className="ml-auto font-mono" style={{ color: '#64748b' }}>
                        {combinedEmotions[e]}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Bar chart */}
            <div
              className="col-span-3 rounded-xl p-4"
              style={{ background: '#0d1424', border: '1px solid rgba(0,212,255,0.1)' }}
            >
              <div className="text-sm font-semibold mb-4" style={{ color: '#94a3b8' }}>
                Expression Frequency
              </div>
              <ResponsiveContainer width="100%" height={130}>
                <BarChart data={barData} barSize={28}>
                  <XAxis
                    dataKey="name"
                    tick={{ fill: '#475569', fontSize: 11 }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <YAxis tick={{ fill: '#334155', fontSize: 10 }} axisLine={false} tickLine={false} />
                  <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(0,212,255,0.04)' }} />
                  <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                    {barData.map((entry, i) => (
                      <Cell key={i} fill={entry.color} fillOpacity={0.85} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Recent sessions */}
          <div
            className="rounded-xl p-4"
            style={{ background: '#0d1424', border: '1px solid rgba(0,212,255,0.1)' }}
          >
            <div className="flex items-center justify-between mb-4">
              <div className="text-sm font-semibold" style={{ color: '#94a3b8' }}>
                Recent Sessions
              </div>
              <button
                onClick={() => onNavigate('history')}
                className="text-xs transition-colors"
                style={{ color: '#00d4ff' }}
              >
                View all →
              </button>
            </div>
            <div className="space-y-2">
              {recentSessions.map((s) => {
                const sourceType = s.source_type || 'webcam'
                const icon = sourceType === 'video' ? '🎥' : sourceType === 'image' ? '📷' : '📹'
                return (
                  <div
                    key={s.session_id}
                    className="flex items-center gap-3 px-3.5 py-2.5 rounded-xl border border-slate-800/80 bg-slate-900/60 transition-all hover:border-cyan-500/30"
                  >
                    <span className="text-base">{icon}</span>
                    <div className="flex flex-col">
                      <div className="text-xs font-semibold text-slate-200">
                        {s.session_name || s.session_id}
                      </div>
                      <div className="text-[11px] text-slate-400 font-mono">
                        {sourceType.toUpperCase()} {s.date ? `· ${formatLocalDateTime(s.date)}` : ''}
                      </div>
                    </div>
                    {sourceType !== 'image' && (
                      <div className="flex items-center gap-1 text-xs text-slate-400 font-mono ml-auto md:ml-4">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <circle cx="12" cy="12" r="10" />
                          <polyline points="12 6 12 12 16 14" />
                        </svg>
                        {Math.round(s.duration_seconds)}s
                      </div>
                    )}
                    <div className="text-xs text-slate-300 ml-auto md:ml-4 font-mono">
                      {s.people_count} {s.people_count === 1 ? 'face' : 'faces'}
                    </div>
                    <div className="flex items-center gap-2">
                      <span
                        className="text-xs px-2.5 py-0.5 rounded-full font-medium"
                        style={{
                          background: `${EMOTION_COLORS[s.dominant_expression as Emotion] || '#00d4ff'}22`,
                          color: EMOTION_COLORS[s.dominant_expression as Emotion] || '#00d4ff',
                          border: `1px solid ${EMOTION_COLORS[s.dominant_expression as Emotion] || '#00d4ff'}44`,
                        }}
                      >
                        {EMOTION_ICONS[s.dominant_expression as Emotion] || '😐'} {s.dominant_expression}
                      </span>
                      <span className="text-xs font-mono text-slate-400">
                        {s.average_confidence}%
                      </span>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
