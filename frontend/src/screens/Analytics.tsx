import { useState, useEffect } from 'react'
import {
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'
import { EMOTIONS, EMOTION_COLORS, EMOTION_ICONS } from '../data'
import type { Emotion, SessionSummary, SessionAnalyticsData, PersonAnalyticsData } from '../types'
import { apiService } from '../services/api'

interface Props {
  initialSessionId?: string | null
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
        {payload[0].name || payload[0].dataKey}: <strong>{payload[0].value}</strong>
      </div>
    )
  }
  return null
}

export default function Analytics({ initialSessionId }: Props) {
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [selectedSessionId, setSelectedSessionId] = useState<string>('')
  const [sessionAnalytics, setSessionAnalytics] = useState<SessionAnalyticsData | null>(null)
  const [selectedPersonId, setSelectedPersonId] = useState<number | null>(null)
  const [personAnalytics, setPersonAnalytics] = useState<PersonAnalyticsData | null>(null)
  const [loading, setLoading] = useState(true)
  const [loadingPerson, setLoadingPerson] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Load list of sessions on mount
  useEffect(() => {
    let isMounted = true
    async function loadSessions() {
      try {
        setLoading(true)
        setError(null)
        const res = await apiService.getSessions(1, 50)
        if (isMounted) {
          setSessions(res.sessions || [])
          if (res.sessions && res.sessions.length > 0) {
            const defaultId = initialSessionId || res.sessions[0].session_id
            setSelectedSessionId(defaultId)
          }
        }
      } catch (err: any) {
        if (isMounted) {
          setError(err.message || 'Failed to fetch session list')
        }
      } finally {
        if (isMounted) setLoading(false)
      }
    }
    loadSessions()
    return () => {
      isMounted = false
    }
  }, [initialSessionId])

  // Load session analytics when selectedSessionId changes
  useEffect(() => {
    if (!selectedSessionId) return
    let isMounted = true
    async function loadAnalytics() {
      try {
        setLoading(true)
        setError(null)
        const data = await apiService.getSessionAnalytics(selectedSessionId)
        if (isMounted) {
          setSessionAnalytics(data)
          if (data.persons && data.persons.length > 0) {
            setSelectedPersonId(data.persons[0])
          } else {
            setSelectedPersonId(null)
            setPersonAnalytics(null)
          }
        }
      } catch (err: any) {
        if (isMounted) {
          setError(err.message || 'Failed to fetch session analytics')
        }
      } finally {
        if (isMounted) setLoading(false)
      }
    }
    loadAnalytics()
    return () => {
      isMounted = false
    }
  }, [selectedSessionId])

  // Load person analytics when selectedPersonId changes
  useEffect(() => {
    if (!selectedSessionId || selectedPersonId === null) return
    let isMounted = true
    async function loadPersonData() {
      try {
        setLoadingPerson(true)
        const data = await apiService.getPersonAnalytics(selectedSessionId, selectedPersonId)
        if (isMounted) {
          setPersonAnalytics(data)
        }
      } catch (err) {
        console.warn('Error fetching person analytics:', err)
      } finally {
        if (isMounted) setLoadingPerson(false)
      }
    }
    loadPersonData()
    return () => {
      isMounted = false
    }
  }, [selectedSessionId, selectedPersonId])

  if (loading && !sessionAnalytics) {
    return (
      <div className="flex-1 flex items-center justify-center p-8 text-xs text-slate-400 animate-pulse">
        Loading analytics from FastAPI backend...
      </div>
    )
  }

  if (sessions.length === 0) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center p-8 gap-3">
        <svg
          width="40"
          height="40"
          viewBox="0 0 24 24"
          fill="none"
          stroke="rgba(0,212,255,0.4)"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
        </svg>
        <h3 className="text-sm font-bold text-slate-200">No Analytics Available</h3>
        <p className="text-xs text-slate-400 max-w-sm text-center">
          No detection sessions found in SQLite database. Start a live session to view detailed emotion analytics.
        </p>
      </div>
    )
  }

  const dist = sessionAnalytics?.expression_distribution || {}
  const donutData = EMOTIONS.map((e) => ({
    name: e,
    value: dist[e] || 0,
  })).filter((d) => d.value > 0)

  const freq = sessionAnalytics?.expression_frequency || {}
  const barData = EMOTIONS.map((e) => ({
    name: e.slice(0, 3),
    value: freq[e] || 0,
    color: EMOTION_COLORS[e as Emotion],
  }))

  const timelineData = sessionAnalytics?.expression_timeline || []

  // Person breakdown metrics
  const personDist = personAnalytics?.expression_distribution || {}
  const personDonutData = EMOTIONS.map((e) => ({
    name: e,
    value: personDist[e] || 0,
  })).filter((d) => d.value > 0)

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-6 animate-fade-in">
      {/* Top Header & Session Selector */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold tracking-tight" style={{ color: '#e2e8f0' }}>
            Session Analytics
          </h1>
          <p className="text-sm mt-0.5" style={{ color: '#64748b' }}>
            Comprehensive Multi-Person Facial Expression Insights
          </p>
        </div>

        {/* Session Selector */}
        <div className="flex items-center gap-2">
          <label className="text-xs text-slate-400 font-medium">Select Session:</label>
          <select
            value={selectedSessionId}
            onChange={(e) => setSelectedSessionId(e.target.value)}
            className="text-xs px-3 py-1.5 rounded-lg font-mono font-medium focus:outline-none"
            style={{
              background: '#0d1424',
              color: '#00d4ff',
              border: '1px solid rgba(0,212,255,0.2)',
            }}
          >
            {sessions.map((s) => (
              <option key={s.session_id} value={s.session_id}>
                {s.session_name || s.session_id} ({s.date})
              </option>
            ))}
          </select>
        </div>
      </div>

      {error && (
        <div
          className="p-3 rounded-lg text-xs"
          style={{ background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.2)', color: '#f87171' }}
        >
          Analytics Error: {error}
        </div>
      )}

      {/* KPI Cards */}
      <div className="grid grid-cols-5 gap-4">
        {[
          {
            label: 'Total People Detected',
            value: sessionAnalytics?.total_people_detected || 0,
            sub: 'Unique Person IDs',
            color: '#00d4ff',
          },
          {
            label: 'Total Predictions',
            value: sessionAnalytics?.total_predictions || 0,
            sub: 'Processed frames',
            color: '#a855f7',
          },
          {
            label: 'Dominant Expression',
            value: sessionAnalytics?.dominant_expression || 'None',
            sub: `${EMOTION_ICONS[sessionAnalytics?.dominant_expression as Emotion] || '😐'} overall`,
            color: EMOTION_COLORS[sessionAnalytics?.dominant_expression as Emotion] || '#22c55e',
          },
          {
            label: 'Average Confidence',
            value: `${sessionAnalytics?.average_confidence || 0}%`,
            sub: 'Model certainty',
            color: '#22c55e',
          },
          {
            label: 'Session Duration',
            value: `${Math.round(sessionAnalytics?.session_duration_seconds || 0)}s`,
            sub: `FPS: ${sessionAnalytics?.fps_stats?.average || 30.0}`,
            color: '#f97316',
          },
        ].map((kpi, idx) => (
          <div
            key={idx}
            className="rounded-xl p-4 flex flex-col gap-1.5"
            style={{ background: '#0d1424', border: '1px solid rgba(0,212,255,0.1)' }}
          >
            <span className="text-xs" style={{ color: '#64748b' }}>
              {kpi.label}
            </span>
            <span
              className="text-xl font-bold font-mono tracking-tight"
              style={{ color: kpi.color }}
            >
              {kpi.value}
            </span>
            <span className="text-xs" style={{ color: '#475569' }}>
              {kpi.sub}
            </span>
          </div>
        ))}
      </div>

      {/* Session Charts Row */}
      <div className="grid grid-cols-5 gap-4">
        {/* Expression Distribution (Donut) */}
        <div
          className="col-span-2 rounded-xl p-4"
          style={{ background: '#0d1424', border: '1px solid rgba(0,212,255,0.1)' }}
        >
          <div className="text-sm font-semibold mb-4" style={{ color: '#94a3b8' }}>
            Expression Distribution (%)
          </div>
          <div className="flex items-center gap-4">
            <ResponsiveContainer width={130} height={130}>
              <PieChart>
                <Pie
                  data={donutData}
                  cx="50%"
                  cy="50%"
                  innerRadius={38}
                  outerRadius={60}
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
                    {dist[e] !== undefined ? `${dist[e]}%` : '0%'}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Expression Frequency (Bar) */}
        <div
          className="col-span-3 rounded-xl p-4"
          style={{ background: '#0d1424', border: '1px solid rgba(0,212,255,0.1)' }}
        >
          <div className="text-sm font-semibold mb-4" style={{ color: '#94a3b8' }}>
            Expression Count Frequency
          </div>
          <ResponsiveContainer width="100%" height={130}>
            <BarChart data={barData} barSize={26}>
              <XAxis dataKey="name" tick={{ fill: '#475569', fontSize: 11 }} axisLine={false} tickLine={false} />
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

      {/* Timeline Area Chart */}
      <div
        className="rounded-xl p-4"
        style={{ background: '#0d1424', border: '1px solid rgba(0,212,255,0.1)' }}
      >
        <div className="text-sm font-semibold mb-4" style={{ color: '#94a3b8' }}>
          Session Expression & People Timeline
        </div>
        <ResponsiveContainer width="100%" height={150}>
          <AreaChart data={timelineData}>
            <defs>
              <linearGradient id="colorCount" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#00d4ff" stopOpacity={0.4} />
                <stop offset="95%" stopColor="#00d4ff" stopOpacity={0.0} />
              </linearGradient>
            </defs>
            <XAxis dataKey="time" tick={{ fill: '#475569', fontSize: 10 }} axisLine={false} tickLine={false} />
            <YAxis tick={{ fill: '#334155', fontSize: 10 }} axisLine={false} tickLine={false} />
            <Tooltip content={<CustomTooltip />} />
            <Area type="monotone" dataKey="count" stroke="#00d4ff" strokeWidth={2} fillOpacity={1} fill="url(#colorCount)" />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Section: Individual Person Analytics */}
      <div
        className="rounded-xl p-5 space-y-4"
        style={{ background: '#09101d', border: '1px solid rgba(168,85,247,0.2)' }}
      >
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-sm font-bold tracking-wide" style={{ color: '#c084fc' }}>
              Individual Person Analytics
            </h2>
            <p className="text-xs mt-0.5" style={{ color: '#64748b' }}>
              Select a tracked Person ID to analyze individual facial expression behavior
            </p>
          </div>

          {/* Person ID Tabs / Select */}
          {sessionAnalytics?.persons && sessionAnalytics.persons.length > 0 ? (
            <div className="flex items-center gap-1.5">
              {sessionAnalytics.persons.map((pid) => (
                <button
                  key={pid}
                  onClick={() => setSelectedPersonId(pid)}
                  className="px-3 py-1 rounded text-xs font-mono font-medium transition-all"
                  style={{
                    background:
                      selectedPersonId === pid ? '#a855f7' : 'rgba(168,85,247,0.1)',
                    color: selectedPersonId === pid ? '#080c14' : '#c084fc',
                    border: '1px solid rgba(168,85,247,0.3)',
                  }}
                >
                  Person {pid}
                </button>
              ))}
            </div>
          ) : (
            <div className="text-xs text-slate-500 font-mono">No person IDs recorded</div>
          )}
        </div>

        {loadingPerson ? (
          <div className="p-4 text-xs text-purple-400 animate-pulse">
            Fetching Person {selectedPersonId} analytics...
          </div>
        ) : personAnalytics ? (
          <div className="grid grid-cols-4 gap-4">
            <div
              className="rounded-lg p-3 flex flex-col gap-1"
              style={{ background: '#0d1424', border: '1px solid rgba(0,212,255,0.1)' }}
            >
              <span className="text-xs text-slate-400">Target Person</span>
              <span className="text-lg font-bold font-mono text-purple-400">
                Person #{personAnalytics.person_id}
              </span>
            </div>

            <div
              className="rounded-lg p-3 flex flex-col gap-1"
              style={{ background: '#0d1424', border: '1px solid rgba(0,212,255,0.1)' }}
            >
              <span className="text-xs text-slate-400">Dominant Expression</span>
              <span className="text-lg font-bold text-green-400">
                {EMOTION_ICONS[personAnalytics.dominant_expression as Emotion] || '😐'}{' '}
                {personAnalytics.dominant_expression}
              </span>
            </div>

            <div
              className="rounded-lg p-3 flex flex-col gap-1"
              style={{ background: '#0d1424', border: '1px solid rgba(0,212,255,0.1)' }}
            >
              <span className="text-xs text-slate-400">Average Confidence</span>
              <span className="text-lg font-bold font-mono text-cyan-400">
                {personAnalytics.average_confidence}%
              </span>
            </div>

            <div
              className="rounded-lg p-3 flex flex-col gap-1"
              style={{ background: '#0d1424', border: '1px solid rgba(0,212,255,0.1)' }}
            >
              <span className="text-xs text-slate-400">Total Samples</span>
              <span className="text-lg font-bold font-mono text-orange-400">
                {personAnalytics.expression_timeline?.length || 0} frames
              </span>
            </div>

            {/* Individual Donut & Timeline */}
            <div
              className="col-span-2 rounded-lg p-3"
              style={{ background: '#0d1424', border: '1px solid rgba(0,212,255,0.1)' }}
            >
              <span className="text-xs font-semibold text-slate-400 mb-2 block">
                Person {personAnalytics.person_id} Distribution (%)
              </span>
              <div className="flex items-center gap-3">
                <ResponsiveContainer width={100} height={100}>
                  <PieChart>
                    <Pie
                      data={personDonutData}
                      cx="50%"
                      cy="50%"
                      innerRadius={28}
                      outerRadius={45}
                      paddingAngle={2}
                      dataKey="value"
                    >
                      {personDonutData.map((entry) => (
                        <Cell key={entry.name} fill={EMOTION_COLORS[entry.name as Emotion]} />
                      ))}
                    </Pie>
                  </PieChart>
                </ResponsiveContainer>
                <div className="flex flex-col gap-1 flex-1 text-xs">
                  {EMOTIONS.map((e) => (
                    <div key={e} className="flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full" style={{ background: EMOTION_COLORS[e] }} />
                      <span className="text-slate-400">{e}</span>
                      <span className="ml-auto font-mono text-slate-500">
                        {personDist[e] !== undefined ? `${personDist[e]}%` : '0%'}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div
              className="col-span-2 rounded-lg p-3"
              style={{ background: '#0d1424', border: '1px solid rgba(0,212,255,0.1)' }}
            >
              <span className="text-xs font-semibold text-slate-400 mb-2 block">
                Expression Sequence History
              </span>
              <div className="flex flex-wrap gap-1.5 max-h-24 overflow-y-auto">
                {personAnalytics.expression_timeline?.map((emo, i) => (
                  <span
                    key={i}
                    className="text-xs px-2 py-0.5 rounded font-mono"
                    style={{
                      background: `${EMOTION_COLORS[emo as Emotion] || '#00d4ff'}20`,
                      color: EMOTION_COLORS[emo as Emotion] || '#00d4ff',
                      border: `1px solid ${EMOTION_COLORS[emo as Emotion] || '#00d4ff'}40`,
                    }}
                  >
                    #{i + 1}: {emo}
                  </span>
                ))}
              </div>
            </div>
          </div>
        ) : null}
      </div>
    </div>
  )
}
