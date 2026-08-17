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
import type { Emotion, SessionSummary, SessionAnalyticsData, PersonAnalyticsData, PersonDetailCard } from '../types'
import { apiService } from '../services/api'
import { SessionPdfReport } from '../components/SessionPdfReport'
import { Printer, User, Download } from 'lucide-react'

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
  const [showPdf, setShowPdf] = useState(false)

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
    const timer = setTimeout(() => {
      if (isMounted) setLoading(false)
    }, 3000)
    return () => {
      isMounted = false
      clearTimeout(timer)
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

  if (showPdf && sessionAnalytics) {
    return (
      <SessionPdfReport
        analytics={sessionAnalytics}
        sessionName={sessionAnalytics.session_name || 'Live Camera Session'}
        onClose={() => setShowPdf(false)}
      />
    )
  }

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
          No detection sessions found in database. Start a live session to view detailed emotion analytics.
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
          <h1 className="text-xl font-bold tracking-tight text-slate-100">
            Session & Person Identification Analytics
          </h1>
          <p className="text-sm mt-0.5 text-slate-400">
            Comprehensive Multi-Person Facial Expression Insights & PDF Report Generation
          </p>
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-3">
          {sessionAnalytics && (
            <button
              onClick={() => setShowPdf(true)}
              className="px-3.5 py-1.5 rounded-lg text-xs font-semibold bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white flex items-center gap-1.5 shadow-lg shadow-cyan-500/20 transition"
            >
              <Printer size={14} />
              <span>Export PDF Report</span>
            </button>
          )}

          {/* Session Selector */}
          <div className="flex items-center gap-2">
            <label className="text-xs text-slate-400 font-medium">Select Session:</label>
            <select
              value={selectedSessionId}
              onChange={(e) => setSelectedSessionId(e.target.value)}
              className="text-xs px-3 py-1.5 rounded-lg font-mono font-medium bg-slate-900 text-cyan-400 border border-cyan-500/20 focus:outline-none"
            >
              {sessions.map((s) => (
                <option key={s.session_id} value={s.session_id}>
                  {s.session_name || s.session_id} ({s.date})
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {error && (
        <div className="p-3 rounded-lg text-xs bg-red-500/10 border border-red-500/20 text-red-400">
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
            className="rounded-xl p-4 flex flex-col gap-1.5 bg-slate-900 border border-cyan-500/10"
          >
            <span className="text-xs text-slate-400">
              {kpi.label}
            </span>
            <span
              className="text-xl font-bold font-mono tracking-tight"
              style={{ color: kpi.color }}
            >
              {kpi.value}
            </span>
            <span className="text-xs text-slate-500">
              {kpi.sub}
            </span>
          </div>
        ))}
      </div>

      {/* Person Identification Gallery Section */}
      {sessionAnalytics?.persons_details && sessionAnalytics.persons_details.length > 0 && (
        <div className="p-5 rounded-xl bg-slate-900 border border-cyan-500/20 space-y-4">
          <h2 className="text-sm font-bold text-cyan-400 flex items-center gap-2">
            <User size={18} />
            <span>Person Identification & Facial Photo Cards</span>
          </h2>

          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
            {sessionAnalytics.persons_details.map((person: PersonDetailCard) => (
              <div
                key={person.person_id}
                onClick={() => setSelectedPersonId(person.person_id)}
                className={`p-3.5 rounded-xl border flex items-center space-x-3.5 cursor-pointer transition ${
                  selectedPersonId === person.person_id
                    ? 'bg-purple-950/40 border-purple-500 shadow-lg shadow-purple-500/10'
                    : 'bg-slate-950/60 border-slate-800 hover:border-cyan-500/30'
                }`}
              >
                {/* Face Thumbnail */}
                <div className="w-14 h-14 bg-slate-800 rounded-lg overflow-hidden border border-cyan-500/30 flex items-center justify-center flex-shrink-0">
                  {person.thumbnail_b64 ? (
                    <img
                      src={person.thumbnail_b64}
                      alt={`Person ${person.person_id} face photo`}
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <User size={24} className="text-slate-500" />
                  )}
                </div>

                {/* Person Details */}
                <div className="flex-1 min-w-0 text-xs">
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-bold text-cyan-300">
                      Person #{person.person_id}
                    </span>
                    <span
                      className="px-1.5 py-0.5 rounded text-[10px] font-semibold text-white"
                      style={{ backgroundColor: EMOTION_COLORS[person.dominant_emotion as Emotion] || '#3b82f6' }}
                    >
                      {person.dominant_emotion}
                    </span>
                  </div>
                  <div className="text-slate-400">
                    Conf: <strong className="text-slate-200">{person.average_confidence}%</strong>
                  </div>
                  <div className="text-slate-400">
                    Frames: <strong className="text-slate-200">{person.total_detections}</strong>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Session Charts Row */}
      <div className="grid grid-cols-5 gap-4">
        {/* Expression Distribution (Donut) */}
        <div className="col-span-2 rounded-xl p-4 bg-slate-900 border border-cyan-500/10">
          <div className="text-sm font-semibold mb-4 text-slate-400">
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
                  <span className="text-slate-400">{e}</span>
                  <span className="ml-auto font-mono text-slate-500">
                    {dist[e] !== undefined ? `${dist[e]}%` : '0%'}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Expression Frequency (Bar) */}
        <div className="col-span-3 rounded-xl p-4 bg-slate-900 border border-cyan-500/10">
          <div className="text-sm font-semibold mb-4 text-slate-400">
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
      <div className="rounded-xl p-4 bg-slate-900 border border-cyan-500/10">
        <div className="text-sm font-semibold mb-4 text-slate-400">
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
      <div className="rounded-xl p-5 space-y-4 bg-slate-950 border border-purple-500/20">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-sm font-bold tracking-wide text-purple-400">
              Individual Person Analytics
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Select a tracked Person ID to analyze individual facial expression behavior
            </p>
          </div>

          {/* Person ID Tabs */}
          {sessionAnalytics?.persons && sessionAnalytics.persons.length > 0 ? (
            <div className="flex items-center gap-1.5">
              {sessionAnalytics.persons.map((pid) => (
                <button
                  key={pid}
                  onClick={() => setSelectedPersonId(pid)}
                  className={`px-3 py-1 rounded text-xs font-mono font-medium transition-all ${
                    selectedPersonId === pid
                      ? 'bg-purple-600 text-white border border-purple-400'
                      : 'bg-purple-950/40 text-purple-300 border border-purple-500/30'
                  }`}
                >
                  Person #{pid}
                </button>
              ))}
            </div>
          ) : (
            <div className="text-xs text-slate-500 font-mono">No person IDs recorded</div>
          )}
        </div>

        {loadingPerson ? (
          <div className="p-4 text-xs text-purple-400 animate-pulse">
            Fetching Person #{selectedPersonId} analytics...
          </div>
        ) : personAnalytics ? (
          <div className="grid grid-cols-4 gap-4">
            <div className="rounded-lg p-3 flex flex-col gap-1 bg-slate-900 border border-cyan-500/10">
              <span className="text-xs text-slate-400">Target Person</span>
              <span className="text-lg font-bold font-mono text-purple-400">
                Person #{personAnalytics.person_id}
              </span>
            </div>

            <div className="rounded-lg p-3 flex flex-col gap-1 bg-slate-900 border border-cyan-500/10">
              <span className="text-xs text-slate-400">Dominant Expression</span>
              <span className="text-lg font-bold text-green-400">
                {EMOTION_ICONS[personAnalytics.dominant_expression as Emotion] || '😐'}{' '}
                {personAnalytics.dominant_expression}
              </span>
            </div>

            <div className="rounded-lg p-3 flex flex-col gap-1 bg-slate-900 border border-cyan-500/10">
              <span className="text-xs text-slate-400">Average Confidence</span>
              <span className="text-lg font-bold font-mono text-cyan-400">
                {personAnalytics.average_confidence}%
              </span>
            </div>

            <div className="rounded-lg p-3 flex flex-col gap-1 bg-slate-900 border border-cyan-500/10">
              <span className="text-xs text-slate-400">Total Samples</span>
              <span className="text-lg font-bold font-mono text-orange-400">
                {personAnalytics.expression_timeline?.length || 0} frames
              </span>
            </div>

            {/* Individual Donut & Timeline */}
            <div className="col-span-2 rounded-lg p-3 bg-slate-900 border border-cyan-500/10">
              <span className="text-xs font-semibold text-slate-400 mb-2 block">
                Person #{personAnalytics.person_id} Distribution (%)
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

            <div className="col-span-2 rounded-lg p-3 bg-slate-900 border border-cyan-500/10">
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
