import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'
import { SESSIONS, EMOTIONS, EMOTION_COLORS, EMOTION_ICONS } from '../data'
import type { Screen } from '../types'

interface Props {
  onNavigate: (s: Screen) => void
}

const totalFaces = SESSIONS.reduce((a, s) => a + s.people, 0)
const avgConf = (SESSIONS.reduce((a, s) => a + s.avgConfidence, 0) / SESSIONS.length).toFixed(1)
const dominantCounts: Record<string, number> = {}
SESSIONS.forEach(s => { dominantCounts[s.dominant] = (dominantCounts[s.dominant] || 0) + 1 })
const mostDetected = Object.entries(dominantCounts).sort((a, b) => b[1] - a[1])[0][0]

const combined: Record<string, number> = {}
SESSIONS.forEach(s => {
  EMOTIONS.forEach(e => {
    combined[e] = (combined[e] || 0) + s.emotionDist[e]
  })
})
const donutData = EMOTIONS.map(e => ({ name: e, value: combined[e] }))
const barData = EMOTIONS.map(e => ({ name: e.slice(0, 3), value: combined[e], color: EMOTION_COLORS[e] }))

function StatCard({ icon, label, value, sub }: { icon: React.ReactNode; label: string; value: string | number; sub?: string }) {
  return (
    <div className="rounded-xl p-4 flex flex-col gap-2" style={{ background: '#0d1424', border: '1px solid rgba(0,212,255,0.1)' }}>
      <div className="flex items-center gap-2">
        <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: 'rgba(0,212,255,0.08)' }}>
          {icon}
        </div>
        <span className="text-xs" style={{ color: '#64748b' }}>{label}</span>
      </div>
      <div className="text-2xl font-bold tracking-tight" style={{ color: '#e2e8f0', fontVariantNumeric: 'tabular-nums' }}>{value}</div>
      {sub && <div className="text-xs" style={{ color: '#475569' }}>{sub}</div>}
    </div>
  )
}

const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload?.length) {
    return (
      <div className="px-3 py-2 rounded-lg text-xs" style={{ background: '#131e30', border: '1px solid rgba(0,212,255,0.15)', color: '#e2e8f0' }}>
        {payload[0].name}: <strong>{payload[0].value}</strong>
      </div>
    )
  }
  return null
}

export default function Dashboard({ onNavigate }: Props) {
  const recent = SESSIONS.slice().reverse().slice(0, 3)

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-xl font-bold tracking-tight" style={{ color: '#e2e8f0' }}>
            Computer Vision — Emotion Recognition
          </h1>
          <p className="text-sm mt-0.5" style={{ color: '#64748b' }}>
            Real-Time Human Emotion Recognition Using Facial Expressions
          </p>
        </div>
        <button
          onClick={() => onNavigate('live')}
          className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all"
          style={{ background: 'linear-gradient(135deg,#00d4ff,#0098cc)', color: '#080c14' }}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M15 10l4.553-2.069A1 1 0 0121 8.845v6.31a1 1 0 01-1.447.894L15 14" />
            <rect x="3" y="6" width="12" height="12" rx="2" />
          </svg>
          Start Live Detection
        </button>
      </div>

      {/* Stat cards */}
      <div className="grid grid-cols-4 gap-4">
        <StatCard
          icon={<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#00d4ff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M17 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 00-3-3.87"/><path d="M16 3.13a4 4 0 010 7.75"/></svg>}
          label="Total Sessions"
          value={SESSIONS.length}
          sub="All time"
        />
        <StatCard
          icon={<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#a855f7" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="8" r="5"/><path d="M3 21a9 9 0 0118 0"/></svg>}
          label="Total Faces Analyzed"
          value={totalFaces}
          sub="Across all sessions"
        />
        <StatCard
          icon={<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#22c55e" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>}
          label="Average Confidence"
          value={`${avgConf}%`}
          sub="Model accuracy"
        />
        <StatCard
          icon={<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#f97316" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>}
          label="Most Detected"
          value={mostDetected}
          sub={`${EMOTION_ICONS[mostDetected as keyof typeof EMOTION_ICONS]} dominant expression`}
        />
      </div>

      {/* Charts row */}
      <div className="grid grid-cols-5 gap-4">
        {/* Donut */}
        <div className="col-span-2 rounded-xl p-4" style={{ background: '#0d1424', border: '1px solid rgba(0,212,255,0.1)' }}>
          <div className="text-sm font-semibold mb-4" style={{ color: '#94a3b8' }}>Expression Distribution</div>
          <div className="flex items-center gap-4">
            <ResponsiveContainer width={140} height={140}>
              <PieChart>
                <Pie data={donutData} cx="50%" cy="50%" innerRadius={42} outerRadius={65} paddingAngle={2} dataKey="value">
                  {donutData.map((entry) => (
                    <Cell key={entry.name} fill={EMOTION_COLORS[entry.name as keyof typeof EMOTION_COLORS]} />
                  ))}
                </Pie>
                <Tooltip content={<CustomTooltip />} />
              </PieChart>
            </ResponsiveContainer>
            <div className="flex flex-col gap-1.5">
              {EMOTIONS.map(e => (
                <div key={e} className="flex items-center gap-2 text-xs">
                  <span className="w-2 h-2 rounded-full shrink-0" style={{ background: EMOTION_COLORS[e] }} />
                  <span style={{ color: '#94a3b8' }}>{e}</span>
                  <span className="ml-auto font-mono" style={{ color: '#64748b' }}>{combined[e]}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Bar chart */}
        <div className="col-span-3 rounded-xl p-4" style={{ background: '#0d1424', border: '1px solid rgba(0,212,255,0.1)' }}>
          <div className="text-sm font-semibold mb-4" style={{ color: '#94a3b8' }}>Expression Frequency</div>
          <ResponsiveContainer width="100%" height={130}>
            <BarChart data={barData} barSize={28}>
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

      {/* Recent sessions */}
      <div className="rounded-xl p-4" style={{ background: '#0d1424', border: '1px solid rgba(0,212,255,0.1)' }}>
        <div className="flex items-center justify-between mb-4">
          <div className="text-sm font-semibold" style={{ color: '#94a3b8' }}>Recent Sessions</div>
          <button onClick={() => onNavigate('history')} className="text-xs transition-colors" style={{ color: '#00d4ff' }}>View all →</button>
        </div>
        <div className="space-y-2">
          {recent.map(s => (
            <div key={s.id} className="flex items-center gap-4 px-3 py-2.5 rounded-lg" style={{ background: '#0a1120' }}>
              <div className="w-2 h-2 rounded-full" style={{ background: EMOTION_COLORS[s.dominant] }} />
              <div className="text-xs font-mono" style={{ color: '#64748b', width: 100 }}>{s.id}</div>
              <div className="text-xs" style={{ color: '#94a3b8' }}>{s.date}</div>
              <div className="flex items-center gap-1.5 text-xs" style={{ color: '#475569' }}>
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
                {s.duration}
              </div>
              <div className="text-xs" style={{ color: '#94a3b8' }}>{s.people} people</div>
              <div className="ml-auto flex items-center gap-1.5">
                <span className="text-xs px-2 py-0.5 rounded" style={{ background: `${EMOTION_COLORS[s.dominant]}18`, color: EMOTION_COLORS[s.dominant] }}>{s.dominant}</span>
                <span className="text-xs font-mono" style={{ color: '#64748b' }}>{s.avgConfidence.toFixed(1)}%</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Recent activity */}
      <div className="rounded-xl p-4" style={{ background: '#0d1424', border: '1px solid rgba(0,212,255,0.1)' }}>
        <div className="text-sm font-semibold mb-3" style={{ color: '#94a3b8' }}>Recent Activity</div>
        <div className="space-y-2">
          {[
            { time: '2 min ago', msg: 'Session SES-2024-005 completed — 9 persons detected', color: '#22c55e' },
            { time: '1 hr ago', msg: 'Model confidence exceeded 90% threshold', color: '#00d4ff' },
            { time: '3 hrs ago', msg: 'Session SES-2024-004 flagged high Angry detection (32%)', color: '#ef4444' },
            { time: '5 hrs ago', msg: 'System calibration complete — FPS stable at 30', color: '#a855f7' },
          ].map((a, i) => (
            <div key={i} className="flex items-start gap-3 text-xs">
              <span className="w-1.5 h-1.5 rounded-full mt-1 shrink-0" style={{ background: a.color }} />
              <span style={{ color: '#94a3b8' }}>{a.msg}</span>
              <span className="ml-auto shrink-0" style={{ color: '#334155' }}>{a.time}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
