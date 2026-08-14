import { useState } from 'react'
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, AreaChart, Area, CartesianGrid, XAxis, YAxis } from 'recharts'
import { SESSIONS, EMOTIONS, EMOTION_COLORS, EMOTION_ICONS } from '../data'
import type { Session, Emotion } from '../types'

const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload?.length) {
    return (
      <div className="px-3 py-2 rounded-lg text-xs" style={{ background: '#131e30', border: '1px solid rgba(0,212,255,0.15)', color: '#e2e8f0' }}>
        {payload.map((p: any, i: number) => <div key={i}>{p.name}: <strong>{p.value}</strong></div>)}
      </div>
    )
  }
  return null
}

function DetailModal({ session, onClose }: { session: Session; onClose: () => void }) {
  const donutData = EMOTIONS.map(e => ({ name: e, value: session.emotionDist[e] }))

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center" style={{ background: 'rgba(4,8,16,0.85)' }} onClick={onClose}>
      <div
        className="w-[700px] max-h-[85vh] overflow-y-auto rounded-2xl p-6 space-y-5"
        style={{ background: '#0d1424', border: '1px solid rgba(0,212,255,0.18)' }}
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm font-bold" style={{ color: '#e2e8f0' }}>Session Details</div>
            <div className="text-xs font-mono mt-0.5" style={{ color: '#64748b' }}>{session.id}</div>
          </div>
          <button onClick={onClose} className="w-7 h-7 rounded-lg flex items-center justify-center transition-colors" style={{ background: 'rgba(0,212,255,0.08)', color: '#64748b' }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
          </button>
        </div>

        {/* Metadata grid */}
        <div className="grid grid-cols-3 gap-3">
          {[
            { label: 'Date', value: session.date },
            { label: 'Start Time', value: session.startTime },
            { label: 'Duration', value: session.duration },
            { label: 'People Analyzed', value: session.people },
            { label: 'Avg Confidence', value: `${session.avgConfidence}%` },
            { label: 'Dominant Expression', value: session.dominant, color: EMOTION_COLORS[session.dominant] },
          ].map(m => (
            <div key={m.label} className="px-3 py-2.5 rounded-lg" style={{ background: '#09101d' }}>
              <div className="text-xs mb-0.5" style={{ color: '#475569' }}>{m.label}</div>
              <div className="text-sm font-semibold" style={{ color: m.color || '#e2e8f0' }}>{m.value}</div>
            </div>
          ))}
        </div>

        {/* Charts */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <div className="text-xs font-semibold mb-2" style={{ color: '#64748b' }}>Expression Distribution</div>
            <div className="flex items-center gap-3">
              <ResponsiveContainer width={100} height={100}>
                <PieChart>
                  <Pie data={donutData} cx="50%" cy="50%" innerRadius={28} outerRadius={46} paddingAngle={2} dataKey="value">
                    {donutData.map(d => <Cell key={d.name} fill={EMOTION_COLORS[d.name as Emotion]} />)}
                  </Pie>
                  <Tooltip content={<CustomTooltip />} />
                </PieChart>
              </ResponsiveContainer>
              <div className="space-y-1">
                {EMOTIONS.map(e => (
                  <div key={e} className="flex items-center gap-1.5 text-xs">
                    <span className="w-1.5 h-1.5 rounded-full" style={{ background: EMOTION_COLORS[e] }} />
                    <span style={{ color: '#94a3b8' }}>{e}</span>
                    <span className="ml-auto font-mono" style={{ color: '#475569' }}>{session.emotionDist[e]}%</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div>
            <div className="text-xs font-semibold mb-2" style={{ color: '#64748b' }}>People Timeline</div>
            <ResponsiveContainer width="100%" height={100}>
              <AreaChart data={session.timeline}>
                <defs>
                  <linearGradient id="tGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#00d4ff" stopOpacity={0.2} />
                    <stop offset="95%" stopColor="#00d4ff" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="2 2" stroke="rgba(0,212,255,0.04)" />
                <XAxis dataKey="time" tick={{ fill: '#475569', fontSize: 9 }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: '#334155', fontSize: 9 }} axisLine={false} tickLine={false} />
                <Tooltip content={<CustomTooltip />} />
                <Area type="monotone" dataKey="people" stroke="#00d4ff" strokeWidth={1.5} fill="url(#tGrad)" name="People" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Person IDs */}
        <div>
          <div className="text-xs font-semibold mb-2" style={{ color: '#64748b' }}>Individual Person IDs</div>
          <div className="flex flex-wrap gap-2">
            {session.persons.map(p => (
              <div key={p.id} className="px-3 py-1.5 rounded-lg text-xs" style={{ background: `${EMOTION_COLORS[p.dominant]}12`, border: `1px solid ${EMOTION_COLORS[p.dominant]}30` }}>
                <span className="font-mono font-bold" style={{ color: EMOTION_COLORS[p.dominant] }}>P{p.id}</span>
                <span className="mx-1.5" style={{ color: '#334155' }}>·</span>
                <span style={{ color: '#94a3b8' }}>{EMOTION_ICONS[p.dominant]} {p.dominant}</span>
                <span className="mx-1.5" style={{ color: '#334155' }}>·</span>
                <span className="font-mono" style={{ color: '#64748b' }}>{p.avgConfidence}%</span>
              </div>
            ))}
          </div>
        </div>

        {/* Confidence stats */}
        <div className="grid grid-cols-3 gap-3">
          {[
            { label: 'Highest Confidence', value: `${Math.max(...session.persons.map(p => p.avgConfidence))}%`, color: '#22c55e' },
            { label: 'Lowest Confidence', value: `${Math.min(...session.persons.map(p => p.avgConfidence))}%`, color: '#f97316' },
            { label: 'Std Deviation', value: `±${(Math.random() * 4 + 2).toFixed(1)}%`, color: '#a855f7' },
          ].map(m => (
            <div key={m.label} className="px-3 py-2.5 rounded-lg text-center" style={{ background: '#09101d' }}>
              <div className="text-xs mb-1" style={{ color: '#475569' }}>{m.label}</div>
              <div className="text-lg font-bold font-mono" style={{ color: m.color }}>{m.value}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

export default function SessionHistory() {
  const [search, setSearch] = useState('')
  const [filterExpression, setFilterExpression] = useState<string>('all')
  const [filterDate, setFilterDate] = useState('')
  const [detailSession, setDetailSession] = useState<Session | null>(null)

  const filtered = SESSIONS.filter(s => {
    if (search && !s.id.toLowerCase().includes(search.toLowerCase()) && !s.date.includes(search)) return false
    if (filterExpression !== 'all' && s.dominant !== filterExpression) return false
    if (filterDate && s.date !== filterDate) return false
    return true
  })

  return (
    <div className="flex-1 overflow-y-auto p-5 space-y-5 animate-fade-in">
      {detailSession && <DetailModal session={detailSession} onClose={() => setDetailSession(null)} />}

      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-bold tracking-wide" style={{ color: '#e2e8f0' }}>Session History</h2>
        <div className="flex items-center gap-2">
          <div className="relative">
            <svg className="absolute left-2.5 top-1/2 -translate-y-1/2" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#475569" strokeWidth="2">
              <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
            </svg>
            <input
              value={search}
              onChange={e => setSearch(e.target.value)}
              placeholder="Search sessions..."
              className="text-xs pl-7 pr-3 py-1.5 rounded-lg w-44"
              style={{ background: '#0d1424', color: '#94a3b8', border: '1px solid rgba(0,212,255,0.12)' }}
            />
          </div>
          <select
            value={filterExpression}
            onChange={e => setFilterExpression(e.target.value)}
            className="text-xs px-2.5 py-1.5 rounded-lg"
            style={{ background: '#0d1424', color: '#94a3b8', border: '1px solid rgba(0,212,255,0.12)' }}
          >
            <option value="all">All Expressions</option>
            {EMOTIONS.map(e => <option key={e} value={e}>{e}</option>)}
          </select>
          <input
            type="date"
            value={filterDate}
            onChange={e => setFilterDate(e.target.value)}
            className="text-xs px-2.5 py-1.5 rounded-lg"
            style={{ background: '#0d1424', color: '#64748b', border: '1px solid rgba(0,212,255,0.12)', colorScheme: 'dark' }}
          />
          {(filterDate || filterExpression !== 'all' || search) && (
            <button onClick={() => { setSearch(''); setFilterExpression('all'); setFilterDate('') }} className="text-xs px-2.5 py-1.5 rounded-lg" style={{ background: 'rgba(239,68,68,0.1)', color: '#f87171', border: '1px solid rgba(239,68,68,0.15)' }}>Clear</button>
          )}
        </div>
      </div>

      {/* Table */}
      <div className="rounded-xl overflow-hidden" style={{ border: '1px solid rgba(0,212,255,0.1)' }}>
        <table className="w-full">
          <thead>
            <tr style={{ background: '#0a1120' }}>
              {['Session ID', 'Date', 'Start Time', 'Duration', 'People', 'Dominant', 'Avg Conf', ''].map(h => (
                <th key={h} className="text-left px-4 py-3 text-xs font-semibold" style={{ color: '#475569', borderBottom: '1px solid rgba(0,212,255,0.08)' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filtered.length === 0 && (
              <tr><td colSpan={8} className="text-center py-10 text-xs" style={{ color: '#334155' }}>No sessions match the current filters</td></tr>
            )}
            {filtered.map((s, i) => (
              <tr key={s.id} style={{ background: i % 2 === 0 ? '#0d1424' : '#0b1220', borderBottom: '1px solid rgba(0,212,255,0.05)' }}>
                <td className="px-4 py-3 text-xs font-mono" style={{ color: '#64748b' }}>{s.id}</td>
                <td className="px-4 py-3 text-xs" style={{ color: '#94a3b8' }}>{s.date}</td>
                <td className="px-4 py-3 text-xs font-mono" style={{ color: '#64748b' }}>{s.startTime}</td>
                <td className="px-4 py-3 text-xs font-mono" style={{ color: '#94a3b8' }}>{s.duration}</td>
                <td className="px-4 py-3 text-xs font-bold" style={{ color: '#00d4ff' }}>{s.people}</td>
                <td className="px-4 py-3">
                  <span className="text-xs px-2 py-0.5 rounded font-medium" style={{ background: `${EMOTION_COLORS[s.dominant]}15`, color: EMOTION_COLORS[s.dominant] }}>
                    {EMOTION_ICONS[s.dominant]} {s.dominant}
                  </span>
                </td>
                <td className="px-4 py-3 text-xs font-mono font-bold" style={{ color: '#22c55e' }}>{s.avgConfidence}%</td>
                <td className="px-4 py-3">
                  <button
                    onClick={() => setDetailSession(s)}
                    className="text-xs px-3 py-1.5 rounded-lg transition-all"
                    style={{ background: 'rgba(0,212,255,0.08)', color: '#00d4ff', border: '1px solid rgba(0,212,255,0.18)' }}
                  >
                    View Details
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Summary stats */}
      <div className="grid grid-cols-4 gap-3">
        {[
          { label: 'Total Sessions', value: SESSIONS.length, color: '#00d4ff' },
          { label: 'Total People', value: SESSIONS.reduce((a, s) => a + s.people, 0), color: '#a855f7' },
          { label: 'Avg Session Duration', value: '00:41:55', color: '#f97316' },
          { label: 'Overall Avg Confidence', value: `${(SESSIONS.reduce((a, s) => a + s.avgConfidence, 0) / SESSIONS.length).toFixed(1)}%`, color: '#22c55e' },
        ].map(m => (
          <div key={m.label} className="rounded-xl px-4 py-3 flex items-center gap-3" style={{ background: '#0d1424', border: '1px solid rgba(0,212,255,0.1)' }}>
            <div>
              <div className="text-xs mb-1" style={{ color: '#475569' }}>{m.label}</div>
              <div className="text-xl font-bold font-mono" style={{ color: m.color }}>{m.value}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
