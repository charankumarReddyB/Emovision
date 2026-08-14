import { useState } from 'react'
import {
  PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  LineChart, Line, CartesianGrid, Legend, AreaChart, Area,
} from 'recharts'
import { SESSIONS, EMOTIONS, EMOTION_COLORS, EMOTION_ICONS } from '../data'
import type { Emotion, Session } from '../types'

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload?.length) {
    return (
      <div className="px-3 py-2 rounded-lg text-xs space-y-1" style={{ background: '#131e30', border: '1px solid rgba(0,212,255,0.15)', color: '#e2e8f0' }}>
        {label && <div style={{ color: '#64748b' }}>{label}</div>}
        {payload.map((p: any, i: number) => (
          <div key={i}><span style={{ color: p.color || '#00d4ff' }}>{p.name}</span>: <strong>{p.value}</strong></div>
        ))}
      </div>
    )
  }
  return null
}

function SessionSelector({ sessions, selected, onSelect }: { sessions: Session[]; selected: Session; onSelect: (s: Session) => void }) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs" style={{ color: '#64748b' }}>Session:</span>
      <select
        value={selected.id}
        onChange={e => onSelect(sessions.find(s => s.id === e.target.value)!)}
        className="text-xs px-2.5 py-1.5 rounded-lg"
        style={{ background: '#0d1424', color: '#94a3b8', border: '1px solid rgba(0,212,255,0.15)' }}
      >
        {sessions.map(s => <option key={s.id} value={s.id}>{s.id} — {s.date}</option>)}
      </select>
    </div>
  )
}

export default function Analytics() {
  const [session, setSession] = useState(SESSIONS[0])
  const [selectedPerson, setSelectedPerson] = useState<number | null>(null)

  const person = selectedPerson !== null ? session.persons.find(p => p.id === selectedPerson) : null

  const donutData = EMOTIONS.map(e => ({ name: e, value: session.emotionDist[e] }))
  const barData = EMOTIONS.map(e => ({ name: e.slice(0, 3), value: session.emotionDist[e], color: EMOTION_COLORS[e] }))

  const confidenceData = session.persons.map(p => ({ name: `P${p.id}`, confidence: p.avgConfidence, emotion: p.dominant }))

  return (
    <div className="flex-1 overflow-y-auto p-5 space-y-5 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-bold tracking-wide" style={{ color: '#e2e8f0' }}>Analytics</h2>
        <SessionSelector sessions={SESSIONS} selected={session} onSelect={s => { setSession(s); setSelectedPerson(null) }} />
      </div>

      {/* Session metadata */}
      <div className="grid grid-cols-6 gap-3">
        {[
          { label: 'Session ID', value: session.id, mono: true },
          { label: 'Date', value: session.date },
          { label: 'Duration', value: session.duration, mono: true },
          { label: 'People Detected', value: session.people },
          { label: 'Avg Confidence', value: `${session.avgConfidence}%` },
          { label: 'Dominant Expression', value: session.dominant, color: EMOTION_COLORS[session.dominant] },
        ].map(m => (
          <div key={m.label} className="rounded-xl p-3" style={{ background: '#0d1424', border: '1px solid rgba(0,212,255,0.1)' }}>
            <div className="text-xs mb-1" style={{ color: '#64748b' }}>{m.label}</div>
            <div className="text-sm font-bold" style={{ color: m.color || '#e2e8f0', fontFamily: m.mono ? 'monospace' : undefined }}>{m.value}</div>
          </div>
        ))}
      </div>

      {/* Charts row 1 */}
      <div className="grid grid-cols-5 gap-4">
        {/* Donut */}
        <div className="col-span-2 rounded-xl p-4" style={{ background: '#0d1424', border: '1px solid rgba(0,212,255,0.1)' }}>
          <div className="text-xs font-semibold mb-3" style={{ color: '#94a3b8' }}>Expression Distribution</div>
          <div className="flex items-center gap-3">
            <ResponsiveContainer width={130} height={130}>
              <PieChart>
                <Pie data={donutData} cx="50%" cy="50%" innerRadius={38} outerRadius={60} paddingAngle={2} dataKey="value">
                  {donutData.map(e => <Cell key={e.name} fill={EMOTION_COLORS[e.name as Emotion]} />)}
                </Pie>
                <Tooltip content={<CustomTooltip />} />
              </PieChart>
            </ResponsiveContainer>
            <div className="flex flex-col gap-1">
              {EMOTIONS.map(e => (
                <div key={e} className="flex items-center gap-1.5 text-xs">
                  <span className="w-1.5 h-1.5 rounded-full" style={{ background: EMOTION_COLORS[e] }} />
                  <span style={{ color: '#94a3b8' }}>{e}</span>
                  <span className="ml-auto font-mono text-xs" style={{ color: '#475569' }}>{session.emotionDist[e]}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Bar */}
        <div className="col-span-3 rounded-xl p-4" style={{ background: '#0d1424', border: '1px solid rgba(0,212,255,0.1)' }}>
          <div className="text-xs font-semibold mb-3" style={{ color: '#94a3b8' }}>Expression Frequency</div>
          <ResponsiveContainer width="100%" height={120}>
            <BarChart data={barData} barSize={26}>
              <XAxis dataKey="name" tick={{ fill: '#475569', fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: '#334155', fontSize: 10 }} axisLine={false} tickLine={false} />
              <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(0,212,255,0.04)' }} />
              <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                {barData.map((e, i) => <Cell key={i} fill={e.color} fillOpacity={0.85} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Charts row 2 */}
      <div className="grid grid-cols-2 gap-4">
        {/* Timeline */}
        <div className="rounded-xl p-4" style={{ background: '#0d1424', border: '1px solid rgba(0,212,255,0.1)' }}>
          <div className="text-xs font-semibold mb-3" style={{ color: '#94a3b8' }}>People Over Time</div>
          <ResponsiveContainer width="100%" height={120}>
            <AreaChart data={session.timeline}>
              <defs>
                <linearGradient id="pGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#00d4ff" stopOpacity={0.2} />
                  <stop offset="95%" stopColor="#00d4ff" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,212,255,0.05)" />
              <XAxis dataKey="time" tick={{ fill: '#475569', fontSize: 10 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: '#334155', fontSize: 10 }} axisLine={false} tickLine={false} />
              <Tooltip content={<CustomTooltip />} />
              <Area type="monotone" dataKey="people" stroke="#00d4ff" strokeWidth={1.5} fill="url(#pGrad)" name="People" />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Confidence graph */}
        <div className="rounded-xl p-4" style={{ background: '#0d1424', border: '1px solid rgba(0,212,255,0.1)' }}>
          <div className="text-xs font-semibold mb-3" style={{ color: '#94a3b8' }}>Confidence by Person</div>
          <ResponsiveContainer width="100%" height={120}>
            <BarChart data={confidenceData} barSize={20}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,212,255,0.05)" />
              <XAxis dataKey="name" tick={{ fill: '#475569', fontSize: 10 }} axisLine={false} tickLine={false} />
              <YAxis domain={[60, 100]} tick={{ fill: '#334155', fontSize: 10 }} axisLine={false} tickLine={false} />
              <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(0,212,255,0.04)' }} />
              <Bar dataKey="confidence" name="Confidence %" radius={[3, 3, 0, 0]}>
                {confidenceData.map((d, i) => <Cell key={i} fill={EMOTION_COLORS[d.emotion as Emotion]} fillOpacity={0.8} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Expression timeline line chart */}
      <div className="rounded-xl p-4" style={{ background: '#0d1424', border: '1px solid rgba(0,212,255,0.1)' }}>
        <div className="text-xs font-semibold mb-3" style={{ color: '#94a3b8' }}>Expression Timeline</div>
        <ResponsiveContainer width="100%" height={110}>
          <LineChart data={session.timeline}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,212,255,0.05)" />
            <XAxis dataKey="time" tick={{ fill: '#475569', fontSize: 10 }} axisLine={false} tickLine={false} />
            <YAxis tick={{ fill: '#334155', fontSize: 10 }} axisLine={false} tickLine={false} />
            <Tooltip content={<CustomTooltip />} />
            <Line type="monotone" dataKey="people" stroke="#00d4ff" strokeWidth={1.5} dot={{ fill: '#00d4ff', r: 3 }} name="People" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Individual person analysis */}
      <div className="rounded-xl p-4" style={{ background: '#0d1424', border: '1px solid rgba(0,212,255,0.1)' }}>
        <div className="flex items-center justify-between mb-4">
          <div className="text-xs font-semibold" style={{ color: '#94a3b8' }}>Individual Person Analysis</div>
          <div className="flex items-center gap-2">
            <span className="text-xs" style={{ color: '#64748b' }}>Select Person:</span>
            <div className="flex gap-1.5 flex-wrap">
              {session.persons.map(p => (
                <button
                  key={p.id}
                  onClick={() => setSelectedPerson(selectedPerson === p.id ? null : p.id)}
                  className="text-xs px-2.5 py-1 rounded-lg font-mono transition-all"
                  style={{
                    background: selectedPerson === p.id ? `${EMOTION_COLORS[p.dominant]}25` : 'rgba(0,212,255,0.06)',
                    border: `1px solid ${selectedPerson === p.id ? EMOTION_COLORS[p.dominant] : 'rgba(0,212,255,0.12)'}`,
                    color: selectedPerson === p.id ? EMOTION_COLORS[p.dominant] : '#64748b',
                  }}
                >P{p.id}</button>
              ))}
            </div>
          </div>
        </div>

        {person ? (
          <div className="grid grid-cols-3 gap-4">
            {/* Person info */}
            <div className="space-y-2">
              {[
                { label: 'Person ID', value: `P${person.id}`, mono: true },
                { label: 'Dominant Expression', value: `${EMOTION_ICONS[person.dominant]} ${person.dominant}`, color: EMOTION_COLORS[person.dominant] },
                { label: 'Avg Confidence', value: `${person.avgConfidence}%`, color: '#22c55e' },
              ].map(m => (
                <div key={m.label} className="px-3 py-2 rounded-lg" style={{ background: '#0a1120' }}>
                  <div className="text-xs mb-0.5" style={{ color: '#475569' }}>{m.label}</div>
                  <div className="text-sm font-bold" style={{ color: m.color || '#e2e8f0', fontFamily: m.mono ? 'monospace' : undefined }}>{m.value}</div>
                </div>
              ))}

              {/* Emotion dist */}
              <div className="px-3 py-2 rounded-lg space-y-1" style={{ background: '#0a1120' }}>
                <div className="text-xs mb-1" style={{ color: '#475569' }}>Expression Distribution</div>
                {EMOTIONS.filter(e => person.emotionDist[e] > 0).map(e => (
                  <div key={e} className="flex items-center gap-2 text-xs">
                    <span className="w-1.5 h-1.5 rounded-full" style={{ background: EMOTION_COLORS[e] }} />
                    <span style={{ color: '#94a3b8' }}>{e}</span>
                    <div className="flex-1 h-1 rounded-full overflow-hidden" style={{ background: '#0d1424' }}>
                      <div style={{ width: `${person.emotionDist[e]}%`, background: EMOTION_COLORS[e], height: '100%', borderRadius: 2 }} />
                    </div>
                    <span className="font-mono" style={{ color: '#475569' }}>{person.emotionDist[e]}%</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Timeline */}
            <div className="col-span-2 px-3 py-3 rounded-lg" style={{ background: '#0a1120' }}>
              <div className="text-xs mb-3" style={{ color: '#475569' }}>Expression Timeline</div>
              <div className="flex items-center gap-2 flex-wrap">
                {person.timeline.map((e, i) => (
                  <div key={i} className="flex items-center gap-1">
                    <span
                      className="text-xs px-2.5 py-1.5 rounded-lg font-medium"
                      style={{ background: `${EMOTION_COLORS[e]}15`, border: `1px solid ${EMOTION_COLORS[e]}35`, color: EMOTION_COLORS[e] }}
                    >
                      {EMOTION_ICONS[e]} {e}
                    </span>
                    {i < person.timeline.length - 1 && (
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#334155" strokeWidth="2"><polyline points="9 18 15 12 9 6" /></svg>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <div className="py-8 text-center text-xs" style={{ color: '#334155' }}>
            Select a person ID above to view individual analysis
          </div>
        )}
      </div>
    </div>
  )
}
