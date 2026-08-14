import { useState, useEffect, useCallback } from 'react'
import { EMOTIONS, EMOTION_COLORS, EMOTION_ICONS } from '../data'
import type { Emotion, SessionSummary, Screen } from '../types'
import { apiService } from '../services/api'

interface Props {
  onNavigate?: (screen: Screen, sessionId?: string) => void
}

export default function SessionHistory({ onNavigate }: Props) {
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [totalSessions, setTotalSessions] = useState(0)
  const [page, setPage] = useState(1)
  const limit = 10

  const [searchQuery, setSearchQuery] = useState('')
  const [emotionFilter, setEmotionFilter] = useState<string>('all')

  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Details Modal State
  const [selectedDetails, setSelectedDetails] = useState<any | null>(null)
  const [loadingDetails, setLoadingDetails] = useState(false)

  const fetchHistory = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      const res = await apiService.getSessions(page, limit)
      setSessions(res.sessions || [])
      setTotalSessions(res.total || 0)
    } catch (err: any) {
      setError(err.message || 'Failed to load session history from backend')
    } finally {
      setLoading(false)
    }
  }, [page, limit])

  useEffect(() => {
    fetchHistory()
  }, [fetchHistory])

  // Filter sessions locally by search query and emotion
  const filteredSessions = sessions.filter((s) => {
    const matchesSearch =
      s.session_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (s.session_name && s.session_name.toLowerCase().includes(searchQuery.toLowerCase()))
    const matchesEmotion =
      emotionFilter === 'all' || s.dominant_expression === emotionFilter
    return matchesSearch && matchesEmotion
  })

  const openDetailsModal = async (sessionId: string) => {
    try {
      setLoadingDetails(true)
      const details = await apiService.getSessionDetails(sessionId)
      setSelectedDetails(details)
    } catch (err) {
      console.error('Failed to load session details:', err)
    } finally {
      setLoadingDetails(false)
    }
  }

  const totalPages = Math.ceil(totalSessions / limit) || 1

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-6 animate-fade-in">
      {/* Header & Controls */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold tracking-tight" style={{ color: '#e2e8f0' }}>
            Session History Log
          </h1>
          <p className="text-sm mt-0.5" style={{ color: '#64748b' }}>
            Historical Database Records of Multi-Person Emotion Recognition Runs
          </p>
        </div>

        {/* Refresh button */}
        <button
          onClick={fetchHistory}
          className="px-3 py-1.5 rounded-lg text-xs font-medium flex items-center gap-1.5 transition-all"
          style={{
            background: '#0d1424',
            color: '#00d4ff',
            border: '1px solid rgba(0,212,255,0.2)',
          }}
        >
          <svg
            width="12"
            height="12"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <path d="M23 4v6h-6" />
            <path d="M1 20v-6h6" />
            <path d="M3.51 9a9 9 0 0114.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0020.49 15" />
          </svg>
          Refresh Log
        </button>
      </div>

      {/* Filter and Search Bar */}
      <div
        className="p-4 rounded-xl flex items-center gap-4"
        style={{ background: '#0d1424', border: '1px solid rgba(0,212,255,0.1)' }}
      >
        {/* Search Input */}
        <div className="relative flex-1">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search by Session ID or Name..."
            className="w-full pl-9 pr-3 py-1.5 rounded-lg text-xs font-mono focus:outline-none"
            style={{
              background: '#070b13',
              color: '#e2e8f0',
              border: '1px solid rgba(0,212,255,0.15)',
            }}
          />
          <svg
            className="absolute left-3 top-2.5"
            width="12"
            height="12"
            viewBox="0 0 24 24"
            fill="none"
            stroke="#64748b"
            strokeWidth="2"
          >
            <circle cx="11" cy="11" r="8" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
        </div>

        {/* Emotion Filter */}
        <div className="flex items-center gap-2">
          <label className="text-xs text-slate-400">Expression:</label>
          <select
            value={emotionFilter}
            onChange={(e) => setEmotionFilter(e.target.value)}
            className="text-xs px-3 py-1.5 rounded-lg font-medium focus:outline-none"
            style={{
              background: '#070b13',
              color: '#00d4ff',
              border: '1px solid rgba(0,212,255,0.15)',
            }}
          >
            <option value="all">All Expressions</option>
            {EMOTIONS.map((e) => (
              <option key={e} value={e}>
                {EMOTION_ICONS[e]} {e}
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
          Backend Error: {error}
        </div>
      )}

      {/* Table */}
      {loading ? (
        <div
          className="rounded-xl p-8 flex items-center justify-center text-xs text-slate-400 animate-pulse"
          style={{ background: '#0d1424', border: '1px solid rgba(0,212,255,0.1)' }}
        >
          Loading session history records...
        </div>
      ) : filteredSessions.length === 0 ? (
        <div
          className="rounded-xl p-8 flex flex-col items-center justify-center text-center gap-2"
          style={{ background: '#0d1424', border: '1px dashed rgba(0,212,255,0.2)' }}
        >
          <svg
            width="36"
            height="36"
            viewBox="0 0 24 24"
            fill="none"
            stroke="rgba(0,212,255,0.3)"
            strokeWidth="1.5"
          >
            <rect x="3" y="4" width="18" height="16" rx="2" />
            <line x1="16" y1="2" x2="16" y2="6" />
            <line x1="8" y1="2" x2="8" y2="6" />
            <line x1="3" y1="10" x2="21" y2="10" />
          </svg>
          <span className="text-xs text-slate-300 font-medium">No Matching Session Records Found</span>
          <span className="text-xs text-slate-500 max-w-xs">
            Try resetting your search query or expression filter, or record a new session.
          </span>
        </div>
      ) : (
        <div
          className="rounded-xl overflow-hidden"
          style={{ background: '#0d1424', border: '1px solid rgba(0,212,255,0.1)' }}
        >
          <table className="w-full text-left border-collapse">
            <thead>
              <tr
                className="text-xs font-semibold uppercase tracking-wider border-b"
                style={{
                  borderColor: 'rgba(0,212,255,0.08)',
                  background: '#09101d',
                  color: '#475569',
                }}
              >
                <th className="py-3 px-4">Session ID</th>
                <th className="py-3 px-4">Date</th>
                <th className="py-3 px-4">Duration</th>
                <th className="py-3 px-4">People Count</th>
                <th className="py-3 px-4">Dominant Expression</th>
                <th className="py-3 px-4">Avg Confidence</th>
                <th className="py-3 px-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y" style={{ borderColor: 'rgba(0,212,255,0.05)' }}>
              {filteredSessions.map((s) => {
                const domColor = EMOTION_COLORS[s.dominant_expression as Emotion] || '#00d4ff'
                return (
                  <tr key={s.session_id} className="hover:bg-slate-900/40 transition-colors text-xs">
                    <td className="py-3 px-4 font-mono font-medium text-cyan-400">
                      {s.session_id}
                    </td>
                    <td className="py-3 px-4 text-slate-300">{s.date}</td>
                    <td className="py-3 px-4 text-slate-400 font-mono">
                      {Math.round(s.duration_seconds)}s
                    </td>
                    <td className="py-3 px-4 text-slate-300 font-mono">
                      {s.people_count}
                    </td>
                    <td className="py-3 px-4">
                      <span
                        className="px-2 py-0.5 rounded text-xs font-medium"
                        style={{
                          background: `${domColor}18`,
                          color: domColor,
                          border: `1px solid ${domColor}30`,
                        }}
                      >
                        {EMOTION_ICONS[s.dominant_expression as Emotion] || '😐'}{' '}
                        {s.dominant_expression}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-slate-300 font-mono font-bold">
                      {s.average_confidence}%
                    </td>
                    <td className="py-3 px-4 text-right space-x-2">
                      <button
                        onClick={() => openDetailsModal(s.session_id)}
                        className="px-2.5 py-1 rounded text-xs font-medium transition-all"
                        style={{
                          background: 'rgba(0,212,255,0.1)',
                          color: '#00d4ff',
                          border: '1px solid rgba(0,212,255,0.25)',
                        }}
                      >
                        View Details
                      </button>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>

          {/* Pagination Footer */}
          <div
            className="px-4 py-3 border-t flex items-center justify-between text-xs"
            style={{ borderColor: 'rgba(0,212,255,0.08)', background: '#09101d' }}
          >
            <span className="text-slate-400">
              Page <strong className="text-cyan-400">{page}</strong> of{' '}
              <strong className="text-slate-200">{totalPages}</strong> ({totalSessions} total records)
            </span>
            <div className="flex items-center gap-2">
              <button
                disabled={page <= 1}
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                className="px-3 py-1 rounded font-medium disabled:opacity-40"
                style={{
                  background: '#0d1424',
                  color: '#94a3b8',
                  border: '1px solid rgba(0,212,255,0.15)',
                }}
              >
                Previous
              </button>
              <button
                disabled={page >= totalPages}
                onClick={() => setPage((p) => p + 1)}
                className="px-3 py-1 rounded font-medium disabled:opacity-40"
                style={{
                  background: '#0d1424',
                  color: '#94a3b8',
                  border: '1px solid rgba(0,212,255,0.15)',
                }}
              >
                Next
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Details Inspection Modal */}
      {selectedDetails && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-sm animate-fade-in">
          <div
            className="w-full max-w-lg rounded-xl p-6 space-y-4 shadow-2xl relative"
            style={{
              background: '#09101d',
              border: '1px solid rgba(0,212,255,0.25)',
            }}
          >
            <div className="flex items-center justify-between border-b pb-3" style={{ borderColor: 'rgba(0,212,255,0.1)' }}>
              <div>
                <h3 className="text-sm font-bold text-slate-100">
                  Session Inspection Details
                </h3>
                <p className="text-xs font-mono text-cyan-400 mt-0.5">
                  {selectedDetails.session_id}
                </p>
              </div>
              <button
                onClick={() => setSelectedDetails(null)}
                className="text-slate-400 hover:text-white text-base"
              >
                ✕
              </button>
            </div>

            <div className="grid grid-cols-2 gap-3 text-xs">
              <div className="p-2.5 rounded" style={{ background: '#0d1424' }}>
                <span className="text-slate-500 block mb-0.5">Session Name</span>
                <span className="text-slate-200 font-medium">
                  {selectedDetails.session_name || 'Webcam Session'}
                </span>
              </div>
              <div className="p-2.5 rounded" style={{ background: '#0d1424' }}>
                <span className="text-slate-500 block mb-0.5">Source Type</span>
                <span className="text-cyan-400 font-mono">
                  {selectedDetails.source_type}
                </span>
              </div>
              <div className="p-2.5 rounded" style={{ background: '#0d1424' }}>
                <span className="text-slate-500 block mb-0.5">Start Time</span>
                <span className="text-slate-300 font-mono">
                  {selectedDetails.start_time || 'N/A'}
                </span>
              </div>
              <div className="p-2.5 rounded" style={{ background: '#0d1424' }}>
                <span className="text-slate-500 block mb-0.5">End Time</span>
                <span className="text-slate-300 font-mono">
                  {selectedDetails.end_time || 'N/A'}
                </span>
              </div>
              <div className="p-2.5 rounded" style={{ background: '#0d1424' }}>
                <span className="text-slate-500 block mb-0.5">Total Predictions</span>
                <span className="text-purple-400 font-mono font-bold">
                  {selectedDetails.total_predictions}
                </span>
              </div>
              <div className="p-2.5 rounded" style={{ background: '#0d1424' }}>
                <span className="text-slate-500 block mb-0.5">People Detected</span>
                <span className="text-orange-400 font-mono font-bold">
                  {selectedDetails.total_people_detected}
                </span>
              </div>
              <div className="p-2.5 rounded" style={{ background: '#0d1424' }}>
                <span className="text-slate-500 block mb-0.5">Dominant Expression</span>
                <span className="text-green-400 font-bold">
                  {selectedDetails.dominant_expression}
                </span>
              </div>
              <div className="p-2.5 rounded" style={{ background: '#0d1424' }}>
                <span className="text-slate-500 block mb-0.5">Average FPS</span>
                <span className="text-cyan-400 font-mono font-bold">
                  {selectedDetails.avg_fps}
                </span>
              </div>
            </div>

            <div className="pt-2 flex justify-end">
              <button
                onClick={() => setSelectedDetails(null)}
                className="px-4 py-1.5 rounded text-xs font-semibold"
                style={{ background: '#00d4ff', color: '#080c14' }}
              >
                Close Inspection
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
