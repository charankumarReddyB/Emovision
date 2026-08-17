import { useState, useEffect, useCallback } from 'react'
import { EMOTIONS, EMOTION_COLORS, EMOTION_ICONS } from '../data'
import type { Emotion, SessionSummary, Screen, PersonDetailCard } from '../types'
import { apiService } from '../services/api'
import { SessionPdfReport } from '../components/SessionPdfReport'
import { Printer, Download, User, ShieldCheck } from 'lucide-react'

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

  // Details Modal & PDF State
  const [selectedDetails, setSelectedDetails] = useState<any | null>(null)
  const [showPdfReport, setShowPdfReport] = useState(false)
  const [pdfAnalyticsData, setPdfAnalyticsData] = useState<any | null>(null)
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
      const analytics = await apiService.getSessionAnalytics(sessionId)
      setSelectedDetails({ ...details, ...analytics })
    } catch (err) {
      console.error('Failed to load session details:', err)
    } finally {
      setLoadingDetails(false)
    }
  }

  const openPdfReport = async (sessionId: string) => {
    try {
      setLoadingDetails(true)
      const analytics = await apiService.getSessionAnalytics(sessionId)
      setPdfAnalyticsData(analytics)
      setShowPdfReport(true)
    } catch (err) {
      console.error('Failed to load PDF report analytics:', err)
    } finally {
      setLoadingDetails(false)
    }
  }

  const totalPages = Math.ceil(totalSessions / limit) || 1

  if (showPdfReport && pdfAnalyticsData) {
    return (
      <SessionPdfReport
        analytics={pdfAnalyticsData}
        sessionName={pdfAnalyticsData.session_name || 'Live Camera Session'}
        onClose={() => setShowPdfReport(false)}
      />
    )
  }

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-6 animate-fade-in">
      {/* Header & Controls */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-slate-100">
            Session History & Person Identification Log
          </h1>
          <p className="text-sm mt-0.5 text-slate-400">
            Historical Records with Person ID Face Identification Cards & PDF Export
          </p>
        </div>

        {/* Refresh button */}
        <button
          onClick={fetchHistory}
          className="px-3.5 py-2 rounded-lg text-xs font-medium flex items-center gap-1.5 transition-all bg-slate-900 text-cyan-400 border border-cyan-500/20 hover:border-cyan-500/40"
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
      <div className="p-4 rounded-xl flex items-center gap-4 bg-slate-900/90 border border-cyan-500/10">
        {/* Search Input */}
        <div className="relative flex-1">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search by Session ID or Name..."
            className="w-full pl-9 pr-3 py-1.5 rounded-lg text-xs font-mono bg-slate-950 text-slate-200 border border-cyan-500/20 focus:outline-none focus:border-cyan-400"
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
            className="text-xs px-3 py-1.5 rounded-lg font-medium bg-slate-950 text-cyan-400 border border-cyan-500/20 focus:outline-none"
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
        <div className="p-3 rounded-lg text-xs bg-red-500/10 border border-red-500/20 text-red-400">
          Backend Error: {error}
        </div>
      )}

      {/* Table */}
      {loading ? (
        <div className="rounded-xl p-8 flex items-center justify-center text-xs text-slate-400 bg-slate-900 border border-cyan-500/10 animate-pulse">
          Loading session history records...
        </div>
      ) : filteredSessions.length === 0 ? (
        <div className="rounded-xl p-8 flex flex-col items-center justify-center text-center gap-2 bg-slate-900/50 border border-dashed border-cyan-500/20">
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
        <div className="rounded-xl overflow-hidden bg-slate-900 border border-cyan-500/10 shadow-xl">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="text-xs font-semibold uppercase tracking-wider bg-slate-950 text-slate-400 border-b border-cyan-500/10">
                <th className="py-3 px-4">Session ID</th>
                <th className="py-3 px-4">Date</th>
                <th className="py-3 px-4">Duration</th>
                <th className="py-3 px-4">Person ID Photos & Expressions</th>
                <th className="py-3 px-4">Dominant Emotion</th>
                <th className="py-3 px-4">Avg Confidence</th>
                <th className="py-3 px-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-cyan-500/5">
              {filteredSessions.map((s) => {
                const domColor = EMOTION_COLORS[s.dominant_expression as Emotion] || '#00d4ff'
                return (
                  <tr key={s.session_id} className="hover:bg-slate-800/40 transition-colors text-xs">
                    <td className="py-3 px-4 font-mono font-medium text-cyan-400">
                      {s.session_id}
                    </td>
                    <td className="py-3 px-4 text-slate-300">{s.date}</td>
                    <td className="py-3 px-4 text-slate-400 font-mono">
                      {Math.round(s.duration_seconds)}s
                    </td>
                    {/* Person Photos & Expressions Column */}
                    <td className="py-3 px-4">
                      {s.persons_details && s.persons_details.length > 0 ? (
                        <div className="flex items-center gap-2.5 overflow-x-auto py-1 max-w-[420px] scrollbar-thin scrollbar-thumb-cyan-500/20">
                          {s.persons_details.map((p) => (
                            <div
                              key={p.person_id}
                              className="flex items-center gap-2 bg-slate-950 border border-cyan-500/20 rounded-lg p-1.5 pr-3 flex-shrink-0 shadow-md min-w-[110px]"
                            >
                              {p.thumbnail_b64 ? (
                                <img
                                  src={p.thumbnail_b64}
                                  alt={`Person ${p.person_id}`}
                                  className="w-10 h-10 rounded-md object-cover flex-shrink-0 border border-cyan-500/50"
                                />
                              ) : (
                                <div className="w-10 h-10 rounded-md bg-slate-800 flex-shrink-0 flex items-center justify-center text-slate-400 border border-slate-700">
                                  <User size={16} />
                                </div>
                              )}
                              <div className="flex flex-col text-[11px] leading-tight">
                                <span className="font-mono font-bold text-cyan-400">#{p.person_id}</span>
                                <span
                                  className="font-semibold mt-0.5"
                                  style={{
                                    color: EMOTION_COLORS[p.dominant_emotion as Emotion] || '#00d4ff',
                                  }}
                                >
                                  {p.dominant_emotion}
                                </span>
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <span className="text-slate-500 text-[11px] italic">
                          {s.people_count} Person{s.people_count === 1 ? '' : 's'} logged
                        </span>
                      )}
                    </td>
                    <td className="py-3 px-4">
                      <span
                        className="px-2.5 py-1 rounded text-xs font-medium inline-flex items-center gap-1.5 shadow-sm"
                        style={{
                          background: `${domColor}18`,
                          color: domColor,
                          border: `1px solid ${domColor}30`,
                        }}
                      >
                        <span>{EMOTION_ICONS[s.dominant_expression as Emotion] || '😐'}</span>
                        <span>{s.dominant_expression}</span>
                      </span>
                    </td>
                    <td className="py-3 px-4 text-slate-300 font-mono font-bold">
                      {s.average_confidence}%
                    </td>
                    <td className="py-3 px-4 text-right">
                      <div className="flex flex-col items-end gap-1.5">
                        <button
                          onClick={() => openPdfReport(s.session_id)}
                          className="w-28 px-2.5 py-1 rounded text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 hover:bg-emerald-500/20 transition-all flex items-center justify-center gap-1.5 shadow-sm"
                        >
                          <Printer size={12} />
                          <span>PDF Report</span>
                        </button>
                        <button
                          onClick={() => openDetailsModal(s.session_id)}
                          className="w-28 px-2.5 py-1 rounded text-xs font-medium bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 hover:bg-cyan-500/20 transition-all text-center shadow-sm"
                        >
                          View Details
                        </button>
                      </div>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>

          {/* Pagination Footer */}
          <div className="px-4 py-3 border-t border-cyan-500/10 bg-slate-950 flex items-center justify-between text-xs">
            <span className="text-slate-400">
              Page <strong className="text-cyan-400">{page}</strong> of{' '}
              <strong className="text-slate-200">{totalPages}</strong> ({totalSessions} total records)
            </span>
            <div className="flex items-center gap-2">
              <button
                disabled={page <= 1}
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                className="px-3 py-1 rounded font-medium disabled:opacity-40 bg-slate-900 text-slate-300 border border-cyan-500/15"
              >
                Previous
              </button>
              <button
                disabled={page >= totalPages}
                onClick={() => setPage((p) => p + 1)}
                className="px-3 py-1 rounded font-medium disabled:opacity-40 bg-slate-900 text-slate-300 border border-cyan-500/15"
              >
                Next
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Session Details & Person Identification Modal */}
      {selectedDetails && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-fade-in overflow-y-auto">
          <div className="w-full max-w-2xl rounded-2xl p-6 space-y-6 bg-slate-950 border border-cyan-500/30 shadow-2xl relative my-8">
            {/* Modal Header */}
            <div className="flex items-center justify-between border-b border-slate-800 pb-4">
              <div>
                <h3 className="text-lg font-bold text-white flex items-center gap-2">
                  <ShieldCheck className="text-cyan-400" size={20} />
                  <span>Session & Person Identification Details</span>
                </h3>
                <p className="text-xs font-mono text-cyan-400 mt-0.5">
                  ID: {selectedDetails.session_id}
                </p>
              </div>
              <button
                onClick={() => setSelectedDetails(null)}
                className="text-slate-400 hover:text-white text-lg px-2"
              >
                ✕
              </button>
            </div>

            {/* Session Stats Grid */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
              <div className="p-3 rounded-xl bg-slate-900 border border-slate-800">
                <span className="text-slate-500 block mb-0.5">People Detected</span>
                <span className="text-cyan-400 font-bold text-base">{selectedDetails.total_people_detected || selectedDetails.people_count || 0}</span>
              </div>
              <div className="p-3 rounded-xl bg-slate-900 border border-slate-800">
                <span className="text-slate-500 block mb-0.5">Dominant Expression</span>
                <span className="text-emerald-400 font-bold text-base">{selectedDetails.dominant_expression}</span>
              </div>
              <div className="p-3 rounded-xl bg-slate-900 border border-slate-800">
                <span className="text-slate-500 block mb-0.5">Avg Confidence</span>
                <span className="text-purple-400 font-bold text-base">{selectedDetails.average_confidence}%</span>
              </div>
              <div className="p-3 rounded-xl bg-slate-900 border border-slate-800">
                <span className="text-slate-500 block mb-0.5">Duration</span>
                <span className="text-amber-400 font-bold text-base">{Math.round(selectedDetails.session_duration_seconds || selectedDetails.duration_seconds || 0)}s</span>
              </div>
            </div>

            {/* Person Identification Cards Gallery */}
            <div>
              <h4 className="text-sm font-bold text-white mb-3 flex items-center gap-2 border-b border-slate-800 pb-2">
                <User size={16} className="text-cyan-400" />
                <span>Detected Person Face Cards & Identification</span>
              </h4>

              {selectedDetails.persons_details && selectedDetails.persons_details.length > 0 ? (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 max-h-60 overflow-y-auto pr-1">
                  {selectedDetails.persons_details.map((person: PersonDetailCard) => (
                    <div
                      key={person.person_id}
                      className="bg-slate-900 border border-slate-800 p-3 rounded-xl flex items-center space-x-3 hover:border-cyan-500/40 transition"
                    >
                      {/* Face Thumbnail */}
                      <div className="w-14 h-14 bg-slate-800 rounded-lg overflow-hidden border border-cyan-500/30 flex items-center justify-center flex-shrink-0">
                        {person.thumbnail_b64 ? (
                          <img
                            src={person.thumbnail_b64}
                            alt={`Person ${person.person_id} face crop`}
                            className="w-full h-full object-cover"
                          />
                        ) : (
                          <User size={24} className="text-slate-500" />
                        )}
                      </div>

                      {/* Person info */}
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
              ) : (
                <div className="p-4 bg-slate-900 rounded-xl text-center text-xs text-slate-500">
                  No individual person face photos logged for this session.
                </div>
              )}
            </div>

            {/* Actions Footer */}
            <div className="pt-3 border-t border-slate-800 flex items-center justify-between">
              <button
                onClick={() => {
                  setSelectedDetails(null)
                  openPdfReport(selectedDetails.session_id)
                }}
                className="px-4 py-2 rounded-lg text-xs font-semibold bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white flex items-center gap-1.5 shadow-lg shadow-cyan-500/20"
              >
                <Printer size={14} />
                <span>Export PDF Report</span>
              </button>

              <button
                onClick={() => setSelectedDetails(null)}
                className="px-4 py-2 rounded-lg text-xs font-semibold bg-slate-800 text-slate-300 hover:bg-slate-700"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
