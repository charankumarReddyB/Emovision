import React, { useState, useEffect, useRef } from 'react'
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, AreaChart, Area, XAxis, YAxis } from 'recharts'
import { EMOTION_COLORS, EMOTION_ICONS } from '../data'

interface VideoTimelinePoint {
  timestamp_sec: number
  frame_idx: number
  face_count: number
  dominant_emotion: string
  detections: Array<{
    bbox: [number, number, number, number]
    emotion: string
    confidence: number
  }>
}

interface VideoAnalysisResult {
  success: boolean
  message?: string
  analysis_id?: string
  filename?: string
  video_duration_seconds?: number
  total_frames_analyzed?: number
  total_face_detections?: number
  dominant_emotion?: string
  average_confidence?: number
  emotion_distribution?: Record<string, { count: number; percentage: number }>
  timeline?: VideoTimelinePoint[]
}

const STORAGE_KEY = 'emovision_active_video_id'

export default function VideoUploadAnalysis() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [progressText, setProgressText] = useState('Initializing...')
  const [result, setResult] = useState<VideoAnalysisResult | null>(null)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)
  const [activeAnalysisId, setActiveAnalysisId] = useState<string | null>(null)

  const pollIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const stopPolling = () => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current)
      pollIntervalRef.current = null
    }
  }

  // Poll video status from backend
  const startPollingStatus = (analysisId: string) => {
    stopPolling()
    setLoading(true)
    setActiveAnalysisId(analysisId)
    localStorage.setItem(STORAGE_KEY, analysisId)

    pollIntervalRef.current = setInterval(async () => {
      try {
        const res = await fetch(`http://127.0.0.1:8000/api/analyze/video/${analysisId}/status`)
        if (!res.ok) {
          stopPolling()
          setLoading(false)
          return
        }

        const statusData = await res.json()
        const prog = statusData.progress || 0
        setProgress(prog)
        setProgressText(`Analyzing video frames (${statusData.frames_processed || 0} / ${statusData.total_frames_to_process || '?'}) — ${prog}%`)

        if (statusData.status === 'completed') {
          stopPolling()
          fetchFinalResult(analysisId)
        } else if (statusData.status === 'failed') {
          stopPolling()
          setLoading(false)
          setErrorMsg(statusData.error_message || 'Video analysis failed')
          localStorage.removeItem(STORAGE_KEY)
        }
      } catch (err) {
        console.warn('Status poll error:', err)
      }
    }, 1000)
  }

  const fetchFinalResult = async (analysisId: string) => {
    try {
      const res = await fetch(`http://127.0.0.1:8000/api/analyze/video/${analysisId}/result`)
      if (res.ok) {
        const resData = await res.json()
        setResult(resData)
        if (!resData.success && resData.message) {
          setErrorMsg(resData.message)
        }
      }
    } catch (err: any) {
      setErrorMsg(err.message || 'Could not fetch final video results')
    } finally {
      setLoading(false)
      setProgress(100)
    }
  }

  // Check on mount if there is an active running or recently finished video job
  useEffect(() => {
    const savedId = localStorage.getItem(STORAGE_KEY)
    if (savedId) {
      setActiveAnalysisId(savedId)
      startPollingStatus(savedId)
    } else {
      // Query backend for active running video job
      fetch('http://127.0.0.1:8000/api/analyze/video/active')
        .then((r) => r.json())
        .then((data) => {
          if (data.active_job && data.analysis_id) {
            startPollingStatus(data.analysis_id)
          }
        })
        .catch(() => {})
    }

    return () => stopPolling()
  }, [])

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0]
      setSelectedFile(file)
      setPreviewUrl(URL.createObjectURL(file))
      setResult(null)
      setErrorMsg(null)
    }
  }

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0]
      setSelectedFile(file)
      setPreviewUrl(URL.createObjectURL(file))
      setResult(null)
      setErrorMsg(null)
    }
  }

  const handleAnalyze = async () => {
    if (!selectedFile) return
    setLoading(true)
    setProgress(5)
    setProgressText('Uploading video to backend...')
    setErrorMsg(null)
    setResult(null)

    try {
      const formData = new FormData()
      formData.append('file', selectedFile)

      const response = await fetch('http://127.0.0.1:8000/api/analyze/video', {
        method: 'POST',
        body: formData,
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to submit video analysis')
      }

      if (data.analysis_id) {
        startPollingStatus(data.analysis_id)
      }
    } catch (err: any) {
      setErrorMsg(err.message || 'Error uploading video')
      setLoading(false)
    }
  }

  const donutData = result?.emotion_distribution
    ? Object.entries(result.emotion_distribution).map(([name, stat]) => ({
        name,
        value: stat.count,
      }))
    : []

  const chartTimeline = result?.timeline
    ? result.timeline.map((pt) => ({
        time: `${pt.timestamp_sec}s`,
        faces: pt.face_count,
        emotion: pt.dominant_emotion,
      }))
    : []

  return (
    <div className="flex-1 flex flex-col overflow-y-auto p-6 space-y-6 animate-fade-in" style={{ background: '#070d19', color: '#f8fafc' }}>
      {/* Top Control Bar */}
      <div
        className="p-6 rounded-2xl border flex flex-col md:flex-row items-center justify-between gap-6"
        style={{
          background: 'rgba(15,23,42,0.6)',
          borderColor: 'rgba(0,212,255,0.15)',
          backdropFilter: 'blur(12px)',
        }}
      >
        <div className="space-y-1">
          <h2 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
            <span className="p-2 rounded-lg bg-cyan-500/10 text-cyan-400">🎬</span>
            Video Frame-by-Frame Emotion Analysis
          </h2>
          <p className="text-sm text-slate-400">
            Upload a video (MP4, AVI, MOV, WEBM). Processing runs independently in backend — tab navigation will NOT cancel the job!
          </p>
        </div>

        <div className="flex items-center gap-4 w-full md:w-auto">
          <label
            htmlFor="video-upload-input"
            className="flex-1 md:flex-initial px-5 py-2.5 rounded-xl font-medium text-sm cursor-pointer border transition-all text-center"
            style={{
              background: 'rgba(0,212,255,0.08)',
              borderColor: 'rgba(0,212,255,0.3)',
              color: '#38bdf8',
            }}
          >
            {selectedFile ? 'Choose Different Video' : 'Select Video File'}
            <input
              id="video-upload-input"
              type="file"
              accept="video/mp4,video/avi,video/quicktime,video/webm"
              onChange={handleFileChange}
              className="hidden"
            />
          </label>

          <button
            onClick={handleAnalyze}
            disabled={!selectedFile || loading}
            className="px-6 py-2.5 rounded-xl font-semibold text-sm transition-all shadow-lg shadow-cyan-500/20 disabled:opacity-50 disabled:cursor-not-allowed"
            style={{
              background: 'linear-gradient(135deg, #00d4ff 0%, #0077ff 100%)',
              color: '#070d19',
            }}
          >
            {loading ? (
              <span className="flex items-center gap-2">
                <span className="w-4 h-4 border-2 border-slate-900 border-t-transparent rounded-full animate-spin" />
                Processing Video ({progress}%)...
              </span>
            ) : (
              'Analyze Video'
            )}
          </button>
        </div>
      </div>

      {/* Real Progress Bar */}
      {loading && (
        <div className="p-4 rounded-xl border bg-slate-900/90 border-cyan-500/30 space-y-2">
          <div className="flex justify-between text-xs font-mono text-cyan-300">
            <span>{progressText}</span>
            <span>{progress}%</span>
          </div>
          <div className="w-full bg-slate-950 rounded-full h-3 overflow-hidden border border-slate-800">
            <div
              className="bg-gradient-to-r from-cyan-400 to-blue-500 h-3 transition-all duration-300 rounded-full"
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="text-[11px] text-slate-400">
            ℹ️ You can freely switch tabs or check Dashboard / Session History while the background job processes.
          </p>
        </div>
      )}

      {/* Error Notice */}
      {errorMsg && (
        <div
          className="p-4 rounded-xl border flex items-center justify-between text-sm"
          style={{
            background: 'rgba(239,68,68,0.1)',
            borderColor: 'rgba(239,68,68,0.3)',
            color: '#f87171',
          }}
        >
          <span className="flex items-center gap-2">
            ⚠️ <strong>Notice:</strong> {errorMsg}
          </span>
          <button onClick={() => setErrorMsg(null)} className="text-xs opacity-80 hover:opacity-100">
            Dismiss
          </button>
        </div>
      )}

      {/* Main Drag/Drop or Analysis Content */}
      {!result && !previewUrl && !loading ? (
        <div
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
          className="flex-1 min-h-[360px] border-2 border-dashed rounded-2xl flex flex-col items-center justify-center p-8 text-center cursor-pointer transition-all"
          style={{
            borderColor: 'rgba(0,212,255,0.2)',
            background: 'rgba(15,23,42,0.3)',
          }}
        >
          <div className="w-16 h-16 rounded-full bg-cyan-500/10 flex items-center justify-center text-3xl mb-4 text-cyan-400">
            📽️
          </div>
          <h3 className="text-lg font-semibold text-white mb-1">Drag & Drop Video Here</h3>
          <p className="text-sm text-slate-400 mb-4 max-w-sm">
            Supports MP4, AVI, MOV, or WEBM format video files.
          </p>
          <label
            htmlFor="video-upload-input-drop"
            className="px-4 py-2 rounded-xl text-xs font-semibold bg-slate-800 text-slate-200 hover:bg-slate-700 cursor-pointer"
          >
            Browse Computer
            <input
              id="video-upload-input-drop"
              type="file"
              accept="video/mp4,video/avi,video/quicktime,video/webm"
              onChange={handleFileChange}
              className="hidden"
            />
          </label>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column: Video Player & Timeline */}
          <div className="lg:col-span-2 space-y-6">
            <div
              className="p-4 rounded-2xl border bg-slate-950/60 flex flex-col items-center justify-center relative overflow-hidden"
              style={{ borderColor: 'rgba(255,255,255,0.08)' }}
            >
              <div className="text-xs font-mono text-slate-400 mb-3 w-full flex items-center justify-between border-b pb-2 border-slate-800">
                <span>FILENAME: {result?.filename || selectedFile?.name}</span>
                {result && <span className="text-cyan-400 font-semibold">ANALYZED FRAMES: {result.total_frames_analyzed}</span>}
              </div>

              {previewUrl && (
                <video
                  src={previewUrl}
                  controls
                  className="max-h-[460px] w-full object-contain rounded-lg shadow-2xl border border-slate-800"
                />
              )}
            </div>

            {/* Video Emotion Timeline Chart */}
            {chartTimeline.length > 0 && (
              <div
                className="p-5 rounded-2xl border space-y-3"
                style={{
                  background: 'rgba(15,23,42,0.6)',
                  borderColor: 'rgba(0,212,255,0.15)',
                }}
              >
                <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
                  Timeline Detection Volume & Dominant Emotion
                </h3>
                <div className="h-48 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={chartTimeline}>
                      <defs>
                        <linearGradient id="facesGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#00d4ff" stopOpacity={0.4} />
                          <stop offset="95%" stopColor="#00d4ff" stopOpacity={0.0} />
                        </linearGradient>
                      </defs>
                      <XAxis dataKey="time" stroke="#64748b" fontSize={11} />
                      <YAxis stroke="#64748b" fontSize={11} />
                      <Tooltip
                        contentStyle={{
                          background: '#09101d',
                          borderColor: 'rgba(0,212,255,0.2)',
                          borderRadius: '8px',
                          color: '#fff',
                        }}
                      />
                      <Area type="monotone" dataKey="faces" stroke="#00d4ff" fillOpacity={1} fill="url(#facesGrad)" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}
          </div>

          {/* Right Column: Video Analytics */}
          <div className="space-y-6 flex flex-col">
            {result && result.success ? (
              <>
                <div
                  className="p-5 rounded-2xl border space-y-4"
                  style={{
                    background: 'rgba(15,23,42,0.6)',
                    borderColor: 'rgba(0,212,255,0.15)',
                  }}
                >
                  <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">
                    Video Analysis Metrics
                  </h3>

                  <div className="grid grid-cols-2 gap-3">
                    <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800">
                      <div className="text-xs text-slate-400">Duration</div>
                      <div className="text-xl font-bold text-cyan-400">{result.video_duration_seconds}s</div>
                    </div>
                    <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800">
                      <div className="text-xs text-slate-400">Total Detections</div>
                      <div className="text-xl font-bold text-purple-400">{result.total_face_detections}</div>
                    </div>
                    <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800">
                      <div className="text-xs text-slate-400">Frames Analyzed</div>
                      <div className="text-xl font-bold text-slate-200">{result.total_frames_analyzed}</div>
                    </div>
                    <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800">
                      <div className="text-xs text-slate-400">Avg Confidence</div>
                      <div className="text-xl font-bold text-emerald-400">
                        {((result.average_confidence || 0) * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>

                  <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800 flex items-center justify-between">
                    <div>
                      <div className="text-xs text-slate-400">Overall Dominant Emotion</div>
                      <div className="text-lg font-bold text-white flex items-center gap-1.5 mt-0.5">
                        <span>{EMOTION_ICONS[result.dominant_emotion || 'Neutral']}</span>
                        {result.dominant_emotion}
                      </div>
                    </div>
                    <div
                      className="w-3 h-3 rounded-full"
                      style={{
                        background: EMOTION_COLORS[result.dominant_emotion || 'Neutral'] || '#38bdf8',
                      }}
                    />
                  </div>
                </div>

                {/* Donut Chart */}
                {donutData.length > 0 && (
                  <div
                    className="p-5 rounded-2xl border flex flex-col items-center"
                    style={{
                      background: 'rgba(15,23,42,0.6)',
                      borderColor: 'rgba(0,212,255,0.15)',
                    }}
                  >
                    <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wider w-full mb-2">
                      Overall Expression Share
                    </h3>
                    <div className="h-44 w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                          <Pie
                            data={donutData}
                            cx="50%"
                            cy="50%"
                            innerRadius={45}
                            outerRadius={65}
                            paddingAngle={3}
                            dataKey="value"
                          >
                            {donutData.map((entry) => (
                              <Cell key={entry.name} fill={EMOTION_COLORS[entry.name] || '#38bdf8'} />
                            ))}
                          </Pie>
                          <Tooltip
                            contentStyle={{
                              background: '#09101d',
                              borderColor: 'rgba(0,212,255,0.2)',
                              borderRadius: '8px',
                              color: '#fff',
                            }}
                          />
                        </PieChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div
                className="p-6 rounded-2xl border flex flex-col items-center justify-center text-center text-slate-400 flex-1 min-h-[240px]"
                style={{
                  background: 'rgba(15,23,42,0.4)',
                  borderColor: 'rgba(255,255,255,0.08)',
                }}
              >
                <div className="text-3xl mb-2">🎬</div>
                <p className="text-sm font-medium">{loading ? 'Processing Background Video Job...' : 'Ready for Video Analysis'}</p>
                <p className="text-xs text-slate-500 max-w-xs mt-1">
                  {loading
                    ? 'Feel free to navigate to Dashboard or Session History — progress is tracked automatically.'
                    : 'Click "Analyze Video" to process video frames asynchronously with EfficientFace batching.'}
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
