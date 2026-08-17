import React, { useState } from 'react'
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts'
import { EMOTION_COLORS, EMOTION_ICONS } from '../data'

interface ImageDetection {
  face_index: number
  bbox: [number, number, number, number]
  emotion: string
  confidence: number
}

interface ImageAnalysisResult {
  success: boolean
  message?: string
  analysis_id?: string
  filename?: string
  total_faces: number
  dominant_emotion?: string
  average_confidence?: number
  emotion_distribution?: Record<string, { count: number; percentage: number }>
  annotated_image_base64?: string
  detections?: ImageDetection[]
}

export default function ImageUploadAnalysis() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<ImageAnalysisResult | null>(null)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

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
    setErrorMsg(null)

    try {
      const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'
      const formData = new FormData()
      formData.append('file', selectedFile)

      const response = await fetch(`${API_BASE_URL}/api/analyze/image`, {
        method: 'POST',
        body: formData,
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Image analysis failed')
      }

      setResult(data)
      if (!data.success && data.message) {
        setErrorMsg(data.message)
      }
    } catch (err: any) {
      setErrorMsg(err.message || 'Error analyzing image')
    } finally {
      setLoading(false)
    }
  }

  const donutData = result?.emotion_distribution
    ? Object.entries(result.emotion_distribution).map(([name, stat]) => ({
        name,
        value: stat.count,
      }))
    : []

  return (
    <div className="flex-1 flex flex-col overflow-y-auto p-6 space-y-6 animate-fade-in" style={{ background: '#070d19', color: '#f8fafc' }}>
      {/* Upload Header / Control Card */}
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
            <span className="p-2 rounded-lg bg-cyan-500/10 text-cyan-400">📷</span>
            Image Expression Recognition
          </h2>
          <p className="text-sm text-slate-400">
            Upload an image (JPG, PNG, WEBP) to detect facial bounding boxes and classify expressions using EfficientFace.
          </p>
        </div>

        <div className="flex items-center gap-4 w-full md:w-auto">
          <label
            htmlFor="image-upload-input"
            className="flex-1 md:flex-initial px-5 py-2.5 rounded-xl font-medium text-sm cursor-pointer border transition-all text-center"
            style={{
              background: 'rgba(0,212,255,0.08)',
              borderColor: 'rgba(0,212,255,0.3)',
              color: '#38bdf8',
            }}
          >
            {selectedFile ? 'Choose Different Image' : 'Select Image File'}
            <input
              id="image-upload-input"
              type="file"
              accept="image/jpeg,image/png,image/webp"
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
                Analyzing...
              </span>
            ) : (
              'Analyze Image'
            )}
          </button>
        </div>
      </div>

      {/* Error Message Notice */}
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

      {/* Main Content Layout */}
      {!result && !previewUrl ? (
        <div
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
          className="flex-1 min-h-[360px] border-2 border-dashed rounded-2xl flex flex-col items-center justify-center p-8 text-center transition-all cursor-pointer"
          style={{
            borderColor: 'rgba(0,212,255,0.2)',
            background: 'rgba(15,23,42,0.3)',
          }}
        >
          <div className="w-16 h-16 rounded-full bg-cyan-500/10 flex items-center justify-center text-3xl mb-4 text-cyan-400">
            📥
          </div>
          <h3 className="text-lg font-semibold text-white mb-1">Drag & Drop Image Here</h3>
          <p className="text-sm text-slate-400 mb-4 max-w-sm">
            Supports JPG, JPEG, PNG, or WEBP images with multiple visible human faces.
          </p>
          <label
            htmlFor="image-upload-input-drop"
            className="px-4 py-2 rounded-xl text-xs font-semibold bg-slate-800 text-slate-200 hover:bg-slate-700 cursor-pointer"
          >
            Browse Computer
            <input
              id="image-upload-input-drop"
              type="file"
              accept="image/jpeg,image/png,image/webp"
              onChange={handleFileChange}
              className="hidden"
            />
          </label>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column: Image Viewer */}
          <div
            className="lg:col-span-2 p-4 rounded-2xl border flex flex-col items-center justify-center bg-slate-950/60 relative overflow-hidden"
            style={{ borderColor: 'rgba(255,255,255,0.08)' }}
          >
            <div className="text-xs font-mono text-slate-400 mb-3 w-full flex items-center justify-between border-b pb-2 border-slate-800">
              <span>FILENAME: {selectedFile?.name}</span>
              {result && <span className="text-cyan-400 font-semibold">DETECTIONS: {result.total_faces}</span>}
            </div>

            <img
              src={result?.annotated_image_base64 || previewUrl || ''}
              alt="Analysis Target"
              className="max-h-[520px] w-auto object-contain rounded-lg shadow-2xl border border-slate-800"
            />
          </div>

          {/* Right Column: Analytics & Face Cards */}
          <div className="space-y-6 flex flex-col">
            {result && result.success ? (
              <>
                {/* Summary Metrics */}
                <div
                  className="p-5 rounded-2xl border space-y-4"
                  style={{
                    background: 'rgba(15,23,42,0.6)',
                    borderColor: 'rgba(0,212,255,0.15)',
                  }}
                >
                  <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">
                    Image Analysis Summary
                  </h3>

                  <div className="grid grid-cols-2 gap-3">
                    <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800">
                      <div className="text-xs text-slate-400">Total Faces</div>
                      <div className="text-2xl font-bold text-cyan-400">{result.total_faces}</div>
                    </div>
                    <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800">
                      <div className="text-xs text-slate-400">Avg Confidence</div>
                      <div className="text-2xl font-bold text-emerald-400">
                        {((result.average_confidence || 0) * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>

                  <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800 flex items-center justify-between">
                    <div>
                      <div className="text-xs text-slate-400">Dominant Expression</div>
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

                {/* Emotion Breakdown Donut */}
                {donutData.length > 0 && (
                  <div
                    className="p-5 rounded-2xl border flex flex-col items-center"
                    style={{
                      background: 'rgba(15,23,42,0.6)',
                      borderColor: 'rgba(0,212,255,0.15)',
                    }}
                  >
                    <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wider w-full mb-2">
                      Expression Distribution
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

                {/* Per-Face Detections List */}
                <div
                  className="p-5 rounded-2xl border space-y-3 flex-1 overflow-y-auto max-h-[300px]"
                  style={{
                    background: 'rgba(15,23,42,0.6)',
                    borderColor: 'rgba(0,212,255,0.15)',
                  }}
                >
                  <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
                    Detected Faces ({result.detections?.length || 0})
                  </h3>
                  <div className="space-y-2">
                    {result.detections?.map((det) => (
                      <div
                        key={det.face_index}
                        className="p-3 rounded-xl bg-slate-900/90 border border-slate-800 flex items-center justify-between"
                      >
                        <div className="flex items-center gap-2.5">
                          <span className="w-6 h-6 rounded-full bg-slate-800 flex items-center justify-center text-xs font-bold text-cyan-400">
                            #{det.face_index}
                          </span>
                          <div>
                            <div className="text-sm font-semibold text-white flex items-center gap-1">
                              <span>{EMOTION_ICONS[det.emotion] || '😐'}</span>
                              {det.emotion}
                            </div>
                            <div className="text-[11px] font-mono text-slate-400">
                              Box: ({det.bbox[0]}, {det.bbox[1]}, {det.bbox[2]}×{det.bbox[3]})
                            </div>
                          </div>
                        </div>

                        <div
                          className="px-2.5 py-1 rounded-lg text-xs font-bold font-mono"
                          style={{
                            background: `${EMOTION_COLORS[det.emotion]}20`,
                            color: EMOTION_COLORS[det.emotion] || '#38bdf8',
                            border: `1px solid ${EMOTION_COLORS[det.emotion]}40`,
                          }}
                        >
                          {(det.confidence * 100).toFixed(0)}%
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            ) : (
              <div
                className="p-6 rounded-2xl border flex flex-col items-center justify-center text-center text-slate-400 flex-1 min-h-[240px]"
                style={{
                  background: 'rgba(15,23,42,0.4)',
                  borderColor: 'rgba(255,255,255,0.08)',
                }}
              >
                <div className="text-3xl mb-2">🔍</div>
                <p className="text-sm font-medium">Ready for Analysis</p>
                <p className="text-xs text-slate-500 max-w-xs mt-1">
                  Click "Analyze Image" above to run SCRFD face detection & EfficientFace emotion classification.
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
