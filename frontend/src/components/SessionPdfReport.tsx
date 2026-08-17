import React from 'react'
import { Printer, Download, User, Activity, Clock, Users, ShieldCheck, PieChart } from 'lucide-react'
import { SessionAnalyticsData, PersonDetailCard } from '../types'

interface SessionPdfReportProps {
  analytics: SessionAnalyticsData
  sessionName?: string
  date?: string
  onClose?: () => void
}

export const SessionPdfReport: React.FC<SessionPdfReportProps> = ({
  analytics,
  sessionName = 'Live Camera Session',
  date = new Date().toLocaleDateString(),
  onClose
}) => {
  const handlePrint = () => {
    window.print()
  }

  const emotionColors: Record<string, string> = {
    Happy: '#22c55e',
    Neutral: '#3b82f6',
    Surprise: '#eab308',
    Sad: '#a855f7',
    Angry: '#ef4444',
    Fear: '#f97316',
    Disgust: '#14b8a6',
    Uncertain: '#6b7280'
  }

  const totalDetections = analytics.total_predictions || 
    Object.values(analytics.expression_distribution || {}).reduce((a, b) => a + b, 0) || 1

  return (
    <div className="bg-slate-900 text-white min-h-screen h-screen overflow-y-auto p-6 sm:p-10 font-sans print:bg-white print:text-black print:p-0 print:h-auto print:overflow-visible">
      {/* Print Action Bar (Hidden when printing) */}
      <div className="max-w-4xl mx-auto mb-6 flex items-center justify-between bg-slate-800/90 border border-slate-700 p-4 rounded-xl print:hidden">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-cyan-500/20 text-cyan-400 rounded-lg">
            <Printer size={22} />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-white">Session PDF Report Preview</h3>
            <p className="text-sm text-slate-400">Click below to print or save this report as a PDF document</p>
          </div>
        </div>
        <div className="flex items-center space-x-3">
          {onClose && (
            <button
              onClick={onClose}
              className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-slate-200 text-sm font-medium rounded-lg transition"
            >
              Back to Dashboard
            </button>
          )}
          <button
            onClick={handlePrint}
            className="flex items-center space-x-2 px-5 py-2.5 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white text-sm font-semibold rounded-lg shadow-lg shadow-cyan-500/25 transition transform active:scale-95"
            title="Click to download or save this report as PDF"
          >
            <Download size={18} />
            <span>Download PDF Report</span>
          </button>
        </div>
      </div>

      {/* Printable Report Container */}
      <div className="max-w-4xl mx-auto bg-slate-950 border border-slate-800 rounded-2xl p-8 sm:p-10 shadow-2xl print:shadow-none print:border-none print:bg-white print:text-black print:max-w-none">
        {/* Report Header */}
        <div className="border-b border-slate-800 print:border-slate-300 pb-6 mb-8 flex items-start justify-between">
          <div>
            <div className="flex items-center space-x-2 text-cyan-400 print:text-blue-600 text-xs font-bold uppercase tracking-widest mb-1">
              <ShieldCheck size={16} />
              <span>Emovision Computer Vision Platform</span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-extrabold text-white print:text-black tracking-tight">
              Facial Expression Analytics Report
            </h1>
            <p className="text-slate-400 print:text-slate-600 text-sm mt-1">
              Session: <span className="text-cyan-300 print:text-blue-600 font-semibold">{sessionName}</span> ({analytics.session_id})
            </p>
          </div>
          <div className="text-right">
            <span className="inline-block px-3 py-1 bg-cyan-500/10 print:bg-blue-100 text-cyan-400 print:text-blue-700 text-xs font-semibold rounded-full border border-cyan-500/20 print:border-blue-300 mb-2">
              Generated: {date}
            </span>
            <p className="text-xs text-slate-500 print:text-slate-400">Duration: {analytics.session_duration_seconds}s</p>
          </div>
        </div>

        {/* Executive Summary Metrics Grid */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-8">
          <div className="bg-slate-900/80 print:bg-slate-50 border border-slate-800 print:border-slate-200 p-4 rounded-xl">
            <div className="flex items-center space-x-2 text-cyan-400 print:text-blue-600 text-xs font-semibold mb-1">
              <Users size={16} />
              <span>People Detected</span>
            </div>
            <div className="text-2xl font-black text-white print:text-black">{analytics.total_people_detected}</div>
          </div>

          <div className="bg-slate-900/80 print:bg-slate-50 border border-slate-800 print:border-slate-200 p-4 rounded-xl">
            <div className="flex items-center space-x-2 text-emerald-400 print:text-emerald-600 text-xs font-semibold mb-1">
              <Activity size={16} />
              <span>Dominant Emotion</span>
            </div>
            <div className="text-xl font-bold text-emerald-400 print:text-emerald-700">
              {analytics.dominant_expression}
            </div>
          </div>

          <div className="bg-slate-900/80 print:bg-slate-50 border border-slate-800 print:border-slate-200 p-4 rounded-xl">
            <div className="flex items-center space-x-2 text-purple-400 print:text-purple-600 text-xs font-semibold mb-1">
              <Clock size={16} />
              <span>Avg Confidence</span>
            </div>
            <div className="text-2xl font-black text-purple-300 print:text-purple-700">
              {analytics.average_confidence}%
            </div>
          </div>

          <div className="bg-slate-900/80 print:bg-slate-50 border border-slate-800 print:border-slate-200 p-4 rounded-xl">
            <div className="flex items-center space-x-2 text-amber-400 print:text-amber-600 text-xs font-semibold mb-1">
              <PieChart size={16} />
              <span>Total Predictions</span>
            </div>
            <div className="text-2xl font-black text-amber-300 print:text-amber-700">
              {analytics.total_predictions}
            </div>
          </div>
        </div>

        {/* Section 1: Person Identification & Facial Photo Gallery */}
        <div className="mb-10">
          <h2 className="text-lg font-bold text-white print:text-black border-b border-slate-800 print:border-slate-300 pb-2 mb-4 flex items-center space-x-2">
            <User size={20} className="text-cyan-400 print:text-blue-600" />
            <span>Person Identification Cards ({analytics.persons_details?.length || analytics.total_people_detected})</span>
          </h2>

          {analytics.persons_details && analytics.persons_details.length > 0 ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {analytics.persons_details.map((person: PersonDetailCard) => (
                <div
                  key={person.person_id}
                  className="bg-slate-900/90 print:bg-slate-50 border border-slate-800 print:border-slate-200 p-4 rounded-xl flex items-center space-x-4"
                >
                  {/* Face Photo Thumbnail */}
                  <div className="relative flex-shrink-0 w-20 h-20 bg-slate-800 print:bg-slate-200 rounded-lg overflow-hidden border border-cyan-500/30 print:border-blue-400 flex items-center justify-center">
                    {person.thumbnail_b64 ? (
                      <img
                        src={person.thumbnail_b64}
                        alt={`Person ${person.person_id} face photo`}
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <div className="flex flex-col items-center justify-center text-slate-400 print:text-slate-500">
                        <User size={28} />
                        <span className="text-[10px] mt-1">Photo</span>
                      </div>
                    )}
                  </div>

                  {/* Person Metadata Details */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-sm font-bold text-cyan-300 print:text-blue-700">
                        Person #{person.person_id}
                      </span>
                      <span
                        className="text-xs px-2 py-0.5 rounded font-semibold text-white print:text-black"
                        style={{ backgroundColor: emotionColors[person.dominant_emotion] || '#3b82f6' }}
                      >
                        {person.dominant_emotion}
                      </span>
                    </div>

                    <div className="text-xs text-slate-400 print:text-slate-600 space-y-1">
                      <div>
                        Confidence: <span className="font-semibold text-slate-200 print:text-slate-800">{person.average_confidence}%</span>
                      </div>
                      <div>
                        Frame Detections: <span className="font-semibold text-slate-200 print:text-slate-800">{person.total_detections}</span>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="p-4 bg-slate-900/50 print:bg-slate-100 rounded-xl text-center text-slate-400 print:text-slate-600 text-sm">
              No individual person face identification photos logged for this session.
            </div>
          )}
        </div>

        {/* Section 2: Emotion Distribution Breakdown Table */}
        <div className="mb-8">
          <h2 className="text-lg font-bold text-white print:text-black border-b border-slate-800 print:border-slate-300 pb-2 mb-4 flex items-center space-x-2">
            <PieChart size={20} className="text-emerald-400 print:text-emerald-600" />
            <span>Facial Emotion Class Breakdown</span>
          </h2>

          <div className="border border-slate-800 print:border-slate-200 rounded-xl overflow-hidden">
            <table className="w-full text-left text-sm">
              <thead className="bg-slate-900 print:bg-slate-100 text-slate-300 print:text-slate-700 font-semibold border-b border-slate-800 print:border-slate-200">
                <tr>
                  <th className="p-3">Expression Class</th>
                  <th className="p-3 text-center">Prediction Count</th>
                  <th className="p-3 text-center">Percentage Distribution</th>
                  <th className="p-3 text-right">Visual Bar</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800 print:divide-slate-200 text-slate-300 print:text-slate-800">
                {Object.entries(analytics.expression_distribution || {}).map(([emotion, count]) => {
                  const pct = ((count / totalDetections) * 100).toFixed(1)
                  const color = emotionColors[emotion] || '#3b82f6'
                  return (
                    <tr key={emotion} className="hover:bg-slate-900/50 print:hover:bg-slate-50">
                      <td className="p-3 font-medium flex items-center space-x-2">
                        <span className="w-3 h-3 rounded-full" style={{ backgroundColor: color }} />
                        <span>{emotion}</span>
                      </td>
                      <td className="p-3 text-center font-semibold">{count}</td>
                      <td className="p-3 text-center font-bold">{pct}%</td>
                      <td className="p-3 text-right">
                        <div className="w-24 sm:w-32 bg-slate-800 print:bg-slate-200 rounded-full h-2.5 ml-auto overflow-hidden">
                          <div
                            className="h-full rounded-full"
                            style={{ width: `${pct}%`, backgroundColor: color }}
                          />
                        </div>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Footer Statement */}
        <div className="border-t border-slate-800 print:border-slate-300 pt-4 text-center text-xs text-slate-500 print:text-slate-500">
          Official Computer Vision Analysis Report • Generated by Emovision Real-Time Facial Emotion Telemetry Engine
        </div>
      </div>
    </div>
  )
}
