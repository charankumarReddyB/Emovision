import type { Screen } from '../types'

interface Props {
  active: Screen
  onNavigate: (s: Screen) => void
}

const items: { id: Screen; label: string; icon: React.ReactNode }[] = [
  {
    id: 'dashboard',
    label: 'Dashboard',
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
        <rect x="3" y="3" width="7" height="7" rx="1" /><rect x="14" y="3" width="7" height="7" rx="1" />
        <rect x="14" y="14" width="7" height="7" rx="1" /><rect x="3" y="14" width="7" height="7" rx="1" />
      </svg>
    ),
  },
  {
    id: 'live',
    label: 'Live Detection',
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
        <path d="M15 10l4.553-2.069A1 1 0 0121 8.845v6.31a1 1 0 01-1.447.894L15 14" />
        <rect x="3" y="6" width="12" height="12" rx="2" />
      </svg>
    ),
  },
  {
    id: 'analytics',
    label: 'Analytics',
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
      </svg>
    ),
  },
  {
    id: 'history',
    label: 'Session History',
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="10" /><polyline points="12 6 12 12 16 14" />
      </svg>
    ),
  },
]

export default function Sidebar({ active, onNavigate }: Props) {
  return (
    <aside className="flex flex-col w-60 shrink-0 h-full border-r" style={{ background: '#09101d', borderColor: 'rgba(0,212,255,0.08)' }}>
      {/* Logo */}
      <div className="px-5 pt-6 pb-5 border-b" style={{ borderColor: 'rgba(0,212,255,0.08)' }}>
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: 'linear-gradient(135deg,#00d4ff22,#7c3aed33)', border: '1px solid rgba(0,212,255,0.3)' }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#00d4ff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="8" r="3" /><path d="M8 21v-1a4 4 0 018 0v1" />
              <path d="M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z" strokeOpacity="0" />
              <circle cx="12" cy="8" r="6" strokeDasharray="2 2" strokeOpacity="0.4" />
            </svg>
          </div>
          <div>
            <div className="text-xs font-bold tracking-widest uppercase" style={{ color: '#00d4ff' }}>EmoVision</div>
            <div className="text-xs" style={{ color: '#475569', fontSize: '10px' }}>CV System</div>
          </div>
        </div>
      </div>

      {/* Status badge */}
      <div className="mx-4 mt-4 mb-2 px-3 py-2 rounded-lg flex items-center gap-2" style={{ background: 'rgba(34,197,94,0.08)', border: '1px solid rgba(34,197,94,0.15)' }}>
        <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse-dot" />
        <span className="text-xs font-medium" style={{ color: '#4ade80' }}>System Online</span>
      </div>

      {/* Nav items */}
      <nav className="flex-1 px-3 py-3 flex flex-col gap-1">
        {items.map(item => {
          const isActive = active === item.id
          return (
            <button
              key={item.id}
              onClick={() => onNavigate(item.id)}
              className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-left transition-all duration-150 group w-full"
              style={{
                background: isActive ? 'linear-gradient(90deg,rgba(0,212,255,0.12),rgba(0,212,255,0.04))' : 'transparent',
                color: isActive ? '#00d4ff' : '#94a3b8',
                border: isActive ? '1px solid rgba(0,212,255,0.18)' : '1px solid transparent',
              }}
            >
              <span style={{ color: isActive ? '#00d4ff' : '#64748b' }}>{item.icon}</span>
              <span className="text-sm font-medium">{item.label}</span>
              {isActive && (
                <span className="ml-auto w-1 h-4 rounded-full" style={{ background: '#00d4ff' }} />
              )}
            </button>
          )
        })}
      </nav>

      {/* Footer */}
      <div className="px-5 py-4 border-t" style={{ borderColor: 'rgba(0,212,255,0.08)' }}>
        <div className="text-xs" style={{ color: '#334155', lineHeight: 1.5 }}>
          <div className="font-semibold" style={{ color: '#475569' }}>Computer Vision</div>
          <div>Capstone Project</div>
        </div>
      </div>
    </aside>
  )
}
