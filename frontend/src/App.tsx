import { useState } from 'react'
import type { Screen } from './types'
import Sidebar from './components/Sidebar'
import Dashboard from './screens/Dashboard'
import LiveDetection from './screens/LiveDetection'
import Analytics from './screens/Analytics'
import SessionHistory from './screens/SessionHistory'

export default function App() {
  const [screen, setScreen] = useState<Screen>('dashboard')
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null)

  const handleNavigate = (targetScreen: Screen, sessionId?: string) => {
    if (sessionId) {
      setSelectedSessionId(sessionId)
    }
    setScreen(targetScreen)
  }

  return (
    <div className="flex h-screen overflow-hidden" style={{ background: '#080c14' }}>
      <Sidebar active={screen} onNavigate={(s) => handleNavigate(s)} />
      <main className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {screen === 'dashboard' && <Dashboard onNavigate={(s) => handleNavigate(s)} />}
        {screen === 'live' && <LiveDetection />}
        {screen === 'analytics' && <Analytics initialSessionId={selectedSessionId} />}
        {screen === 'history' && <SessionHistory onNavigate={handleNavigate} />}
      </main>
    </div>
  )
}
