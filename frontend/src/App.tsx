import { useState } from 'react'
import type { Screen } from './types'
import Sidebar from './components/Sidebar'
import Dashboard from './screens/Dashboard'
import LiveDetection from './screens/LiveDetection'
import Analytics from './screens/Analytics'
import SessionHistory from './screens/SessionHistory'

export default function App() {
  const [screen, setScreen] = useState<Screen>('dashboard')

  return (
    <div className="flex h-screen overflow-hidden" style={{ background: '#080c14' }}>
      <Sidebar active={screen} onNavigate={setScreen} />
      <main className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {screen === 'dashboard' && <Dashboard onNavigate={setScreen} />}
        {screen === 'live' && <LiveDetection />}
        {screen === 'analytics' && <Analytics />}
        {screen === 'history' && <SessionHistory />}
      </main>
    </div>
  )
}
