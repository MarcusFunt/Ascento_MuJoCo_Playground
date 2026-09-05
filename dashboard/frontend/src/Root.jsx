import { useState } from 'react'
import MonitorPage from './App'
import RunsPage from './RunsPage'
import SystemPage from './SystemPage'
import './runs.css'

function Root() {
  const [page, setPage] = useState('runs')

  return (
    <>
      <nav className="dashboard-nav">
        <div className="dashboard-nav-brand">ASCENTO CONTROL</div>
        <div className="dashboard-nav-tabs">
          <button className={page === 'runs' ? 'active' : ''} onClick={() => setPage('runs')}>Runs</button>
          <button className={page === 'monitor' ? 'active' : ''} onClick={() => setPage('monitor')}>Monitor</button>
          <button className={page === 'system' ? 'active' : ''} onClick={() => setPage('system')}>System</button>
        </div>
      </nav>
      {page === 'runs' && <RunsPage />}
      {page === 'monitor' && <MonitorPage />}
      {page === 'system' && <SystemPage />}
    </>
  )
}

export default Root
