import { useState } from 'react'
import MonitorPage from './App'
import RunsPage from './RunsPage'
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
        </div>
      </nav>
      {page === 'runs' ? <RunsPage /> : <MonitorPage />}
    </>
  )
}

export default Root
