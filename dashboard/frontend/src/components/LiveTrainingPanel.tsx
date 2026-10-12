import { useEffect, useMemo, useState } from 'react'
import { Activity, Radio } from 'lucide-react'
import { api, type LivePpoFrame, type LivePpoEnvironment } from '../api'

const fmt = (n: number | null | undefined, d = 2) => typeof n === 'number' && Number.isFinite(n) ? n.toFixed(d) : '—'

function WorldXY({ robot, trail }: { robot: LivePpoEnvironment; trail: LivePpoEnvironment[] }) {
  const xy = [...trail.map((r) => r.position_xyz.slice(0, 2)), ...(robot.target_xy ? [robot.target_xy] : [])]
  const xx = xy.map((a) => a[0]), yy = xy.map((a) => a[1])
  const mx = (Math.min(...xx) + Math.max(...xx)) / 2, my = (Math.min(...yy) + Math.max(...yy)) / 2
  const scale = Math.min(540 / Math.max(2.5, Math.max(...xx) - Math.min(...xx) + .7), 270 / Math.max(2, Math.max(...yy) - Math.min(...yy) + .7))
  const p = (x: number, y: number): [number, number] => [300 + (x - mx) * scale, 170 - (y - my) * scale]
  const [x, y] = p(robot.position_xyz[0], robot.position_xyz[1])
  const route = trail.map((r) => p(r.position_xyz[0], r.position_xyz[1]).join(',')).join(' ')
  const target = robot.target_xy ? p(robot.target_xy[0], robot.target_xy[1]) : null
  const actual = p(robot.position_xyz[0] + robot.velocity_xy[0], robot.position_xyz[1] + robot.velocity_xy[1])
  const dx = robot.target_xy ? robot.target_xy[0] - robot.position_xyz[0] : 0
  const dy = robot.target_xy ? robot.target_xy[1] - robot.position_xyz[1] : 0
  const dist = Math.hypot(dx, dy), requested = robot.speed_command_mps
  const commanded = dist > 1e-5 && requested !== null ? p(robot.position_xyz[0] + dx / dist * requested, robot.position_xyz[1] + dy / dist * requested) : null
  const [w, qx, qy, qz] = robot.quaternion_wxyz
  const heading = -Math.atan2(2 * (w*qz + qx*qy), 1 - 2 * (qy*qy + qz*qz)) * 180 / Math.PI
  return <svg viewBox="0 0 600 340" className="w-full rounded-lg border border-border bg-[#101820]" role="img" aria-label="Top-down training trajectory and velocity vectors">
    <defs>
      <pattern id="ppo-grid" width="40" height="40" patternUnits="userSpaceOnUse"><path d="M40 0H0V40" fill="none" stroke="#3b4c57" strokeWidth=".8" /></pattern>
      <marker id="ppo-green-arrow" markerWidth="6" markerHeight="6" refX="4" refY="3" orient="auto"><path d="M0 0L6 3L0 6" fill="none" stroke="#5bdba8" /></marker>
      <marker id="ppo-gold-arrow" markerWidth="6" markerHeight="6" refX="4" refY="3" orient="auto"><path d="M0 0L6 3L0 6" fill="none" stroke="#efbc6c" /></marker>
    </defs>
    <rect width="600" height="340" fill="url(#ppo-grid)" />
    <text x="14" y="23" fill="#b3c0cb" fontSize="12">World XY · metres</text>
    {trail.length > 1 && <polyline points={route} fill="none" stroke="#70b8e6" strokeWidth="3" />}
    {target && <><line x1={x} y1={y} x2={target[0]} y2={target[1]} stroke="#efbc6c" strokeDasharray="6 7" opacity=".5" />
      <circle cx={target[0]} cy={target[1]} r="10" fill="none" stroke="#efbc6c" strokeWidth="2" /></>}
    <line x1={x} y1={y} x2={actual[0]} y2={actual[1]} stroke="#5bdba8" strokeWidth="3" markerEnd="url(#ppo-green-arrow)" />
    {commanded && <line x1={x} y1={y} x2={commanded[0]} y2={commanded[1]} stroke="#efbc6c" strokeWidth="2" markerEnd="url(#ppo-gold-arrow)" />}
    <polygon points={(x+12)+','+y+' '+(x-10)+','+(y-8)+' '+(x-10)+','+(y+8)} transform={'rotate('+heading+' '+x+' '+y+')'} fill="#74c4eb" stroke="#e0f2fc" strokeWidth="2" />
    <text x="14" y="316" fill="#aebdc7" fontSize="11">Blue: path · Green: actual speed · Gold: requested speed / goal</text>
  </svg>
}

function MiniTrend({ label, values, color }: { label: string; values: number[]; color: string }) {
  const lo = Math.min(0, ...values), span = Math.max(1e-6, Math.max(0, ...values)-lo)
  const points = values.map((n, i) => (i * 250 / Math.max(1, values.length-1))+','+(49-42*(n-lo)/span)).join(' ')
  return <div className="min-w-0 rounded-lg bg-raised p-3"><p className="text-xs text-muted">{label}</p>
    <svg viewBox="0 0 250 55" preserveAspectRatio="none" className="h-14 w-full"><polyline points={points} stroke={color} strokeWidth="2" fill="none" /></svg></div>
}

export function LiveTrainingPanel({ runId }: { runId: string }) {
  const [frame, setFrame] = useState<LivePpoFrame | null>(null)
  const [frames, setFrames] = useState<LivePpoFrame[]>([])
  const [envId, setEnvId] = useState<number | null>(null)
  const [connected, setConnected] = useState(false)
  const [, setHeartbeat] = useState(0)
  useEffect(() => {
    let active = true
    setFrame(null); setFrames([]); setEnvId(null); setConnected(false)
    void api.livePpoLatest(runId).then((value) => { if (active && value.available) { setFrame(value); setFrames([value]) } }).catch(() => {})
    const source = new EventSource(api.livePpoStreamUrl(runId))
    source.addEventListener('frame', (event) => {
      if (!active) return
      try {
        const value = JSON.parse((event as MessageEvent<string>).data) as LivePpoFrame
        if (!value.available || value.source !== 'actual_ppo_rollout') return
        setFrame(value); setConnected(true)
        setFrames((old) => old.at(-1)?.sequence === value.sequence && old.at(-1)?.sampled_at === value.sampled_at ? old : [...old.slice(-119), value])
      } catch { setConnected(false) }
    })
    source.onerror = () => { if (active) setConnected(false) }
    const timer = window.setInterval(() => setHeartbeat((n) => n+1), 1500)
    return () => { active = false; source.close(); window.clearInterval(timer) }
  }, [runId])
  const envs = frame?.environments ?? []
  const selectedId = envs.some((e) => e.environment_id === envId) ? envId : envs[0]?.environment_id
  const current = envs.find((e) => e.environment_id === selectedId)
  const trail = useMemo(() => frames.map((f) => f.environments.find((e) => e.environment_id === selectedId))
    .filter((e): e is LivePpoEnvironment => e !== undefined).reduce<LivePpoEnvironment[]>((out, item) => {
      if (item.reset || (out.length && item.episode_step < out[out.length-1].episode_step)) return [item]
      return [...out, item]
    }, []), [frames, selectedId])
  const stale = !frame || Date.now()/1000-frame.sampled_at>10
  const collapse = current && (current.tilt_deg > 45 || current.root_height_m < 0.35)
  const rates = current ? Object.entries(current.reward_terms_rate).sort((a,b) => Math.abs(b[1])-Math.abs(a[1])).slice(0,6) : []
  const trend = (field: 'speed_mps'|'tilt_deg'|'step_reward') => trail.map((f) => f[field])
  return <section className="rounded-xl border border-border bg-panel p-4 sm:p-6" aria-label="Live training monitor">
    <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
      <div><p className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-muted"><Radio size={14} /> Actual PPO rollout</p>
        <h2 className="mt-1 text-xl font-semibold">Live spatial monitor</h2>
        <p className="mt-1 text-xs text-muted">Sampled training state · 2D only · no live renderer</p></div>
      <div className="flex items-center gap-2 text-xs text-muted"><Activity size={15} className={connected && !stale ? 'text-emerald-400' : ''} />
        {stale ? 'No current telemetry' : connected ? 'Live · '+fmt(Date.now()/1000-(frame?.sampled_at ?? 0),1)+'s ago' : 'Reconnecting'}</div>
    </div>
    {!current ? <div className="rounded-lg border border-dashed border-border p-6 text-center text-sm text-muted">No PPO telemetry available for this run yet. Existing runs require restarting training with the monitor-enabled code.</div> : <>
      <div className="mb-3 flex flex-wrap justify-between gap-3 text-xs text-muted"><label>Environment
        <select className="ml-2 rounded-lg border border-border bg-background p-2 text-foreground" value={selectedId ?? ''} onChange={(e)=>setEnvId(Number(e.target.value))}>
          {envs.map((e)=><option key={e.environment_id} value={e.environment_id}>#{e.environment_id}</option>)}</select></label>
        <span>Step {frame?.global_step.toLocaleString()} · episode {current.episode_step} · sample cost {fmt(frame?.capture_overhead_ms,1)} ms</span></div>
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.5fr)_minmax(250px,1fr)]">
        <WorldXY robot={current} trail={trail} />
        <div className="grid grid-cols-2 gap-2 self-start">
          {([
            ['Actual speed',fmt(current.speed_mps)+' m/s'],['Target speed',fmt(current.speed_command_mps)+' m/s'],
            ['Closing speed',fmt(current.closing_speed_mps)+' m/s'],['Goal distance',fmt(current.target_distance_m)+' m'],
            ['Body tilt',fmt(current.tilt_deg,0)+'°'],['Root height',fmt(current.root_height_m)+' m'],
            ['Step reward',fmt(current.step_reward,4)],['Contacts',[current.wheel_contacts.left,current.wheel_contacts.right].map((x)=>x===null?'?':x?'Yes':'No').join(' / ')],
          ] as const).map(([name,value])=><div key={name} className="rounded-lg bg-raised p-3"><p className="text-[11px] text-muted">{name}</p><p className="mt-1 text-base font-semibold tabular-nums">{value}</p></div>)}
          <div className={'col-span-2 rounded-lg border p-3 text-sm '+(collapse?'border-orange-500/40 bg-orange-500/10 text-orange-200':'border-border bg-raised text-secondary')}>
            {collapse?'Possible collapse / loss of balance':'No gross collapse detected'}
            <p className="mt-1 text-xs opacity-70">{current.reset?'Episode ended and was automatically reset.':'Based on body tilt and root height.'}</p></div>
        </div>
      </div>
      <div className="mt-4 grid gap-3 sm:grid-cols-3">
        <MiniTrend label="Actual speed (m/s)" values={trend('speed_mps')} color="#58d4a6" />
        <MiniTrend label="Body tilt (degrees)" values={trend('tilt_deg')} color="#efbc6c" />
        <MiniTrend label="Step reward" values={trend('step_reward')} color="#65b8e9" />
      </div>
      {!!rates.length && <div className="mt-4 border-t border-border pt-3"><h3 className="text-sm font-semibold">Dominant weighted reward rates</h3>
        <div className="mt-2 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">{rates.map(([name,value])=><div key={name} className="flex min-w-0 justify-between gap-2 rounded bg-raised px-3 py-2 text-xs"><span className="truncate" title={name}>{name}</span><span className={value<0?'text-orange-300':'text-emerald-300'}>{fmt(value,3)}</span></div>)}</div></div>}
      <p className="mt-3 text-xs text-muted">One sampled environment is not representative of all PPO environments. Reward terms are weighted rates before dt scaling.{frame?.dropped_display_frames?' Dropped display frames: '+frame.dropped_display_frames:''}</p>
    </>}
  </section>
}
