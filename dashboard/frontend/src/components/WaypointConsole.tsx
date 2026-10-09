import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '../api'
import type { WaypointCommand, WaypointState } from '../types'
import { Button } from './ui/button'
import { Input } from './ui/input'

type Props = { viewerId: string; viewerState: string; task?: string }
type AvailableWaypointState = WaypointState & { available: true }

function isAvailable(value: WaypointState | { available: false; message: string } | undefined): value is AvailableWaypointState {
  return value?.available === true
}

export function WaypointConsole({ viewerId, viewerState, task }: Props) {
  const client = useQueryClient()
  const [x, setX] = useState('1.0')
  const [y, setY] = useState('0.0')
  const [yawDeg, setYawDeg] = useState('0')
  const [desiredSpeed, setDesiredSpeed] = useState('0')
  const [ack, setAck] = useState('')
  const waypoint = useQuery({
    queryKey: ['waypoint-state', viewerId],
    queryFn: () => api.waypointState(viewerId),
    enabled: Boolean(viewerId),
    refetchInterval: viewerState === 'running' ? 750 : false,
  })
  const state = isAvailable(waypoint.data) ? waypoint.data : null
  const command = useMutation({
    mutationFn: (payload: WaypointCommand) => api.commandWaypoint(viewerId, payload),
    onSuccess: async (result) => {
      setAck(`Command ${result.request_id} accepted by the viewer queue.`)
      await client.invalidateQueries({ queryKey: ['waypoint-state', viewerId] })
    },
  })

  useEffect(() => {
    if (state?.speed_command) setDesiredSpeed(String(state.speed_command.requested_mps))
  }, [state?.speed_command?.requested_mps])

  const validation = useMemo(() => {
    if (!state) return { ok: false, message: 'Waiting for a live waypoint state.' }
    const xValue = Number(x)
    const yValue = Number(y)
    const yawValue = Number(yawDeg)
    if (![xValue, yValue, yawValue].every(Number.isFinite)) return { ok: false, message: 'X, Y and heading must be finite numbers.' }
    const extent = state.limits.arena_half_extent_m
    if (Math.max(Math.abs(xValue - state.origin.x_m), Math.abs(yValue - state.origin.y_m)) > extent) {
      return { ok: false, message: `Target must stay inside the ±${extent.toFixed(1)} m arena.` }
    }
    const queuedLast = state.queue.at(-1)
    const segmentStart = queuedLast || (state.active && state.state !== 'arrived' ? state.active : state.robot)
    const segmentLength = Math.hypot(xValue - segmentStart.x_m, yValue - segmentStart.y_m)
    if (segmentLength > state.limits.max_segment_m + 1e-6) {
      return { ok: false, message: `Segment is ${segmentLength.toFixed(2)} m; limit is ${state.limits.max_segment_m.toFixed(1)} m.` }
    }
    return { ok: true, message: `${segmentLength.toFixed(2)} m segment · heading ${yawValue.toFixed(0)}°` }
  }, [state, x, y, yawDeg])

  const map = useMemo(() => {
    if (!state) return null
    const width = 480
    const height = 360
    const extent = state.limits.arena_half_extent_m
    const px = (value: number) => ((value - state.origin.x_m + extent) / (2 * extent)) * width
    const py = (value: number) => height - ((value - state.origin.y_m + extent) / (2 * extent)) * height
    return { width, height, extent, px, py }
  }, [state])

  const send = async (payload: WaypointCommand) => {
    setAck('Sending command…')
    await command.mutateAsync(payload)
  }

  const addWaypoint = async (operation: 'set' | 'queue') => {
    if (!state || !validation.ok) return
    if (operation === 'set' && (state.active || state.queue.length) && !window.confirm('Set this waypoint and clear the active route?')) return
    await send({ operation, x_m: Number(x), y_m: Number(y), yaw_rad: Number(yawDeg) * Math.PI / 180 })
  }

  const cancelRoute = async () => {
    if (!state || (!state.active && !state.queue.length)) return
    if (!window.confirm('Cancel the route and clear every queued waypoint?')) return
    await send({ operation: 'cancel' })
  }

  const speed = state?.speed_command
  const speedValue = Number(desiredSpeed)
  const speedValid = Boolean(speed && Number.isFinite(speedValue) && speedValue >= 0 && speedValue <= speed.max_speed_mps)

  return (
    <section className="rounded-xl border border-border bg-panel p-5 sm:p-6" aria-label="Waypoint and speed controls">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="text-xs font-bold uppercase tracking-[0.1em] text-muted">Live simulation controls</div>
          <h2 className="mt-1 text-xl font-semibold">World route and speed command</h2>
          <p className="mt-1 text-sm text-muted">Coordinates use absolute simulation-world metres. Heading is displayed in degrees and sent in radians.</p>
        </div>
        {state ? <span className="rounded border border-border-strong bg-raised px-2.5 py-1.5 text-xs font-bold uppercase text-secondary">Robot {state.state}</span> : null}
      </div>

      {waypoint.error ? <p className="mt-4 rounded-lg border border-danger/35 bg-danger/10 p-3 text-sm text-danger" role="alert">Waypoint telemetry failed: {waypoint.error.message}</p> : null}
      {waypoint.isLoading ? <p className="mt-4 text-sm text-muted">Connecting to viewer simulation state…</p> : null}
      {!state ? <p className="mt-4 rounded-lg border border-border bg-background/35 p-4 text-sm text-secondary" role="status">{waypoint.data && 'message' in waypoint.data ? waypoint.data.message : viewerState === 'running' ? 'Waypoint telemetry is not ready yet.' : 'Start the viewer to enable simulation controls.'}</p> : null}

      {state && map ? (
        <>
          <div className="mt-5 grid gap-5 xl:grid-cols-[minmax(0,1.2fr)_minmax(19rem,0.8fr)]">
            <div className="min-w-0 rounded-lg border border-border bg-background/35 p-3">
              <div className="mb-2 flex items-center justify-between gap-2 text-xs text-muted">
                <span>Top-down arena · {state.limits.arena_half_extent_m.toFixed(1)} m half-extent</span>
                <span>Frame: sim_world</span>
              </div>
              <svg viewBox={`0 0 ${map.width} ${map.height}`} className="w-full rounded-md bg-[#0b1116]" role="img" aria-label="Top-down world map showing the robot, active target and queued route">
                {[0, 1, 2, 3, 4].map((index) => {
                  const fraction = index / 4
                  const gx = fraction * map.width
                  const gy = fraction * map.height
                  return <g key={index} stroke="#263440" strokeWidth="1"><line x1={gx} y1="0" x2={gx} y2={map.height} /><line x1="0" y1={gy} x2={map.width} y2={gy} /></g>
                })}
                <line x1={map.px(state.origin.x_m)} y1="0" x2={map.px(state.origin.x_m)} y2={map.height} stroke="#526474" strokeDasharray="4 6" />
                <line x1="0" y1={map.py(state.origin.y_m)} x2={map.width} y2={map.py(state.origin.y_m)} stroke="#526474" strokeDasharray="4 6" />
                {state.active ? <line x1={map.px(state.robot.x_m)} y1={map.py(state.robot.y_m)} x2={map.px(state.active.x_m)} y2={map.py(state.active.y_m)} stroke="#62d6a0" strokeWidth="2" strokeDasharray="7 5" /> : null}
                {state.queue.map((point, index) => {
                  const previous = index === 0 ? state.active : state.queue[index - 1]
                  return previous ? <line key={`route-${point.id}`} x1={map.px(previous.x_m)} y1={map.py(previous.y_m)} x2={map.px(point.x_m)} y2={map.py(point.y_m)} stroke="#75b8ff" strokeWidth="2" strokeDasharray="5 5" /> : null
                })}
                {state.active ? <g><circle cx={map.px(state.active.x_m)} cy={map.py(state.active.y_m)} r="8" fill="#62d6a0" /><text x={map.px(state.active.x_m) + 10} y={map.py(state.active.y_m) - 10} fill="#b5f5d7" fontSize="12">active</text></g> : null}
                {state.queue.map((point, index) => <g key={point.id}><circle cx={map.px(point.x_m)} cy={map.py(point.y_m)} r="6" fill="#75b8ff" /><text x={map.px(point.x_m) + 9} y={map.py(point.y_m) - 8} fill="#c3e0ff" fontSize="11">{index + 1}</text></g>)}
                <g transform={`translate(${map.px(state.robot.x_m)} ${map.py(state.robot.y_m)}) rotate(${-state.robot.yaw_rad * 180 / Math.PI})`}>
                  <circle r="7" fill="#f3c969" stroke="#fff0bd" strokeWidth="2" /><path d="M 0 -8 L 0 -25" stroke="#f3c969" strokeWidth="3" /><path d="M -5 -18 L 0 -26 L 5 -18" fill="none" stroke="#f3c969" strokeWidth="2" />
                </g>
              </svg>
              <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-xs text-secondary"><span><b className="text-[#f3c969]">●</b> robot</span><span><b className="text-[#62d6a0]">●</b> active target</span><span><b className="text-[#75b8ff]">●</b> queued targets</span></div>
            </div>

            <div className="space-y-4">
              <div className="rounded-lg border border-border bg-background/35 p-4">
                <h3 className="font-semibold">Route editor</h3>
                <div className="mt-3 grid grid-cols-3 gap-2">
                  <label className="text-[11px] font-bold uppercase text-muted">X · m<Input className="mt-1" type="number" step="0.05" value={x} onChange={(event) => setX(event.target.value)} /></label>
                  <label className="text-[11px] font-bold uppercase text-muted">Y · m<Input className="mt-1" type="number" step="0.05" value={y} onChange={(event) => setY(event.target.value)} /></label>
                  <label className="text-[11px] font-bold uppercase text-muted">Heading · °<Input className="mt-1" type="number" step="5" value={yawDeg} onChange={(event) => setYawDeg(event.target.value)} /></label>
                </div>
                <p className={`mt-2 text-xs ${validation.ok ? 'text-muted' : 'text-warning'}`}>{validation.message}</p>
                <div className="mt-3 flex flex-wrap gap-2">
                  <Button size="sm" variant="primary" disabled={viewerState !== 'running' || !validation.ok || command.isPending} onClick={() => void addWaypoint('set')}>Set target</Button>
                  <Button size="sm" disabled={viewerState !== 'running' || !validation.ok || state.queue.length >= state.limits.max_queued || command.isPending} onClick={() => void addWaypoint('queue')}>Queue target</Button>
                  <Button size="sm" disabled={viewerState !== 'running' || command.isPending} onClick={() => void send({ operation: 'hold' })}>Hold</Button>
                  <Button size="sm" disabled={viewerState !== 'running' || command.isPending} onClick={() => void send({ operation: 'resume' })}>Resume</Button>
                  <Button size="sm" variant="danger" disabled={viewerState !== 'running' || command.isPending || (!state.active && !state.queue.length)} onClick={() => void cancelRoute()}>Cancel route</Button>
                </div>
                <p className="mt-2 text-[11px] text-muted">Queue capacity: {state.queue.length}/{state.limits.max_queued}. Arrival requires position, heading, speed, yaw rate, support and dwell thresholds.</p>
              </div>

              <div className="rounded-lg border border-border bg-background/35 p-4">
                <h3 className="font-semibold">Speed-conditioned policy</h3>
                {speed ? (
                  <>
                    <label className="mt-3 block text-xs font-bold uppercase tracking-wide text-muted">Requested speed · 0 to {speed.max_speed_mps.toFixed(2)} m/s
                      <input aria-label="Requested speed in metres per second" className="mt-2 w-full accent-accent" type="range" min="0" max={speed.max_speed_mps} step="0.01" value={Math.min(speedValue || 0, speed.max_speed_mps)} onChange={(event) => setDesiredSpeed(event.target.value)} />
                    </label>
                    <div className="mt-3 grid grid-cols-3 gap-2 text-center">
                      <SpeedValue label="Requested" value={speed.requested_mps} />
                      <SpeedValue label="Applied" value={speed.applied_mps} />
                      <SpeedValue label="Measured" value={state.robot.speed_m_s} />
                    </div>
                    <Button className="mt-3 w-full" size="sm" variant="secondary" disabled={!speedValid || viewerState !== 'running' || command.isPending} onClick={() => void send({ operation: 'speed', speed_mps: speedValue })}>Apply speed command</Button>
                    <p className="mt-2 text-[11px] text-muted">Command cap {speed.max_speed_mps.toFixed(2)} m/s · slew limit {speed.max_slew_rate_mps_per_s.toFixed(2)} m/s². The applied command slews toward the request; measured speed can lag.</p>
                    {!speedValid ? <p className="mt-1 text-xs text-warning">Enter a finite speed within the task cap.</p> : null}
                  </>
                ) : <p className="mt-3 text-sm text-muted">Speed command is unavailable for {task || 'this task'}.</p>}
              </div>
            </div>
          </div>

          <div className="mt-4 grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-border bg-border sm:grid-cols-3 xl:grid-cols-6" aria-label="Measured viewer state">
            <PoseValue label="World pose" value={`${state.robot.x_m.toFixed(2)}, ${state.robot.y_m.toFixed(2)} m · ${(state.robot.yaw_rad * 180 / Math.PI).toFixed(0)}°`} />
            <PoseValue label="Planar speed" value={`${state.robot.speed_m_s.toFixed(3)} m/s`} />
            <PoseValue label="Target error" value={`${state.distance_m.toFixed(3)} m`} />
            <PoseValue label="Settle dwell" value={`${state.dwell_s.toFixed(2)} / ${(state.arrival?.dwell_s ?? 0).toFixed(2)} s`} />
            <PoseValue label="Wheel support" value={state.robot.both_wheels_supported ? 'Both supported' : 'Support lost'} />
            <PoseValue label="Completed" value={String(state.completed)} />
          </div>
          <div className="mt-3 flex flex-wrap items-center justify-between gap-2 rounded-lg border border-border bg-background/30 px-4 py-3 text-xs" role="status" aria-live="polite">
            <span className="text-secondary">{ack || 'No new command sent.'}</span>
            <span className={state.last_command?.state === 'rejected' ? 'text-danger' : 'text-muted'}>Viewer acknowledgement: {state.last_command?.state || 'none'}{state.last_command?.error ? ` · ${state.last_command.error}` : ''}</span>
          </div>
          {command.error ? <p className="mt-3 text-sm text-danger" role="alert">Command rejected: {command.error.message}</p> : null}
        </>
      ) : null}
    </section>
  )
}

function PoseValue({ label, value }: { label: string; value: string }) {
  return <div className="min-w-0 bg-panel p-3"><span className="block text-[10px] uppercase tracking-wide text-muted">{label}</span><strong className="mt-1 block truncate text-sm">{value}</strong></div>
}

function SpeedValue({ label, value }: { label: string; value: number }) {
  return <div className="rounded border border-border bg-panel p-2"><span className="block text-[10px] uppercase tracking-wide text-muted">{label}</span><strong className="numeric mt-1 block text-sm">{value.toFixed(3)} m/s</strong></div>
}
