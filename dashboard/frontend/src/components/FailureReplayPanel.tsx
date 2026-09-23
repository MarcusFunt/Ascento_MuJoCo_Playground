import type { IntrospectionCapture, IntrospectionCaptureSummary, PolicyIntrospectionFrame } from '../types'

function trace(values: number[], row: number, min: number, max: number) {
  const height = 36
  if (values.length < 2) return ''
  const span = Math.max(1e-9, max - min)
  return values.map((value, index) => {
    if (!Number.isFinite(value)) return ''
    const x = index * 1000 / (values.length - 1)
    const y = row + height - Math.max(0, Math.min(1, (value - min) / span)) * height
    const previousValid = index > 0 && Number.isFinite(values[index - 1])
    return `${previousValid ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`
  }).filter(Boolean).join(' ')
}

function range(values: number[]): [number, number] {
  const finite = values.filter(Number.isFinite)
  return finite.length ? [Math.min(...finite), Math.max(...finite)] : [0, 1]
}

export function FailureReplayPanel({
  captures,
  capture,
  captureId,
  frameIndex,
  liveFrame,
  loading,
  onSelectCapture,
  onSelectFrame,
  onManualCapture,
  manualCapturePending,
}: {
  captures: IntrospectionCaptureSummary[]
  capture: IntrospectionCapture | undefined
  captureId: string | null
  frameIndex: number
  liveFrame: PolicyIntrospectionFrame | null
  loading: boolean
  onSelectCapture: (eventId: string | null) => void
  onSelectFrame: (index: number) => void
  onManualCapture: () => void
  manualCapturePending: boolean
}) {
  const selectedIndex = capture ? Math.max(0, Math.min(frameIndex, capture.frames.length - 1)) : 0
  const selectedFrame = capture?.frames[selectedIndex]

  return (
    <section className="rounded-lg border border-border bg-background/30 p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold">Failure & recovery replay</h3>
          <p className="mt-1 text-xs text-muted">
            The viewer keeps a rolling ten seconds. Falls, sustained recoveries, and manual captures preserve a synchronized policy trace.
          </p>
        </div>
        <button
          type="button"
          className="rounded-md border border-border-strong bg-raised px-3 py-2 text-xs font-semibold hover:bg-hover disabled:opacity-50"
          disabled={!liveFrame || manualCapturePending}
          onClick={onManualCapture}
        >
          {manualCapturePending ? 'Queueing…' : 'Capture last 10 s'}
        </button>
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-3">
        <label className="text-xs font-medium text-secondary" htmlFor="capture-select">Capture</label>
        <select
          id="capture-select"
          className="min-w-52 rounded-md border border-border bg-panel px-2.5 py-2 text-xs"
          value={captureId || ''}
          onChange={(event) => onSelectCapture(event.target.value || null)}
        >
          <option value="">Live frame</option>
          {captures.map((item) => (
            <option value={item.event_id} key={item.event_id}>
              {item.event_type.toUpperCase()} · seq {item.trigger_sequence} · {new Date(item.created_at).toLocaleTimeString()}
            </option>
          ))}
        </select>
        {capture ? (
          <span className="text-xs text-muted">
            {capture.frames.length} frames · {capture.checkpoint} · {capture.trigger_reason}
          </span>
        ) : captureId && loading ? (
          <span className="text-xs text-muted">Loading captured trace…</span>
        ) : null}
      </div>

      {capture && capture.frames.length > 0 ? (
        <div className="mt-4">
          <div className="flex items-center justify-between text-xs text-secondary">
            <span>Frame {selectedIndex + 1} / {capture.frames.length}</span>
            <span className="font-mono">seq {selectedFrame?.sequence_id ?? '—'} · {selectedFrame?.transition ? `episode ${selectedFrame.transition.episode_id}, step ${selectedFrame.transition.episode_step}` : 'transition unavailable'}</span>
          </div>
          <input
            aria-label="Replay frame"
            className="mt-2 w-full accent-primary"
            type="range"
            min={0}
            max={capture.frames.length - 1}
            value={selectedIndex}
            onChange={(event) => onSelectFrame(Number(event.target.value))}
          />
          {capture.frames.some((item) => item.transition) ? (() => {
            const transitions = capture.frames.map((item) => item.transition)
            const tilt = transitions.map((item) => item ? item.tilt_rad * 180 / Math.PI : 0)
            const margin = transitions.map((item) => item?.balance_margin ?? 0)
            const rewards = transitions.map((item) => item?.reward_rate ?? 0)
            const critics = capture.frames.map((item) => item.critic_value ?? Number.NaN)
            const sigma = capture.frames.map((item) => item.policy_std?.length
              ? item.policy_std.reduce((total, value) => total + value, 0) / item.policy_std.length
              : Number.NaN)
            const maxTilt = Math.max(10, ...tilt)
            const minReward = Math.min(0, ...rewards)
            const maxReward = Math.max(0.01, ...rewards)
            const [minCritic, maxCritic] = range(critics)
            const [minSigma, maxSigma] = range(sigma)
            const hasCritic = critics.some((value) => Number.isFinite(value))
            const hasSigma = sigma.some((value) => Number.isFinite(value))
            const cursorX = selectedIndex * 1000 / Math.max(1, capture.frames.length - 1)
            return (
              <div className="mt-2 rounded border border-border bg-panel/60 p-2">
                <svg className="h-40 w-full" viewBox="0 0 1000 180" preserveAspectRatio="none" role="img" aria-label="Replay tilt, balance margin, reward, critic value, and policy scale timeline">
                  {[0, 36, 72, 108, 144, 180].map((y) => <line key={y} x1="0" x2="1000" y1={y} y2={y} stroke="currentColor" opacity="0.12" />)}
                  {capture.markers.map((marker, index) => {
                    const markerIndex = capture.frames.findIndex((item) => item.sequence_id === marker.sequence)
                    if (markerIndex < 0) return null
                    const x = markerIndex * 1000 / Math.max(1, capture.frames.length - 1)
                    return <line key={`${marker.sequence}-${index}`} x1={x} x2={x} y1="0" y2="180" stroke={marker.kind === 'fall' ? '#ef4444' : '#f59e0b'} strokeDasharray="3 3" opacity="0.7" />
                  })}
                  <path d={trace(tilt, 0, 0, maxTilt)} fill="none" stroke="#fb7185" strokeWidth="2" vectorEffect="non-scaling-stroke" />
                  <path d={trace(margin, 36, 0, 1)} fill="none" stroke="#22c55e" strokeWidth="2" vectorEffect="non-scaling-stroke" />
                  <path d={trace(rewards, 72, minReward, maxReward)} fill="none" stroke="#60a5fa" strokeWidth="2" vectorEffect="non-scaling-stroke" />
                  <path d={trace(critics, 108, minCritic, maxCritic)} fill="none" stroke="#c084fc" strokeWidth="2" vectorEffect="non-scaling-stroke" />
                  <path d={trace(sigma, 144, minSigma, maxSigma)} fill="none" stroke="#fbbf24" strokeWidth="2" vectorEffect="non-scaling-stroke" />
                  <line x1={cursorX} x2={cursorX} y1="0" y2="180" stroke="currentColor" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
                </svg>
                <div className="flex flex-wrap gap-3 px-1 text-[10px] text-muted">
                  <span className="text-rose-400">Tilt · 0–{maxTilt.toFixed(0)}°</span>
                  <span className="text-green-400">Balance margin · 0–100%</span>
                  <span className="text-blue-400">Reward rate · {minReward.toFixed(1)} to {maxReward.toFixed(1)}/s</span>
                  {hasCritic ? <span className="text-purple-400">Critic V(s) · {minCritic.toFixed(2)} to {maxCritic.toFixed(2)}</span> : null}
                  {hasSigma ? <span className="text-amber-400">Mean policy σ · {minSigma.toFixed(3)} to {maxSigma.toFixed(3)}</span> : null}
                </div>
              </div>
            )
          })() : null}
          {capture.markers.length > 0 ? (
            <div className="mt-2 flex flex-wrap gap-2">
              {capture.markers.map((marker, index) => {
                const frameIndex = capture.frames.findIndex((item) => item.sequence_id === marker.sequence)
                return (
                  <button
                    key={`${marker.sequence}-${index}`}
                    type="button"
                    className="rounded border border-border px-2 py-1 text-[11px] text-secondary hover:bg-hover"
                    onClick={() => frameIndex >= 0 && onSelectFrame(frameIndex)}
                    title={marker.label}
                  >
                    {marker.kind} · seq {marker.sequence}
                  </button>
                )
              })}
            </div>
          ) : null}
        </div>
      ) : captureId && !loading ? (
        <p className="mt-3 text-xs text-muted">This capture contains no replay frames.</p>
      ) : null}
    </section>
  )
}
