import { useState } from 'react'
import type { IntrospectionSchema, PolicyIntrospectionFrame } from '../types'

function cellColor(value: number, maxAbs: number, signed: boolean): string {
  const intensity = Math.min(1, Math.abs(value) / Math.max(maxAbs, 1e-9))
  const alpha = 0.08 + intensity * 0.72
  if (!signed) return `rgba(88, 199, 207, ${alpha})`
  return value >= 0 ? `rgba(88, 199, 207, ${alpha})` : `rgba(226, 132, 107, ${alpha})`
}

export function PolicyJacobianHeatmap({
  schema,
  frame,
  selectedActionIndex,
  selectedFeatureIndex,
  onActionSelect,
  onFeatureSelect,
}: {
  schema: IntrospectionSchema
  frame: PolicyIntrospectionFrame
  selectedActionIndex: number | null
  selectedFeatureIndex: number | null
  onActionSelect: (index: number) => void
  onFeatureSelect: (index: number) => void
}) {
  const [signed, setSigned] = useState(true)
  const [normalizePerAction, setNormalizePerAction] = useState(false)
  const matrix = frame.jacobian
  const maxAbs = Math.max(0, ...(matrix || []).flat().map((value) => Math.abs(value)))
  const labels = frame.action_pipeline?.targets.map((target) => target.channel) || []
  const stale = frame.jacobian_age_ms !== null && frame.jacobian_age_ms > 1_000
  return (
    <section className="rounded-xl border border-border bg-panel p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="text-[15px] font-semibold">Local policy Jacobian</h3>
          <p className="mt-1 text-xs text-muted">∂ deterministic action / ∂ raw actor observation · rows are actions, columns are runtime feature dimensions.</p>
        </div>
        {matrix ? <div className="flex flex-wrap items-center gap-2">
          <span className={`rounded-md border px-2 py-1 font-mono text-xs ${stale ? 'border-warning/40 bg-warning/10 text-warning' : 'border-border bg-background/40 text-secondary'}`}>{stale ? 'stale · ' : ''}{frame.jacobian_age_ms?.toFixed(0) ?? '—'} ms old · {frame.jacobian_age_steps ?? '—'} steps</span>
          <button type="button" className="rounded border border-border px-2 py-1 text-[11px] text-secondary hover:bg-hover" onClick={() => setSigned((value) => !value)} aria-pressed={!signed}>{signed ? 'Signed' : 'Absolute'}</button>
          <button type="button" className="rounded border border-border px-2 py-1 text-[11px] text-secondary hover:bg-hover" onClick={() => setNormalizePerAction((value) => !value)} aria-pressed={normalizePerAction}>{normalizePerAction ? 'Rows normalized' : 'Global color scale'}</button>
        </div> : null}
      </div>
      {!matrix ? (
        <p className="mt-4 rounded-md border border-border bg-background/30 p-4 text-sm text-muted">Jacobian sampling is disabled or has not produced a frame yet.</p>
      ) : (
        <div className="mt-4 max-h-[360px] overflow-auto rounded-lg border border-border">
          <table className="text-center text-[10px]">
            <thead className="sticky top-0 z-10 bg-panel text-muted">
              <tr><th className="sticky left-0 z-20 bg-panel px-3 py-2 text-left">Action \ feature</th>{schema.actor.features.map((feature) => <th key={feature.index} className={`min-w-9 px-1 py-2 ${feature.index === selectedFeatureIndex ? 'text-foreground' : 'text-muted'}`} title={`${feature.term} · ${feature.label}`}><button type="button" className="w-full hover:text-foreground" onClick={() => onFeatureSelect(feature.index)}>{feature.index}</button></th>)}</tr>
            </thead>
            <tbody>
              {matrix.map((row, actionIndex) => (
                <tr key={actionIndex} className={`border-t border-border/70 ${actionIndex === selectedActionIndex ? 'outline outline-1 outline-inset outline-primary/40' : ''}`}>
                  <th className="sticky left-0 z-10 whitespace-nowrap bg-panel px-3 py-2 text-left font-semibold text-foreground"><button type="button" onClick={() => onActionSelect(actionIndex)} className="text-left hover:text-primary">{labels[actionIndex] || `action ${actionIndex}`}</button></th>
                  {row.map((value, featureIndex) => {
                    const scale = normalizePerAction ? Math.max(0, ...row.map((item) => Math.abs(item))) : maxAbs
                    const feature = schema.actor.features[featureIndex]
                    return (
                    <td key={featureIndex} className={`border-l border-border/40 p-0 font-mono text-foreground ${featureIndex === selectedFeatureIndex ? 'outline outline-1 outline-inset outline-primary' : ''}`} style={{ backgroundColor: cellColor(signed ? value : Math.abs(value), scale, signed) }} title={`${labels[actionIndex] || `action ${actionIndex}`} ← ${feature?.term}.${feature?.label}: ${value.toExponential(3)}`}>
                      <button type="button" className="block w-full px-1 py-2" onClick={() => { onActionSelect(actionIndex); onFeatureSelect(featureIndex) }} aria-label={`Action ${labels[actionIndex] || actionIndex}, feature ${feature?.label || featureIndex}, derivative ${value}`}>
                        {signed ? value.toPrecision(2) : Math.abs(value).toPrecision(2)}
                      </button>
                    </td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}
