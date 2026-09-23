import { useMemo, useState } from 'react'
import type { ObservationFeature, ObservationSchema, PolicyIntrospectionFrame } from '../types'

function formatValue(value: number | undefined): string {
  return Number.isFinite(value) ? Number(value).toFixed(4) : '—'
}

function clipLabel(clip: ObservationFeature['clip']): string {
  return Array.isArray(clip) ? `${clip[0]} … ${clip[1]}` : '—'
}

function atClipBoundary(value: number | undefined, clip: ObservationFeature['clip']): boolean {
  if (!Array.isArray(clip) || !Number.isFinite(value)) return false
  const tolerance = Math.max(1e-6, Math.abs(clip[1] - clip[0]) * 1e-5)
  return Math.abs(Number(value) - clip[0]) <= tolerance || Math.abs(Number(value) - clip[1]) <= tolerance
}

export function ObservationInspector({
  schema,
  frame,
  selectedActionIndex,
  selectedFeatureIndex,
}: {
  schema: ObservationSchema
  frame: PolicyIntrospectionFrame
  selectedActionIndex: number | null
  selectedFeatureIndex: number | null
}) {
  const [query, setQuery] = useState('')
  const [largestFirst, setLargestFirst] = useState(false)
  const jacobianMagnitude = useMemo(() => {
    const matrix = frame.jacobian || []
    return schema.features.map((_feature, index) => selectedActionIndex !== null && matrix[selectedActionIndex]
      ? Math.abs(matrix[selectedActionIndex][index] || 0)
      : Math.max(0, ...matrix.map((row) => Math.abs(row[index] || 0)))
    )
  }, [frame.jacobian, schema.features, selectedActionIndex])

  const groups = schema.terms.map((term) => {
    const features = schema.features
      .filter((feature) => feature.index >= term.start && feature.index < term.end)
      .filter((feature) => `${term.name} ${feature.label} ${feature.source}`.toLowerCase().includes(query.toLowerCase()))
    const rows = largestFirst
      ? [...features].sort((a, b) => (jacobianMagnitude[b.index] || 0) - (jacobianMagnitude[a.index] || 0))
      : features
    return { term, rows }
  }).filter((group) => group.rows.length > 0)

  return (
    <section className="rounded-xl border border-border bg-panel p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="text-[15px] font-semibold">Actor observations</h3>
          <p className="mt-1 text-xs text-muted">
            Exact policy input before model normalization, paired with the normalizer output. ObservationManager scaling/clipping may already have been applied.
          </p>
        </div>
        <span className="font-mono text-xs text-muted">{schema.input_dim} dimensions</span>
      </div>
      <div className="mt-4 flex flex-wrap gap-2">
        <input
          className="h-9 min-w-48 flex-1 rounded-md border border-border bg-background px-3 text-sm text-foreground outline-none focus:border-foreground/50"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search observation terms or features"
          aria-label="Search actor observations"
        />
        <button
          type="button"
          className="rounded-md border border-border bg-background px-3 text-xs font-semibold text-secondary hover:bg-hover"
          onClick={() => setLargestFirst((value) => !value)}
          aria-pressed={largestFirst}
        >
          {largestFirst ? 'Jacobian-ranked' : 'Schema order'}
        </button>
      </div>
      <div className="mt-3 max-h-[440px] space-y-2 overflow-auto pr-1">
        {groups.map(({ term, rows }) => (
          <details key={`${term.name}-${term.start}`} open className="rounded-lg border border-border bg-background/30">
            <summary className="cursor-pointer px-3 py-2 text-xs font-bold text-secondary">
              {term.name} <span className="ml-2 font-mono font-normal text-muted">[{term.start}:{term.end}] · {term.source}</span>
            </summary>
            <div className="overflow-x-auto border-t border-border">
              <table className="w-full min-w-[720px] text-left text-xs">
                <thead className="sticky top-0 bg-panel text-[10px] uppercase tracking-wide text-muted">
                  <tr>
                    <th className="px-3 py-2">Feature</th><th className="px-3 py-2">Raw</th><th className="px-3 py-2">Normalized</th>
                    <th className="px-3 py-2">Unit</th><th className="px-3 py-2">Scale</th><th className="px-3 py-2">Obs clip</th><th className="px-3 py-2">Max |Jacobian|</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((feature) => {
                    const raw = frame.raw_observation[feature.index]
                    const normalized = frame.normalized_observation[feature.index]
                    const clipped = atClipBoundary(raw, feature.clip)
                    return (
                      <tr key={feature.index} className={`border-t border-border/70 ${feature.index === selectedFeatureIndex ? 'bg-primary/10' : ''}`}>
                        <th className="whitespace-nowrap px-3 py-2 font-medium text-foreground">
                          {feature.label}{clipped ? <span className="ml-2 rounded bg-warning/15 px-1.5 py-0.5 text-[10px] text-warning">at clip</span> : null}
                        </th>
                        <td className="numeric px-3 py-2 text-secondary">{formatValue(raw)}</td>
                        <td className="numeric px-3 py-2 text-secondary">{formatValue(normalized)}</td>
                        <td className="whitespace-nowrap px-3 py-2 text-muted">{feature.unit || '—'}</td>
                        <td className="numeric px-3 py-2 text-muted">{feature.scale === null ? '—' : String(feature.scale)}</td>
                        <td className="numeric whitespace-nowrap px-3 py-2 text-muted">{clipLabel(feature.clip)}</td>
                        <td className="numeric px-3 py-2 text-muted">{formatValue(jacobianMagnitude[feature.index])}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </details>
        ))}
        {!groups.length ? <p className="p-4 text-sm text-muted">No observation features match this search.</p> : null}
      </div>
    </section>
  )
}
