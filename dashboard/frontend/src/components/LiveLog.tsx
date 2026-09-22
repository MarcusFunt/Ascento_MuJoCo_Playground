import { useEffect, useRef, useState } from 'react'
import { api } from '../api'

export function LiveLog({ runId }: { runId: string }) {
  const [lines, setLines] = useState<string[]>([])
  const [autoScroll, setAutoScroll] = useState(true)
  const ref = useRef<HTMLPreElement>(null)

  useEffect(() => {
    let cancelled = false
    setLines([])
    void api.logs(runId, 350).then((value) => {
      if (!cancelled) setLines(value.lines.slice(-700))
    }).catch(() => undefined)

    const source = new EventSource(`/api/runs/${runId}/logs/stream`)
    source.onmessage = (event) => {
      try {
        const value = JSON.parse(event.data)
        if (typeof value.line === 'string') {
          setLines((current) => [...current, value.line].slice(-700))
        }
      } catch {
        // Keep the stream alive if a malformed line arrives.
      }
    }
    return () => {
      cancelled = true
      source.close()
    }
  }, [runId])

  useEffect(() => {
    if (autoScroll && ref.current) ref.current.scrollTop = ref.current.scrollHeight
  }, [lines, autoScroll])

  return (
    <section className="rounded-xl border border-border bg-panel p-5">
      <div className="flex items-center justify-between gap-3">
        <h3 className="text-[15px] font-semibold">Live training log</h3>
        <label className="flex items-center gap-2 text-xs text-muted">
          <input type="checkbox" checked={autoScroll} onChange={(event) => setAutoScroll(event.target.checked)} />
          Follow
        </label>
      </div>
      <pre
        ref={ref}
        className="mt-4 h-[360px] overflow-auto rounded-lg border border-border bg-[#050607] p-4 font-mono text-[12px] leading-relaxed text-secondary"
      >
        {lines.join('\n')}
      </pre>
    </section>
  )
}
