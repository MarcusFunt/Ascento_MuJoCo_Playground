import type { PolicyIntrospectionFrame } from '../types'

export function ActionPipelinePanel({ frame }: { frame: PolicyIntrospectionFrame }) {
  const pipeline = frame.action_pipeline
  return (
    <section className="rounded-xl border border-border bg-panel p-5">
      <div>
        <h3 className="text-[15px] font-semibold">Action → actuator pipeline</h3>
        <p className="mt-1 text-xs text-muted">Values are paired to the policy observation at the same sequence.</p>
      </div>
      {!pipeline ? (
        <p className="mt-4 rounded-md border border-border bg-background/30 p-4 text-sm text-muted">Waiting for the first completed simulation transition.</p>
      ) : (
        <div className="mt-4 overflow-x-auto">
          <table className="w-full min-w-[720px] text-left text-xs">
            <thead className="text-[10px] uppercase tracking-wide text-muted">
              <tr><th className="px-3 py-2">Channel</th><th className="px-3 py-2">Actor</th><th className="px-3 py-2">Wrapper</th><th className="px-3 py-2">Action term</th><th className="px-3 py-2">Physical target</th></tr>
            </thead>
            <tbody>
              {pipeline.targets.map((target, index) => {
                const badges = [
                  pipeline.wrapper_clipped_flags[index] ? 'wrapper clip' : null,
                  pipeline.action_term_clipped_flags[index] ? 'action clip' : null,
                  target.joint_limit_clipped ? 'joint limit' : null,
                ].filter(Boolean)
                return (
                  <tr key={target.channel} className="border-t border-border/70">
                    <th className="whitespace-nowrap px-3 py-2 font-semibold text-foreground">{target.channel}</th>
                    <td className="numeric px-3 py-2 text-secondary">{pipeline.actor_output[index]?.toFixed(4) ?? '—'}</td>
                    <td className="numeric px-3 py-2 text-secondary">{pipeline.wrapper_clipped[index]?.toFixed(4) ?? '—'}</td>
                    <td className="numeric px-3 py-2 text-secondary">{pipeline.processed_action[index]?.toFixed(4) ?? '—'}</td>
                    <td className="px-3 py-2">
                      <span className="numeric text-foreground">{target.value.toFixed(4)} {target.unit}</span>
                      {badges.map((badge) => <span key={badge} className="ml-2 rounded bg-warning/15 px-1.5 py-0.5 text-[10px] text-warning">{badge}</span>)}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}
