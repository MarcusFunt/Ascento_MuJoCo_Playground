import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Network } from 'lucide-react'
import { api } from '../api'
import type { IntrospectionSchema, PolicyBranchArchitecture, PolicyIntrospectionFrame } from '../types'
import { AnimatedPolicyNetwork } from './AnimatedPolicyNetwork'
import { Button } from './ui/button'

type BranchName = 'actor' | 'critic'

export function PolicyArchitectureCard({
  runId,
  checkpointPath,
  liveFrame,
  runtimeSchema,
}: {
  runId: string
  checkpointPath?: string
  liveFrame?: PolicyIntrospectionFrame | null
  runtimeSchema?: IntrospectionSchema | null
}) {
  const [branchName, setBranchName] = useState<BranchName>('actor')
  const architecture = useQuery({
    queryKey: ['architecture', runId],
    queryFn: () => api.architecture(runId),
    staleTime: 20_000,
    refetchInterval: 60_000,
  })

  const runtimeBranch = branchName === 'actor'
    ? runtimeSchema?.actor_network
    : runtimeSchema?.critic_network
  const branch = (runtimeBranch || architecture.data?.[branchName]) as PolicyBranchArchitecture | undefined
  const layers = branch?.layers || []
  const selectedBranchLabel = branchName === 'actor' ? 'Actor policy' : 'Critic value'
  const architectureDescription = layers.length
    ? `${selectedBranchLabel}: ${layers[0]} inputs, ${layers.slice(1, -1).join(', ') || 'no hidden layers'} hidden units, ${layers[layers.length - 1]} ${branchName === 'actor' ? 'action outputs' : 'value output'}. Activation: ${branch?.activation || 'unknown'}.`
    : ''

  const liveActivations = Object.entries(liveFrame?.hidden_activations || {})
    .filter(([name]) => name.startsWith(`${branchName}.`))
    .map(([, values]) => values)

  return (
    <section className="rounded-xl border border-border bg-panel p-5 sm:p-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-[0.1em] text-muted"><Network size={14} /> Model structure</div>
          <h2 className="mt-1 text-xl font-semibold tracking-[-0.02em]">Policy network</h2>
          <p className="mt-2 max-w-3xl text-sm leading-relaxed text-muted">
            Checkpoint-backed layer sizes and parameter statistics, with signed activation colors sampled from the selected policy frame.
          </p>
        </div>
        {runtimeSchema ? (
          <span className="rounded-md border border-border bg-background/50 px-2.5 py-1.5 font-mono text-xs text-secondary">
            {runtimeSchema.checkpoint} · generation {runtimeSchema.policy_generation ?? liveFrame?.policy_generation ?? '—'}
          </span>
        ) : architecture.data?.available ? (
          <span className="rounded-md border border-border bg-background/50 px-2.5 py-1.5 font-mono text-xs text-secondary">
            {architecture.data.iteration !== null && architecture.data.iteration !== undefined
              ? `model · ${architecture.data.iteration.toLocaleString()}`
              : architecture.data.checkpoint}
          </span>
        ) : null}
      </div>

      {architecture.isLoading && !runtimeSchema ? (
        <div className="mt-5 h-[360px] animate-pulse rounded-lg border border-border bg-background/40" aria-label="Loading policy architecture" />
      ) : architecture.error && !runtimeSchema ? (
        <p className="mt-5 rounded-lg border border-danger/35 bg-danger/10 p-4 text-sm text-danger">{architecture.error.message}</p>
      ) : (!runtimeSchema && !architecture.data?.available) || !branch ? (
        <div className="mt-5 rounded-lg border border-border bg-background/35 p-4 text-sm text-secondary" role="status">
          <p>{architecture.data?.message || 'Policy architecture is not available for this run yet.'}</p>
          {checkpointPath ? <p className="mt-2 break-all font-mono text-xs text-muted">Run record: {checkpointPath}</p> : null}
        </div>
      ) : (
        <>
          <div className="mt-5 flex flex-wrap items-center justify-between gap-3">
            <div className="flex flex-wrap gap-2" role="tablist" aria-label="Policy network">
              {(['actor', 'critic'] as const).map((name) => (
                <Button
                  key={name}
                  type="button"
                  size="sm"
                  variant={branchName === name ? 'primary' : 'secondary'}
                  role="tab"
                  aria-selected={branchName === name}
                  onClick={() => setBranchName(name)}
                >
                  {name === 'actor' ? 'Actor policy' : 'Critic value'}
                </Button>
              ))}
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <span className="ml-1 text-xs text-muted">
                {branchName === 'actor'
                  ? `${branch.distribution || 'Policy'} output${branch.std_parameters ? ` · ${branch.std_parameters} scale values` : ''}`
                  : 'Scalar value estimate'}
              </span>
            </div>
          </div>

          <div className="mt-3" aria-label={architectureDescription}>
            <AnimatedPolicyNetwork
              branch={branch}
              branchName={branchName}
              liveActivations={liveActivations}
            />
          </div>

          <p className="mt-3 text-xs leading-relaxed text-muted" aria-live="polite">{architectureDescription}</p>
          <p className="mt-1 text-xs leading-relaxed text-muted">
            Circle columns are sampled deterministically from {branchName} units{liveFrame ? ` at sequence ${liveFrame.sequence_id}` : ''}. Color shows activation sign and relative magnitude; connections and weight statistics are from the selected checkpoint.
          </p>
        </>
      )}
    </section>
  )
}
