import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Network, Pause, Play, Rabbit } from 'lucide-react'
import { api } from '../api'
import type { PolicyBranchArchitecture } from '../types'
import { AnimatedPolicyNetwork } from './AnimatedPolicyNetwork'
import { Button } from './ui/button'

type BranchName = 'actor' | 'critic'

export function PolicyArchitectureCard({
  runId,
  checkpointPath,
}: {
  runId: string
  checkpointPath?: string
}) {
  const [branchName, setBranchName] = useState<BranchName>('actor')
  const [paused, setPaused] = useState(false)
  const [speed, setSpeed] = useState(1)
  const architecture = useQuery({
    queryKey: ['architecture', runId],
    queryFn: () => api.architecture(runId),
    staleTime: 60_000,
    refetchInterval: (query) => query.state.data?.available ? false : 15_000,
  })

  const branch = architecture.data?.[branchName] as PolicyBranchArchitecture | undefined
  const layers = branch?.layers || []
  const selectedBranchLabel = branchName === 'actor' ? 'Actor policy' : 'Critic value'
  const architectureDescription = layers.length
    ? `${selectedBranchLabel}: ${layers[0]} inputs, ${layers.slice(1, -1).join(', ') || 'no hidden layers'} hidden units, ${layers[layers.length - 1]} ${branchName === 'actor' ? 'action outputs' : 'value output'}. Activation: ${branch?.activation || 'unknown'}.`
    : ''

  const cycleSpeed = () => {
    setSpeed((current) => current === 0.5 ? 1 : current === 1 ? 2 : 0.5)
  }

  return (
    <section className="rounded-xl border border-border bg-panel p-5 sm:p-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-[0.1em] text-muted"><Network size={14} /> Model structure</div>
          <h2 className="mt-1 text-xl font-semibold tracking-[-0.02em]">Policy network</h2>
          <p className="mt-2 max-w-3xl text-sm leading-relaxed text-muted">
            Checkpoint-backed layer sizes and parameter statistics, shown with a sampled neuron view. The moving traces illustrate feed-forward signal direction.
          </p>
        </div>
        {architecture.data?.available ? (
          <span className="rounded-md border border-border bg-background/50 px-2.5 py-1.5 font-mono text-xs text-secondary">
            {architecture.data.iteration !== null && architecture.data.iteration !== undefined
              ? `model · ${architecture.data.iteration.toLocaleString()}`
              : architecture.data.checkpoint}
          </span>
        ) : null}
      </div>

      {architecture.isLoading ? (
        <div className="mt-5 h-[360px] animate-pulse rounded-lg border border-border bg-background/40" aria-label="Loading policy architecture" />
      ) : architecture.error ? (
        <p className="mt-5 rounded-lg border border-danger/35 bg-danger/10 p-4 text-sm text-danger">{architecture.error.message}</p>
      ) : !architecture.data?.available || !branch ? (
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
              <Button
                type="button"
                size="sm"
                variant="ghost"
                onClick={() => setPaused((value) => !value)}
                aria-pressed={paused}
              >
                {paused ? <Play size={14} /> : <Pause size={14} />}
                {paused ? 'Animate' : 'Pause'}
              </Button>
              <Button type="button" size="sm" variant="ghost" onClick={cycleSpeed} title="Cycle animation speed">
                <Rabbit size={14} /> {speed}×
              </Button>
              <span className="ml-1 text-xs text-muted">
                {branchName === 'actor'
                  ? `${branch.distribution || 'Policy'} output${branch.std_parameters ? ` · ${branch.std_parameters} scale values` : ''}`
                  : 'Scalar value estimate'}
              </span>
            </div>
          </div>

          <div className="mt-3" aria-label={architectureDescription}>
            <AnimatedPolicyNetwork branch={branch} branchName={branchName} paused={paused} speed={speed} />
          </div>

          <p className="mt-3 text-xs leading-relaxed text-muted" aria-live="polite">{architectureDescription}</p>
          <p className="mt-1 text-xs leading-relaxed text-muted">
            Circle columns are sampled when a layer is wider than the display. Connection density and the weight statistics come from the checkpoint; the flowing dashes are a schematic animation, not live neuron activations.
          </p>
        </>
      )}
    </section>
  )
}
