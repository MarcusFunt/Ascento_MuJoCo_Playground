import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import * as echarts from 'echarts/core'
import { GraphChart } from 'echarts/charts'
import { TooltipComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import ReactEChartsCore from 'echarts-for-react/lib/core'
import { Network } from 'lucide-react'
import { api } from '../api'
import type { PolicyBranchArchitecture } from '../types'
import { Button } from './ui/button'

echarts.use([GraphChart, TooltipComponent, CanvasRenderer])

type BranchName = 'actor' | 'critic'

function layerName(index: number, count: number, branch: BranchName): string {
  if (index === 0) return branch === 'actor' ? 'Observations' : 'Critic inputs'
  if (index === count - 1) return branch === 'actor' ? 'Action means' : 'Value estimate'
  return `Hidden layer ${index}`
}

export function PolicyArchitectureCard({
  runId,
  checkpointPath,
}: {
  runId: string
  checkpointPath?: string
}) {
  const [branchName, setBranchName] = useState<BranchName>('actor')
  const architecture = useQuery({
    queryKey: ['architecture', runId],
    queryFn: () => api.architecture(runId),
    staleTime: 60_000,
    refetchInterval: (query) => query.state.data?.available ? false : 15_000,
  })

  const branch = architecture.data?.[branchName] as PolicyBranchArchitecture | undefined
  const layers = branch?.layers || []
  const chartOption = useMemo(() => {
    const accent = branchName === 'actor' ? '#58c7cf' : '#e5aa45'
    const nodes = layers.map((size, index) => {
      const title = layerName(index, layers.length, branchName)
      const detail = `${size.toLocaleString()} ${index === 0 ? 'features' : 'units'}`
      return {
        id: `layer-${index}`,
        name: title,
        title,
        detail,
        x: 58 + index * 140,
        y: 112,
        symbol: 'roundRect',
        symbolSize: [110, 72],
        itemStyle: {
          color: '#14171a',
          borderColor: index === layers.length - 1 ? accent : '#343a40',
          borderWidth: index === layers.length - 1 ? 2 : 1,
        },
        label: {
          show: true,
          position: 'inside',
          formatter: `{title|${title}}\n{detail|${detail}}${index > 0 && index < layers.length - 1 ? `\n{activation|${branch?.activation || 'Unknown'}` : ''}${index > 0 && index < layers.length - 1 ? '}' : ''}`,
          rich: {
            title: { color: '#f1f3f5', fontSize: 12, fontWeight: 700, lineHeight: 20 },
            detail: { color: '#a8afb7', fontSize: 11, lineHeight: 17 },
            activation: { color: accent, fontSize: 9, fontWeight: 700, lineHeight: 15 },
          },
        },
      }
    })
    const links = layers.slice(1).map((_, index) => ({
      source: `layer-${index}`,
      target: `layer-${index + 1}`,
      lineStyle: { color: accent, opacity: 0.52, width: 1.5 },
    }))
    const summary = `${branchName === 'actor' ? 'Actor policy' : 'Critic value'}: ${layers.join(' to ')} layer widths.`
    return {
      animation: false,
      backgroundColor: 'transparent',
      aria: { enabled: true, description: summary },
      tooltip: {
        trigger: 'item',
        backgroundColor: '#14171a',
        borderColor: '#343a40',
        textStyle: { color: '#f1f3f5', fontSize: 12 },
        formatter: (params: { data?: { title?: string; detail?: string } }) =>
          `${params.data?.title || ''}<br/>${params.data?.detail || ''}`,
      },
      series: [{
        type: 'graph',
        layout: 'none',
        roam: false,
        edgeSymbol: ['none', 'arrow'],
        edgeSymbolSize: [0, 8],
        lineStyle: { color: accent, opacity: 0.52 },
        data: nodes,
        links,
        emphasis: { focus: 'adjacency', lineStyle: { width: 2.5, opacity: 0.9 } },
      }],
    }
  }, [branch?.activation, branchName, layers])

  const selectedBranchLabel = branchName === 'actor' ? 'Actor policy' : 'Critic value'
  const architectureDescription = layers.length
    ? `${selectedBranchLabel}: ${layers[0]} inputs, ${layers.slice(1, -1).join(', ') || 'no hidden layers'} hidden units, ${layers[layers.length - 1]} ${branchName === 'actor' ? 'action outputs' : 'value output'}. Activation: ${branch?.activation || 'unknown'}.`
    : ''

  return (
    <section className="rounded-xl border border-border bg-panel p-5 sm:p-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-[0.1em] text-muted"><Network size={14} /> Model structure</div>
          <h2 className="mt-1 text-xl font-semibold tracking-[-0.02em]">Policy architecture</h2>
          <p className="mt-2 max-w-2xl text-sm leading-relaxed text-muted">
            Layer widths are read from the saved checkpoint; each node represents a whole layer, not an individual neuron.
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
        <div className="mt-5 h-48 animate-pulse rounded-lg border border-border bg-background/40" aria-label="Loading policy architecture" />
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
            <div className="flex gap-2" role="tablist" aria-label="Policy network">
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
            <span className="text-xs text-muted">
              {branchName === 'actor'
                ? `${branch.distribution || 'Policy'} output${branch.std_parameters ? ` · ${branch.std_parameters} scale values` : ''}`
                : 'Scalar value estimate'}
            </span>
          </div>

          <div className="mt-3 overflow-x-auto rounded-lg border border-border bg-background/35" role="img" aria-label={architectureDescription}>
            <ReactEChartsCore
              echarts={echarts}
              option={chartOption}
              style={{ width: 700, height: 224, maxWidth: 'none' }}
              notMerge
              lazyUpdate
            />
          </div>
          <p className="mt-3 text-xs leading-relaxed text-muted" aria-live="polite">{architectureDescription}</p>
          <p className="mt-1 text-xs leading-relaxed text-muted">
            This is the configured network structure, not live neuron activations or a visualization of individual weights.
          </p>
        </>
      )}
    </section>
  )
}
