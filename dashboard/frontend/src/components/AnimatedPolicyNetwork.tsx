/*
 * Layered SVG network visualization adapted from the visual approach used by
 * TensorFlow Playground (Copyright 2016 Google Inc., Apache-2.0):
 * https://github.com/tensorflow/playground
 *
 * Reimplemented for React/SVG and large PPO MLPs. Individual circles are
 * representative neurons when a layer is wider than the visible sample.
 */

import { useMemo, useState } from 'react'
import type { PolicyBranchArchitecture } from '../types'
import { fmtNumber } from '../lib/utils'

type BranchName = 'actor' | 'critic'

type VisibleNeuron = {
  id: string
  y: number
  outputLabel?: string
}

const MAX_VISIBLE = 11
const ACTOR_OUTPUT_LABELS = ['L hip', 'L knee', 'L wheel', 'R hip', 'R knee', 'R wheel']

function layerLabel(index: number, count: number, branch: BranchName): string {
  if (index === 0) return branch === 'actor' ? 'Observations' : 'Critic inputs'
  if (index === count - 1) return branch === 'actor' ? 'Actions' : 'Value'
  return `Hidden ${index}`
}

function visibleNeurons(
  size: number,
  layerIndex: number,
  x: number,
  branch: BranchName,
  isOutput: boolean,
): VisibleNeuron[] {
  const count = Math.max(1, Math.min(MAX_VISIBLE, size))
  const top = 76
  const bottom = 274
  const span = Math.max(1, count - 1)
  return Array.from({ length: count }, (_, index) => {
    const outputLabel = isOutput && branch === 'actor' && size === 6
      ? ACTOR_OUTPUT_LABELS[index]
      : isOutput && branch === 'critic' && size === 1
        ? 'V(s)'
        : undefined
    return {
      id: `l${layerIndex}-n${index}-x${Math.round(x)}`,
      y: count === 1 ? (top + bottom) / 2 : top + (index / span) * (bottom - top),
      outputLabel,
    }
  })
}

function transitionStrength(branch: PolicyBranchArchitecture, index: number): number {
  const stats = branch.linear_layers?.[index]
  if (!stats) return 0.5
  const all = branch.linear_layers || []
  const maximum = Math.max(...all.map((item) => item.weight_rms || 0), 1e-9)
  return Math.max(0.18, Math.min(1, stats.weight_rms / maximum))
}

export function AnimatedPolicyNetwork({
  branch,
  branchName,
  paused,
  speed,
}: {
  branch: PolicyBranchArchitecture
  branchName: BranchName
  paused: boolean
  speed: number
}) {
  const [selectedLayer, setSelectedLayer] = useState(1)
  const width = 820
  const height = 340
  const left = 68
  const right = branchName === 'actor' ? 112 : 74
  const usable = width - left - right
  const layerCount = branch.layers.length
  const accent = branchName === 'actor' ? '#58c7cf' : '#e5aa45'

  const layout = useMemo(() => branch.layers.map((size, index) => {
    const x = layerCount <= 1 ? width / 2 : left + (index / (layerCount - 1)) * usable
    return {
      size,
      index,
      x,
      neurons: visibleNeurons(size, index, x, branchName, index === layerCount - 1),
    }
  }), [branch.layers, branchName, layerCount, usable])

  const selected = Math.max(0, Math.min(selectedLayer, layerCount - 1))
  const inbound = selected > 0 ? branch.linear_layers?.[selected - 1] : undefined
  const selectedSize = branch.layers[selected] || 0
  const selectedName = layerLabel(selected, layerCount, branchName)
  const hidden = selected > 0 && selected < layerCount - 1

  return (
    <div className={paused ? 'nn-animation-paused' : undefined}>
      <div className="overflow-x-auto rounded-lg border border-border bg-background/35">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="block min-w-[720px] w-full"
          role="img"
          aria-label={`${branchName} neural network with layer widths ${branch.layers.join(', ')}`}
        >
          <defs>
            <filter id={`nn-glow-${branchName}`} x="-100%" y="-100%" width="300%" height="300%">
              <feGaussianBlur stdDeviation="2.4" result="blur" />
              <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
            </filter>
          </defs>

          {layout.slice(0, -1).map((sourceLayer, transitionIndex) => {
            const targetLayer = layout[transitionIndex + 1]
            const strength = transitionStrength(branch, transitionIndex)
            const edges = sourceLayer.neurons.flatMap((source, sourceIndex) =>
              targetLayer.neurons.map((target, targetIndex) => ({
                key: `${source.id}-${target.id}`,
                source,
                target,
                sourceIndex,
                targetIndex,
              })),
            )
            const activeEdges = edges.filter((edge) =>
              (edge.sourceIndex * 3 + edge.targetIndex * 5 + transitionIndex) % 11 === 0,
            ).slice(0, 8)

            return (
              <g key={`transition-${transitionIndex}`}>
                {edges.map((edge) => (
                  <line
                    key={edge.key}
                    x1={sourceLayer.x + 8}
                    y1={edge.source.y}
                    x2={targetLayer.x - 8}
                    y2={edge.target.y}
                    stroke="#66707a"
                    strokeWidth={0.45 + strength * 0.55}
                    opacity={0.045 + strength * 0.055}
                  />
                ))}
                {activeEdges.map((edge, edgeIndex) => (
                  <line
                    key={`active-${edge.key}`}
                    x1={sourceLayer.x + 8}
                    y1={edge.source.y}
                    x2={targetLayer.x - 8}
                    y2={edge.target.y}
                    stroke={accent}
                    strokeWidth={0.9 + strength * 0.9}
                    strokeDasharray="2 16"
                    opacity={0.72}
                    className="nn-flow-edge"
                    style={{
                      animationDuration: `${Math.max(0.55, 1.65 / speed)}s`,
                      animationDelay: `${transitionIndex * 0.18 + edgeIndex * 0.08}s`,
                    }}
                  />
                ))}
              </g>
            )
          })}

          {layout.map((layer) => {
            const isSelected = selected === layer.index
            const isHidden = layer.index > 0 && layer.index < layerCount - 1
            return (
              <g
                key={`layer-${layer.index}`}
                role="button"
                tabIndex={0}
                aria-label={`${layerLabel(layer.index, layerCount, branchName)}: ${layer.size} units`}
                onClick={() => setSelectedLayer(layer.index)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' || event.key === ' ') {
                    event.preventDefault()
                    setSelectedLayer(layer.index)
                  }
                }}
                className="cursor-pointer outline-none"
              >
                <text
                  x={layer.x}
                  y={30}
                  textAnchor="middle"
                  fill={isSelected ? '#f1f3f5' : '#a8afb7'}
                  fontSize="12"
                  fontWeight="700"
                >
                  {layerLabel(layer.index, layerCount, branchName)}
                </text>
                <text x={layer.x} y={48} textAnchor="middle" fill="#6e767f" fontSize="10">
                  {layer.size.toLocaleString()} {layer.index === 0 ? 'features' : layer.index === layerCount - 1 ? 'outputs' : 'neurons'}
                </text>

                {layer.neurons.map((neuron, neuronIndex) => (
                  <g key={neuron.id}>
                    <circle
                      cx={layer.x}
                      cy={neuron.y}
                      r={isSelected ? 7.2 : 6.2}
                      fill={isSelected ? '#171b1f' : '#111417'}
                      stroke={isSelected ? accent : '#58616a'}
                      strokeWidth={isSelected ? 1.8 : 1}
                      className={isSelected ? 'nn-neuron-pulse' : undefined}
                      style={isSelected ? {
                        animationDuration: `${Math.max(0.8, 2.2 / speed)}s`,
                        animationDelay: `${layer.index * 0.16 + neuronIndex * 0.045}s`,
                        transformOrigin: `${layer.x}px ${neuron.y}px`,
                      } : undefined}
                    >
                      <title>{`${layerLabel(layer.index, layerCount, branchName)} representative neuron ${neuronIndex + 1}`}</title>
                    </circle>
                    {neuron.outputLabel ? (
                      <text x={layer.x + 13} y={neuron.y + 3} fill="#a8afb7" fontSize="9">{neuron.outputLabel}</text>
                    ) : null}
                  </g>
                ))}

                {layer.size > layer.neurons.length ? (
                  <>
                    <text x={layer.x} y={294} textAnchor="middle" fill="#6e767f" fontSize="15">⋮</text>
                    <text x={layer.x} y={313} textAnchor="middle" fill="#6e767f" fontSize="9">
                      showing {layer.neurons.length} / {layer.size.toLocaleString()}
                    </text>
                  </>
                ) : null}

                {isHidden ? (
                  <text x={layer.x} y={330} textAnchor="middle" fill={accent} fontSize="9" fontWeight="700">
                    {branch.activation}
                  </text>
                ) : null}
              </g>
            )
          })}
        </svg>
      </div>

      <div className="mt-3 grid gap-px overflow-hidden rounded-lg border border-border bg-border sm:grid-cols-2 lg:grid-cols-4">
        <div className="bg-panel px-4 py-3">
          <span className="block text-[10px] font-bold uppercase tracking-[0.08em] text-muted">Selected layer</span>
          <strong className="mt-1 block text-sm">{selectedName}</strong>
          <span className="numeric mt-0.5 block text-xs text-muted">{selectedSize.toLocaleString()} {hidden ? 'ELU neurons' : selected === 0 ? 'features' : 'outputs'}</span>
        </div>
        <div className="bg-panel px-4 py-3">
          <span className="block text-[10px] font-bold uppercase tracking-[0.08em] text-muted">Inbound weights</span>
          <strong className="numeric mt-1 block text-sm">{inbound ? inbound.weight_count.toLocaleString() : '—'}</strong>
          <span className="mt-0.5 block text-xs text-muted">{inbound ? `${inbound.input_size} × ${inbound.output_size}` : 'input layer'}</span>
        </div>
        <div className="bg-panel px-4 py-3">
          <span className="block text-[10px] font-bold uppercase tracking-[0.08em] text-muted">Weight RMS</span>
          <strong className="numeric mt-1 block text-sm">{inbound ? fmtNumber(inbound.weight_rms, 5) : '—'}</strong>
          <span className="numeric mt-0.5 block text-xs text-muted">{inbound ? `max |w| ${fmtNumber(inbound.weight_max_abs, 4)}` : 'checkpoint-derived'}</span>
        </div>
        <div className="bg-panel px-4 py-3">
          <span className="block text-[10px] font-bold uppercase tracking-[0.08em] text-muted">Branch parameters</span>
          <strong className="numeric mt-1 block text-sm">{branch.parameter_count?.toLocaleString() || '—'}</strong>
          <span className="mt-0.5 block text-xs text-muted">{branchName === 'actor' ? branch.distribution || 'policy' : 'value function'}</span>
        </div>
      </div>
    </div>
  )
}
