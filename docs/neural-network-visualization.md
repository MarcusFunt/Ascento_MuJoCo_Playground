# Neural network visualization

The run-detail dashboard includes an animated visualization of the checkpoint-backed
actor and critic MLPs.

## Research basis

The interaction and visual vocabulary are intentionally based on proven open-source
neural-network viewers rather than inventing a graph style from scratch:

- **TensorFlow Playground** (Apache-2.0) uses an SVG network diagram with neurons
  arranged by layer and weighted links between them. The Ascento implementation
  adapts that layered-neuron visual model to large PPO MLPs and React.
  Source: https://github.com/tensorflow/playground
- **TensorSpace** (Apache-2.0) demonstrates animated/intermediate neural-network
  visualizations in the browser, but its Three.js/TensorFlow.js stack is much
  heavier than this dashboard needs.
  Source: https://github.com/tensorspace-team/tensorspace
- **Netron** (MIT) is excellent for precise static model inspection, but it is
  primarily a topology/model-file viewer rather than an animated control-room
  visualization.
  Source: https://github.com/lutzroeder/netron

TensorFlow Playground was the closest fit. Its original implementation is D3 v3 /
TypeScript 2.x and assumes very small educational networks, so its source is not
vendored directly. `AnimatedPolicyNetwork.tsx` reimplements the same layered SVG
concept in React 18 and samples wide layers such as 256-unit MLP layers.

The component source carries the TensorFlow Playground attribution.

## What is real versus schematic

The dashboard reads the actual stable checkpoint and reports:

- actor and critic layer widths,
- activation function,
- policy distribution metadata,
- total parameter count,
- per-linear-layer weight count,
- weight RMS, mean absolute value, maximum absolute value, and bias RMS.

The diagram renders representative neurons when a layer is too wide to display
literally. Base connection density and transition emphasis are derived from the
checkpoint's weight summaries.

The moving dashed traces are deliberately **schematic feed-forward motion**. They
show signal direction and network depth but do not claim to be live individual
neuron activations. The UI says this explicitly.

## Animation and accessibility

- Actor and critic branches can be switched without leaving the page.
- Each displayed layer can be clicked or keyboard-selected to inspect its inbound
  parameter statistics.
- Animation can be paused and switched between 0.5x, 1x, and 2x.
- `prefers-reduced-motion: reduce` disables the continuous animation.
- The SVG retains an accessible architecture description even when animation is
  disabled.
