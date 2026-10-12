import { Link } from '@tanstack/react-router'
import { PolicyArchitectureCard } from './PolicyArchitectureCard'
import { PolicyDebugger } from './PolicyDebugger'
import { usePolicyIntrospection } from '../hooks/usePolicyIntrospection'

/** Loaded only when the Debug workspace is open; never subscribes in the background. */
export function WorkspaceInspector({ runId, checkpointPath }: { runId: string; checkpointPath?: string }) {
  const inspector = usePolicyIntrospection(runId)
  return <div className="space-y-5">
    <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-border bg-panel px-4 py-3 text-sm text-secondary">
      <span>This debugger inspects a separate checkpoint viewer, not the live PPO training batch.</span>
      <Link to="/analyze/$runId" params={{ runId }} hash="visualize"
        className="control-focus rounded-md border border-border-strong px-3 py-2 font-semibold text-foreground hover:bg-hover">
        Manage checkpoint viewer
      </Link>
    </div>
    <PolicyDebugger
      viewer={inspector.viewer}
      schema={inspector.displaySchema}
      frame={inspector.displayFrame}
      liveFrame={inspector.frame}
      connected={inspector.connected}
      schemaError={inspector.schemaError as Error | null}
      captures={inspector.captures}
      capture={inspector.capture}
      selectedCaptureId={inspector.selectedCaptureId}
      selectedFrameIndex={inspector.selectedFrameIndex}
      selectCapture={inspector.selectCapture}
      selectFrame={inspector.selectFrame}
      onManualCapture={inspector.manualCapture}
      manualCapturePending={inspector.manualCapturePending}
      manualCaptureError={inspector.manualCaptureError as Error | null}
      captureLoading={inspector.captureLoading}
    />
    <PolicyArchitectureCard
      runId={runId}
      checkpointPath={checkpointPath}
      liveFrame={inspector.displayFrame}
      runtimeSchema={inspector.displaySchema}
    />
  </div>
}
