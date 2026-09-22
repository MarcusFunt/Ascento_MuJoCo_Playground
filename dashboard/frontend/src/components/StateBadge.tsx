import { Badge } from './ui/badge'
import { cn } from '../lib/utils'

const tones: Record<string, string> = {
  running: 'border-success/45 bg-success/10 text-[#79ddb3]',
  starting: 'border-success/35 bg-success/5 text-[#79ddb3]',
  stopping: 'border-warning/45 bg-warning/10 text-[#f0c473]',
  finished: 'border-border-strong bg-raised text-secondary',
  stopped: 'border-border-strong bg-raised text-secondary',
  error: 'border-danger/50 bg-danger/10 text-[#ff9292]',
  stale: 'border-warning/45 bg-warning/10 text-[#f0c473]',
}

export function StateBadge({ state, stale = false }: { state?: string | null; stale?: boolean }) {
  const value = stale ? 'stale' : String(state || 'unknown')
  return <Badge className={cn(tones[value] || '')}>{value}</Badge>
}
