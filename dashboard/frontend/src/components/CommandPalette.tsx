import { useEffect, useState } from 'react'
import { Command } from 'cmdk'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from '@tanstack/react-router'
import { Activity, ChartNoAxesCombined, Gauge, ListTree, Search, Settings } from 'lucide-react'
import { api } from '../api'
import { Dialog, DialogContent, DialogDescription, DialogTitle } from './ui/dialog'

export function CommandPalette() {
  const [open, setOpen] = useState(false)
  const navigate = useNavigate()
  const runs = useQuery({ queryKey: ['runs'], queryFn: api.runs, staleTime: 10_000 })

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault()
        setOpen((value) => !value)
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [])

  const go = async (to: string) => {
    setOpen(false)
    await navigate({ to })
  }

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        className="control-focus flex h-10 items-center gap-2 rounded-lg border border-border bg-raised px-3 text-sm text-secondary hover:bg-hover hover:text-foreground"
      >
        <Search size={16} />
        <span className="hidden sm:inline">Search</span>
        <kbd className="ml-2 hidden rounded border border-border-strong bg-background px-1.5 py-0.5 font-mono text-[10px] text-muted md:inline">Ctrl K</kbd>
      </button>

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="max-w-xl p-0">
          <DialogTitle className="sr-only">Command palette</DialogTitle>
          <DialogDescription className="sr-only">Navigate to runs and dashboard tools.</DialogDescription>
          <Command className="overflow-hidden rounded-xl bg-panel text-foreground">
            <div className="flex items-center gap-2 border-b border-border px-4">
              <Search size={17} className="text-muted" />
              <Command.Input
                autoFocus
                placeholder="Search runs or actions…"
                className="h-14 flex-1 bg-transparent text-[15px] outline-none placeholder:text-subtle"
              />
            </div>
            <Command.List className="max-h-[55vh] overflow-y-auto p-2">
              <Command.Empty className="px-3 py-8 text-center text-sm text-muted">Nothing found.</Command.Empty>
              <Command.Group heading="Navigation" className="[&_[cmdk-group-heading]]:px-3 [&_[cmdk-group-heading]]:py-2 [&_[cmdk-group-heading]]:text-[11px] [&_[cmdk-group-heading]]:font-bold [&_[cmdk-group-heading]]:uppercase [&_[cmdk-group-heading]]:tracking-[0.1em] [&_[cmdk-group-heading]]:text-muted">
                {[
                  ['Overview', '/', Gauge],
                  ['Runs', '/runs', ListTree],
                  ['Analyze', '/analyze', ChartNoAxesCombined],
                  ['System', '/system', Settings],
                ].map(([label, to, Icon]) => (
                  <Command.Item
                    key={String(to)}
                    value={String(label)}
                    onSelect={() => void go(String(to))}
                    className="flex cursor-pointer items-center gap-3 rounded-lg px-3 py-3 text-sm text-secondary outline-none data-[selected=true]:bg-hover data-[selected=true]:text-foreground"
                  >
                    <Icon size={16} /> {String(label)}
                  </Command.Item>
                ))}
              </Command.Group>
              <Command.Group heading="Runs" className="[&_[cmdk-group-heading]]:px-3 [&_[cmdk-group-heading]]:py-2 [&_[cmdk-group-heading]]:text-[11px] [&_[cmdk-group-heading]]:font-bold [&_[cmdk-group-heading]]:uppercase [&_[cmdk-group-heading]]:tracking-[0.1em] [&_[cmdk-group-heading]]:text-muted">
                {(runs.data?.runs || []).slice(0, 40).map((run) => (
                  <Command.Item
                    key={run.id}
                    value={`${run.display_name} ${run.task || ''} ${run.stage || ''}`}
                    onSelect={() => {
                      setOpen(false)
                      void navigate({ to: '/runs/$runId', params: { runId: run.id } })
                    }}
                    className="flex cursor-pointer items-center gap-3 rounded-lg px-3 py-3 text-sm text-secondary outline-none data-[selected=true]:bg-hover data-[selected=true]:text-foreground"
                  >
                    <Activity size={16} className={run.state === 'running' ? 'text-success' : 'text-muted'} />
                    <span className="min-w-0 flex-1 truncate">{run.display_name}</span>
                    <span className="text-xs text-muted">{run.state}</span>
                  </Command.Item>
                ))}
              </Command.Group>
            </Command.List>
          </Command>
        </DialogContent>
      </Dialog>
    </>
  )
}
