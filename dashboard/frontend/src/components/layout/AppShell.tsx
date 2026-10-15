import { Suspense, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Link, Outlet, useRouterState } from '@tanstack/react-router'
import {
  ChartNoAxesCombined,
  Microscope,
  ChevronLeft,
  ChevronRight,
  Gauge,
  ListTree,
  Settings,
  Radio,
} from 'lucide-react'
import { api } from '../../api'
import { cn } from '../../lib/utils'
import { StateBadge } from '../StateBadge'
import { useLiveTrainingSummary } from '../LiveTrainingSurface'
import { CommandPalette } from '../CommandPalette'

const navigation = [
  { label: 'Overview', to: '/' as const, icon: Gauge },
  { label: 'Runs', to: '/runs' as const, icon: ListTree },
  { label: 'Workspace', to: '/analyze' as const, icon: ChartNoAxesCombined },
  { label: 'Deep Analysis', to: '/deep-analysis' as const, icon: Microscope },
  { label: 'System', to: '/system' as const, icon: Settings },
]

export function AppShell() {
  const [collapsed, setCollapsed] = useState(() => localStorage.getItem('ascento-sidebar-collapsed') === '1')
  const [selectedLiveId, setSelectedLiveId] = useState('')
  const liveSummary = useLiveTrainingSummary()
  const liveRuns = liveSummary.data?.runs ?? []
  const selectedLive = liveRuns.find(run => run.id === selectedLiveId) ?? liveRuns[0]
  const liveHref = (id: string) => '/deep-analysis?live=' + encodeURIComponent(id)
  const pathname = useRouterState({ select: (state) => state.location.pathname })
  const isOverviewPage = pathname === '/'
  const overview = useQuery({
    queryKey: ['overview'],
    queryFn: api.overview,
    enabled: isOverviewPage,
    refetchInterval: 15_000,
    staleTime: 10_000,
  })
  const runs = useQuery({
    queryKey: ['runs'],
    queryFn: api.runs,
    enabled: !isOverviewPage,
    refetchInterval: 15_000,
    staleTime: 10_000,
  })
  const active = isOverviewPage
    ? overview.data?.active_run
    : runs.data?.runs.find((run) => ['starting', 'running', 'stopping'].includes(run.state))

  const toggle = () => {
    setCollapsed((value) => {
      localStorage.setItem('ascento-sidebar-collapsed', value ? '0' : '1')
      return !value
    })
  }

  return (
    <div className="min-h-screen bg-background text-foreground">
      <aside
        className={cn(
          'fixed inset-y-0 left-0 z-40 hidden border-r border-border bg-sidebar transition-[width] duration-200 lg:flex lg:flex-col',
          collapsed ? 'w-[72px]' : 'w-[230px]',
        )}
      >
        <div className={cn('flex h-16 items-center border-b border-border px-5', collapsed && 'justify-center px-0')}>
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-border-strong bg-raised text-[11px] font-black tracking-tight">A</div>
          {!collapsed ? <div className="ml-3 text-xs font-black tracking-[0.15em]">ASCENTO</div> : null}
        </div>

        <nav className="flex-1 space-y-1 p-3">
          {selectedLive && <div className="mb-3 rounded-xl border border-success/35 bg-success/5 p-2">
            <a href={liveHref(selectedLive.id)} title={selectedLive.label} className={cn('control-focus flex min-h-10 items-center gap-2 rounded-lg px-2 text-xs font-bold text-success hover:bg-hover', collapsed && 'justify-center')}>
              <Radio size={17} aria-hidden="true" />
              {!collapsed && <span className="min-w-0 flex-1 truncate">LIVE · {selectedLive.root.toUpperCase()}</span>}
            </a>
            {!collapsed && liveRuns.length > 1 && <select aria-label="Choose live training run" className="mt-1 w-full min-w-0 rounded-md border border-border bg-raised p-2 text-xs text-foreground" value={selectedLive.id}
              onChange={event => { setSelectedLiveId(event.target.value); window.location.assign(liveHref(event.target.value)) }}>
              {liveRuns.map(run => <option value={run.id} key={run.id}>{run.root.toUpperCase()} · {run.label.split('/').at(-1)}</option>)}
            </select>}
          </div>}
          {navigation.map((item) => {
            const activeRoute = item.to === '/' ? pathname === '/' : pathname.startsWith(item.to)
            const Icon = item.icon
            return (
              <Link
                key={item.to}
                to={item.to}
                className={cn(
                  'control-focus relative flex h-11 items-center rounded-lg px-3 text-sm font-medium transition-colors',
                  activeRoute ? 'bg-foreground text-background' : 'text-muted hover:bg-hover hover:text-foreground',
                  collapsed && 'justify-center px-0',
                )}
              >
                <Icon size={18} />
                {!collapsed ? <span className="ml-3">{item.label}</span> : null}
                {item.to === '/deep-analysis' && liveRuns.length > 0 && <span className={cn('rounded-full bg-success', collapsed ? 'absolute right-2 top-2 h-2 w-2' : 'ml-auto h-2 w-2')} aria-label="Live training active" />}
              </Link>
            )
          })}
        </nav>

        <div className="border-t border-border p-3">
          <button
            onClick={toggle}
            className={cn(
              'control-focus flex h-10 w-full items-center rounded-lg px-3 text-sm text-muted hover:bg-hover hover:text-foreground',
              collapsed && 'justify-center px-0',
            )}
            aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {collapsed ? <ChevronRight size={17} /> : <><ChevronLeft size={17} /><span className="ml-3">Collapse</span></>}
          </button>
        </div>
      </aside>

      <div className={cn('transition-[padding] duration-200', collapsed ? 'lg:pl-[72px]' : 'lg:pl-[230px]')}>
        <header className="sticky top-0 z-30 flex min-h-[calc(4rem_+_env(safe-area-inset-top))] items-center justify-between gap-3 border-b border-border bg-background/90 px-4 pt-[env(safe-area-inset-top)] backdrop-blur-xl sm:px-6 lg:px-8">
          <div className="flex min-w-0 flex-1 items-center">
            {selectedLive ? (
              <a href={liveHref(selectedLive.id)} title={selectedLive.label} className="control-focus flex min-h-11 min-w-0 items-center gap-2 rounded-lg text-xs font-semibold text-success sm:text-sm">
                <Radio size={17} className="shrink-0" /> <span className="shrink-0">LIVE</span>
                <span className="min-w-0 max-w-[44vw] truncate text-foreground sm:max-w-[28rem]">{selectedLive.label.split('/').slice(-2).join(' / ')}</span>
              </a>
            ) : active ? (
              <Link to="/analyze/$runId" params={{ runId: active.id }} hash="monitor"
                title="Open workspace for the active training run"
                className="control-focus flex min-h-11 min-w-0 items-center gap-2 rounded sm:gap-3">
                <StateBadge state={active.state} stale={active.stale} />
                <span className="min-w-0 max-w-[42vw] truncate text-xs font-semibold sm:max-w-[24rem] sm:text-sm">{active.display_name}</span>
              </Link>
            ) : (
              <span className="truncate text-sm text-muted">No active training</span>
            )}
          </div>
          <div className="flex items-center gap-2">
            {liveRuns.length > 1 && <select aria-label="Switch live training run" value={selectedLive?.id ?? ''}
              className="max-w-24 rounded-lg border border-border bg-raised p-2 text-xs sm:max-w-36"
              onChange={event => { setSelectedLiveId(event.target.value); window.location.assign(liveHref(event.target.value)) }}>
              {liveRuns.map(run => <option key={run.id} value={run.id}>{run.root.toUpperCase()} · {run.label.split('/').at(-1)}</option>)}
            </select>}
            <CommandPalette />
          </div>
        </header>
        <main className="mx-auto w-full max-w-[1680px] px-4 pt-6 pb-[calc(6rem_+_env(safe-area-inset-bottom))] sm:px-6 sm:pt-7 lg:px-8 lg:pt-9 lg:pb-9">
          <Suspense fallback={<div className="h-[420px] animate-pulse rounded-xl border border-border bg-panel" />}>
            <Outlet />
          </Suspense>
        </main>
      </div>

      <nav className="mobile-nav fixed inset-x-0 bottom-0 z-40 flex border-t border-border bg-sidebar/95 backdrop-blur-xl lg:hidden" aria-label="Primary navigation">
        {navigation.map((item) => {
          const activeRoute = item.to === '/' ? pathname === '/' : pathname.startsWith(item.to)
          const Icon = item.icon
          return (
            <Link
              key={item.to}
              to={item.to}
              aria-current={activeRoute ? 'page' : undefined}
              className={cn(
                'control-focus relative flex min-h-12 min-w-0 flex-1 flex-col items-center justify-center gap-1 rounded-lg px-1 py-1 text-[10px] font-semibold leading-none transition-colors',
                activeRoute ? 'bg-hover text-foreground' : 'text-muted hover:bg-hover hover:text-foreground',
              )}
            >
              <Icon size={18} aria-hidden="true" />
              <span className="truncate">{item.label}</span>
              {item.to === '/deep-analysis' && liveRuns.length > 0 && <span className="absolute right-3 top-2 h-2 w-2 rounded-full bg-success" aria-label="Live training active" />}
            </Link>
          )
        })}
      </nav>
    </div>
  )
}
