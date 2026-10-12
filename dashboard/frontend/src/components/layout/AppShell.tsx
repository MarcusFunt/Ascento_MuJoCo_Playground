import { Suspense, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Link, Outlet, useRouterState } from '@tanstack/react-router'
import {
  ChartNoAxesCombined,
  ChevronLeft,
  ChevronRight,
  Gauge,
  ListTree,
  Settings,
} from 'lucide-react'
import { api } from '../../api'
import { cn } from '../../lib/utils'
import { StateBadge } from '../StateBadge'
import { CommandPalette } from '../CommandPalette'

const navigation = [
  { label: 'Overview', to: '/' as const, icon: Gauge },
  { label: 'Runs', to: '/runs' as const, icon: ListTree },
  { label: 'Workspace', to: '/analyze' as const, icon: ChartNoAxesCombined },
  { label: 'System', to: '/system' as const, icon: Settings },
]

export function AppShell() {
  const [collapsed, setCollapsed] = useState(() => localStorage.getItem('ascento-sidebar-collapsed') === '1')
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
          {navigation.map((item) => {
            const activeRoute = item.to === '/' ? pathname === '/' : pathname.startsWith(item.to)
            const Icon = item.icon
            return (
              <Link
                key={item.to}
                to={item.to}
                className={cn(
                  'control-focus flex h-11 items-center rounded-lg px-3 text-sm font-medium transition-colors',
                  activeRoute ? 'bg-foreground text-background' : 'text-muted hover:bg-hover hover:text-foreground',
                  collapsed && 'justify-center px-0',
                )}
              >
                <Icon size={18} />
                {!collapsed ? <span className="ml-3">{item.label}</span> : null}
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
            {active ? (
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
                'control-focus flex min-h-12 min-w-0 flex-1 flex-col items-center justify-center gap-1 rounded-lg px-1 py-1 text-[10px] font-semibold leading-none transition-colors',
                activeRoute ? 'bg-hover text-foreground' : 'text-muted hover:bg-hover hover:text-foreground',
              )}
            >
              <Icon size={18} aria-hidden="true" />
              <span className="truncate">{item.label}</span>
            </Link>
          )
        })}
      </nav>
    </div>
  )
}
