import { useState } from 'react'
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
  { label: 'Analyze', to: '/analyze' as const, icon: ChartNoAxesCombined },
  { label: 'System', to: '/system' as const, icon: Settings },
]

export function AppShell() {
  const [collapsed, setCollapsed] = useState(() => localStorage.getItem('ascento-sidebar-collapsed') === '1')
  const pathname = useRouterState({ select: (state) => state.location.pathname })
  const overview = useQuery({
    queryKey: ['overview'],
    queryFn: api.overview,
    refetchInterval: 5_000,
    staleTime: 2_000,
  })
  const active = overview.data?.active_run

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
        <header className="sticky top-0 z-30 flex h-16 items-center justify-between gap-4 border-b border-border bg-background/90 px-4 backdrop-blur-xl sm:px-6 lg:px-8">
          <div className="min-w-0">
            {active ? (
              <Link to="/runs/$runId" params={{ runId: active.id }} className="control-focus flex min-w-0 items-center gap-3 rounded">
                <StateBadge state={active.state} stale={active.stale} />
                <span className="hidden min-w-0 truncate text-sm font-semibold sm:block">{active.display_name}</span>
              </Link>
            ) : (
              <span className="text-sm text-muted">No active training</span>
            )}
          </div>
          <div className="flex items-center gap-2">
            <nav className="flex lg:hidden">
              {navigation.slice(0, 3).map((item) => (
                <Link key={item.to} to={item.to} className="control-focus rounded-lg p-2 text-muted hover:bg-hover hover:text-foreground" aria-label={item.label}>
                  <item.icon size={18} />
                </Link>
              ))}
            </nav>
            <CommandPalette />
          </div>
        </header>
        <main className="mx-auto w-full max-w-[1680px] px-4 py-7 sm:px-6 lg:px-8 lg:py-9">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
