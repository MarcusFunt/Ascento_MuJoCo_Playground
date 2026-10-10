import { Suspense, useEffect, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Link, Outlet, useRouterState } from '@tanstack/react-router'
import {
  ChartNoAxesCombined,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  ClipboardCheck,
  Eye,
  Film,
  FlaskConical,
  Gauge,
  ListTree,
  Menu,
  Settings,
  X,
} from 'lucide-react'
import { api } from '../../api'
import { cn } from '../../lib/utils'
import { StateBadge } from '../StateBadge'
import { CommandPalette } from '../CommandPalette'
import { ControlAccess } from '../ControlAccess'

const navigation = [
  { label: 'Overview', to: '/' as const, icon: Gauge },
  { label: 'Experiments', to: '/experiments' as const, icon: FlaskConical },
  { label: 'Evaluations', to: '/evaluations' as const, icon: ClipboardCheck },
  { label: 'Visualizer', to: '/visualizer' as const, icon: Eye },
  { label: 'Captures', to: '/captures' as const, icon: Film },
  { label: 'Runs', to: '/runs' as const, icon: ListTree },
  { label: 'Analyze', to: '/analyze' as const, icon: ChartNoAxesCombined },
  { label: 'System', to: '/system' as const, icon: Settings },
]

function currentNav(pathname: string, to: string) {
  return to === '/' ? pathname === '/' : pathname === to || pathname.startsWith(to + '/')
}

export function AppShell() {
  const [collapsed, setCollapsed] = useState(() => localStorage.getItem('ascento-sidebar-collapsed') === '1')
  const [mobileOpen, setMobileOpen] = useState(false)
  const pathname = useRouterState({ select: (state) => state.location.pathname })

  useEffect(() => setMobileOpen(false), [pathname])
  useEffect(() => {
    if (!mobileOpen) return
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setMobileOpen(false)
    }
    window.addEventListener('keydown', onKeyDown)
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      window.removeEventListener('keydown', onKeyDown)
      document.body.style.overflow = previousOverflow
    }
  }, [mobileOpen])

  const overview = useQuery({
    queryKey: ['overview'],
    queryFn: api.overview,
    refetchInterval: 15_000,
    staleTime: 2_000,
  })
  const active = overview.data?.active_run
  const health = useQuery({
    queryKey: ['health'],
    queryFn: api.health,
    refetchInterval: 30_000,
  })
  const systemHealth = health.data
  const unhealthyComponents = Object.entries(systemHealth?.components || {})
    .filter(([, component]) => component.status === 'degraded' || component.status === 'unavailable')
    .map(([name]) => name)
  const healthOk = !health.isError && systemHealth?.status === 'healthy'
  const healthTone = healthOk
    ? 'border-success/20 bg-success/[0.035] text-success'
    : systemHealth?.status === 'unavailable' || health.isError
      ? 'border-danger/40 bg-danger/10 text-danger'
      : 'border-warning/35 bg-warning/5 text-warning'
  const activeNav = navigation.find((item) => currentNav(pathname, item.to))

  const toggle = () => setCollapsed((value) => {
    localStorage.setItem('ascento-sidebar-collapsed', value ? '0' : '1')
    return !value
  })

  const links = (mobile: boolean) => navigation.map((item) => {
    const selected = currentNav(pathname, item.to)
    const Icon = item.icon
    return (
      <Link
        key={item.to}
        to={item.to}
        onClick={mobile ? () => setMobileOpen(false) : undefined}
        aria-current={selected ? 'page' : undefined}
        title={!mobile && collapsed ? item.label : undefined}
        className={cn(
          'control-focus flex min-h-11 items-center gap-3 rounded-lg px-3 text-sm font-medium transition-colors',
          selected ? 'bg-foreground text-background' : 'text-secondary hover:bg-hover hover:text-foreground',
          !mobile && collapsed && 'justify-center gap-0 px-0',
        )}
      >
        <Icon size={19} aria-hidden="true" />
        {(mobile || !collapsed) ? <span>{item.label}</span> : null}
      </Link>
    )
  })

  return (
    <div className="min-h-screen min-w-0 bg-background text-foreground">
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
        <nav aria-label="Primary navigation" className="flex-1 space-y-1 overflow-y-auto p-3">{links(false)}</nav>
        <div className="border-t border-border p-3">
          <button onClick={toggle} className={cn('control-focus flex h-10 w-full items-center rounded-lg px-3 text-sm text-secondary hover:bg-hover', collapsed && 'justify-center px-0')} aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}>
            {collapsed ? <ChevronRight size={17} /> : <><ChevronLeft size={17} /><span className="ml-3">Collapse</span></>}
          </button>
        </div>
      </aside>

      {mobileOpen ? (
        <div className="fixed inset-0 z-50 lg:hidden" role="presentation">
          <button
            className="absolute inset-0 bg-black/75"
            onClick={() => setMobileOpen(false)}
            aria-label="Close navigation"
            tabIndex={-1}
          />
          <aside id="mobile-navigation" role="dialog" aria-modal="true" aria-label="Navigation" className="absolute inset-y-0 left-0 flex w-[min(310px,calc(100vw-3rem))] flex-col border-r border-border bg-sidebar shadow-2xl">
            <div className="flex h-16 items-center justify-between border-b border-border px-4">
              <span className="text-sm font-bold tracking-[0.12em]">ASCENTO CONTROL</span>
              <button autoFocus onClick={() => setMobileOpen(false)} className="control-focus inline-flex h-10 w-10 items-center justify-center rounded-lg text-secondary hover:bg-hover" aria-label="Close menu"><X size={20} /></button>
            </div>
            <nav aria-label="Mobile navigation" className="flex-1 space-y-1 overflow-y-auto p-3">{links(true)}</nav>
            <div className="border-t border-border px-4 py-3 text-xs text-muted">Robot training and evaluation workstation</div>
          </aside>
        </div>
      ) : null}

      <div className={cn('min-w-0 transition-[padding] duration-200', collapsed ? 'lg:pl-[72px]' : 'lg:pl-[230px]')}>
        <header className="sticky top-0 z-30 flex h-16 min-w-0 items-center justify-between gap-2 border-b border-border bg-background/95 px-3 backdrop-blur-xl sm:px-6 lg:px-8">
          <div className="flex min-w-0 items-center gap-2 sm:gap-3">
            <button type="button" onClick={() => setMobileOpen(true)} className="control-focus flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border border-border-strong text-secondary hover:bg-hover lg:hidden" aria-label="Open navigation" aria-expanded={mobileOpen} aria-controls="mobile-navigation"><Menu size={20} /></button>
            <span className="min-w-0 truncate text-sm font-semibold lg:hidden">{activeNav?.label || 'Ascento'}</span>
            <div className="hidden min-w-0 items-center gap-2 lg:flex">
              {active ? (
                <Link to="/runs/$runId" params={{ runId: active.id }} className="control-focus flex min-w-0 items-center gap-3 rounded">
                  <StateBadge state={active.state} stale={active.stale} />
                  <span className="hidden min-w-0 truncate text-sm font-medium text-secondary xl:block">{active.display_name}</span>
                </Link>
              ) : <span className="truncate text-sm text-secondary">{systemHealth?.components?.supervisor?.status === 'unavailable' ? 'Training state unverified' : systemHealth ? 'Training idle' : 'Checking trainer…'}</span>}
            </div>
          </div>
          <div className="flex shrink-0 items-center gap-1.5 sm:gap-2">
            <ControlAccess />
            <CommandPalette />
          </div>
        </header>
        <div className={cn('flex min-w-0 flex-wrap items-center justify-between gap-2 border-b px-3 py-2 text-xs sm:px-6 lg:px-8', healthTone)} role="status" aria-live="polite">
          <div className="flex min-w-0 flex-wrap items-center gap-x-3 gap-y-1">
            {healthOk ? <CheckCircle2 size={14} aria-hidden="true" /> : null}
            <strong className="uppercase tracking-[0.08em]">{health.isError ? 'Health check unavailable' : systemHealth?.status || 'Checking health'}</strong>
            {unhealthyComponents.length ? <span>Needs attention: {unhealthyComponents.join(', ')}</span> : null}
            {healthOk ? <span className="hidden sm:inline">Dashboard services operational</span> : null}
          </div>
          <Link to="/system" className="control-focus rounded font-semibold underline-offset-2 hover:underline">{healthOk ? 'Details' : 'System details'}</Link>
        </div>
        <main id="main-content" className="mx-auto w-full min-w-0 max-w-[1680px] px-4 py-5 sm:px-6 sm:py-7 lg:px-8 lg:py-9">
          <Suspense fallback={<div className="h-[420px] animate-pulse rounded-xl border border-border bg-panel" />}>
            <Outlet />
          </Suspense>
        </main>
      </div>
    </div>
  )
}
