import { lazy } from 'react'
import { createRootRoute, createRoute, createRouter } from '@tanstack/react-router'
import { AppShell } from './components/layout/AppShell'
import { NotFoundPage } from './pages/NotFoundPage'
import { OverviewPage } from './pages/OverviewPage'

const RunsPage = lazy(() => import('./pages/RunsPage').then((module) => ({ default: module.RunsPage })))
const RunDetailPage = lazy(() => import('./pages/RunDetailPage').then((module) => ({ default: module.RunDetailPage })))
const AnalyzePage = lazy(() => import('./pages/AnalyzePage').then((module) => ({ default: module.AnalyzePage })))
const DeepAnalysisPage = lazy(() => import('./pages/DeepAnalysisPage').then((module) => ({ default: module.DeepAnalysisPage })))
const SystemPage = lazy(() => import('./pages/SystemPage').then((module) => ({ default: module.SystemPage })))

const rootRoute = createRootRoute({
  component: AppShell,
  notFoundComponent: NotFoundPage,
})

const overviewRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/',
  component: OverviewPage,
})

const runsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/runs',
  component: RunsPage,
})

const runDetailRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/runs/$runId',
  component: RunDetailPage,
})

const analyzeRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/analyze',
  component: AnalyzePage,
})

const analyzeRunRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/analyze/$runId',
  component: AnalyzePage,
})

const deepAnalysisRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/deep-analysis',
  component: DeepAnalysisPage,
})

const systemRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/system',
  component: SystemPage,
})

const routeTree = rootRoute.addChildren([
  overviewRoute,
  runsRoute,
  runDetailRoute,
  analyzeRoute,
  analyzeRunRoute,
  deepAnalysisRoute,
  systemRoute,
])

export const router = createRouter({
  routeTree,
  defaultPreload: 'intent',
  defaultPreloadStaleTime: 5_000,
  scrollRestoration: true,
})

declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router
  }
}
