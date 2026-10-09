import { lazy } from 'react'
import { createRootRoute, createRoute, createRouter } from '@tanstack/react-router'
import { AppShell } from './components/layout/AppShell'
import { NotFoundPage } from './pages/NotFoundPage'
import { OverviewPage } from './pages/OverviewPage'

const RunsPage = lazy(() => import('./pages/RunsPage').then((module) => ({ default: module.RunsPage })))
const RunDetailPage = lazy(() => import('./pages/RunDetailPage').then((module) => ({ default: module.RunDetailPage })))
const AnalyzePage = lazy(() => import('./pages/AnalyzePage').then((module) => ({ default: module.AnalyzePage })))
const SystemPage = lazy(() => import('./pages/SystemPage').then((module) => ({ default: module.SystemPage })))
const EvaluationsPage = lazy(() => import('./pages/EvaluationsPage').then((module) => ({ default: module.EvaluationsPage })))
const ExperimentsPage = lazy(() => import('./pages/ExperimentsPage').then((module) => ({ default: module.ExperimentsPage })))
const VisualizerPage = lazy(() => import('./pages/VisualizerPage').then((module) => ({ default: module.VisualizerPage })))
const CapturesPage = lazy(() => import('./pages/CapturesPage').then((module) => ({ default: module.CapturesPage })))

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

const systemRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/system',
  component: SystemPage,
})

const evaluationsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/evaluations',
  component: EvaluationsPage,
})

const evaluationDetailRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/evaluations/$evaluationId',
  component: EvaluationsPage,
})

const experimentsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/experiments',
  component: ExperimentsPage,
})

const experimentDetailRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/experiments/$experimentId',
  component: ExperimentsPage,
})

const visualizerRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/visualizer',
  component: VisualizerPage,
})

const visualizerRunRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/visualizer/$runId',
  component: VisualizerPage,
})

const capturesRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/captures',
  component: CapturesPage,
})

const routeTree = rootRoute.addChildren([
  overviewRoute,
  runsRoute,
  runDetailRoute,
  analyzeRoute,
  analyzeRunRoute,
  systemRoute,
  evaluationsRoute,
  evaluationDetailRoute,
  experimentsRoute,
  experimentDetailRoute,
  visualizerRoute,
  visualizerRunRoute,
  capturesRoute,
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
