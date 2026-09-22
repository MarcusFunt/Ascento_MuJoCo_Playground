import { Link } from '@tanstack/react-router'
import { Button } from '../components/ui/button'

export function NotFoundPage() {
  return (
    <div className="rounded-xl border border-border bg-panel p-12 text-center">
      <div className="text-xs font-bold uppercase tracking-[0.12em] text-muted">404</div>
      <h1 className="mt-2 text-3xl font-semibold tracking-[-0.04em]">Page not found</h1>
      <p className="mt-3 text-sm text-muted">The requested dashboard route does not exist.</p>
      <Link to="/" className="mt-6 inline-block"><Button variant="primary">Back to overview</Button></Link>
    </div>
  )
}
