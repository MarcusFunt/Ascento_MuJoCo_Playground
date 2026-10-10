import type { ReactNode } from 'react'

export function PageHeader({
  eyebrow,
  title,
  description,
  actions,
}: {
  eyebrow?: string
  title: string
  description?: string
  actions?: ReactNode
}) {
  return (
    <header className="mb-6 grid min-w-0 items-end gap-4 sm:mb-8 lg:grid-cols-[minmax(0,1fr)_auto]">
      <div className="min-w-0 max-w-3xl">
        {eyebrow ? <div className="text-xs font-bold uppercase tracking-[0.12em] text-muted">{eyebrow}</div> : null}
        <h1 className="mt-1 text-[30px] font-semibold leading-tight tracking-[-0.045em] sm:text-[38px]">{title}</h1>
        {description ? <p className="mt-3 text-[15px] leading-relaxed text-secondary">{description}</p> : null}
      </div>
      {actions ? <div className="flex w-full min-w-0 flex-wrap gap-2 lg:w-auto">{actions}</div> : null}
    </header>
  )
}
