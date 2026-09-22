import { cn } from '../lib/utils'

export function Sparkline({
  values,
  className,
}: {
  values: Array<number | null | undefined>
  className?: string
}) {
  const clean = values.map((value) => (Number.isFinite(Number(value)) ? Number(value) : null))
  const finite = clean.filter((value): value is number => value !== null)
  if (finite.length < 2) {
    return <div className={cn('h-12 rounded-md bg-raised/50', className)} />
  }
  const min = Math.min(...finite)
  const max = Math.max(...finite)
  const span = Math.max(max - min, Math.abs(max) * 0.02, 1e-9)
  const width = 240
  const height = 48
  let cursor = 0
  const segments: string[] = []
  clean.forEach((value, index) => {
    if (value === null) {
      cursor = index + 1
      return
    }
    const x = clean.length === 1 ? 0 : (index / (clean.length - 1)) * width
    const y = height - 4 - ((value - min) / span) * (height - 8)
    const command = index === cursor ? 'M' : 'L'
    segments.push(`${command}${x.toFixed(2)},${y.toFixed(2)}`)
  })
  return (
    <svg className={cn('h-12 w-full overflow-visible', className)} viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
      <path d={segments.join(' ')} fill="none" stroke="currentColor" strokeWidth="2" vectorEffect="non-scaling-stroke" />
    </svg>
  )
}
