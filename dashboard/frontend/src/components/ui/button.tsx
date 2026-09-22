import * as React from 'react'
import { cva, type VariantProps } from 'class-variance-authority'
import { cn } from '../../lib/utils'

const buttonVariants = cva(
  'control-focus inline-flex min-h-10 items-center justify-center gap-2 rounded-lg border px-4 text-sm font-semibold transition-colors disabled:pointer-events-none disabled:opacity-40',
  {
    variants: {
      variant: {
        primary: 'border-foreground bg-foreground text-background hover:bg-[#dfe2e5]',
        secondary: 'border-border-strong bg-raised text-foreground hover:bg-hover',
        ghost: 'border-transparent bg-transparent text-secondary hover:bg-hover hover:text-foreground',
        danger: 'border-danger/45 bg-danger/10 text-[#ff9898] hover:bg-danger/15',
      },
      size: {
        default: 'h-10',
        sm: 'h-9 min-h-9 px-3 text-xs',
        lg: 'h-12 min-h-12 px-5 text-base',
        icon: 'h-10 min-h-10 w-10 px-0',
      },
    },
    defaultVariants: { variant: 'secondary', size: 'default' },
  },
)

export type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> &
  VariantProps<typeof buttonVariants>

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, ...props }, ref) => (
    <button ref={ref} className={cn(buttonVariants({ variant, size }), className)} {...props} />
  ),
)
Button.displayName = 'Button'

export { buttonVariants }
