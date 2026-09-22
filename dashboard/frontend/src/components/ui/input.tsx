import * as React from 'react'
import { cn } from '../../lib/utils'

export const Input = React.forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(
  ({ className, ...props }, ref) => (
    <input
      ref={ref}
      className={cn(
        'control-focus h-11 w-full rounded-lg border border-border-strong bg-background px-3 text-[15px] text-foreground placeholder:text-subtle',
        className,
      )}
      {...props}
    />
  ),
)
Input.displayName = 'Input'

export const Textarea = React.forwardRef<HTMLTextAreaElement, React.TextareaHTMLAttributes<HTMLTextAreaElement>>(
  ({ className, ...props }, ref) => (
    <textarea
      ref={ref}
      className={cn(
        'control-focus w-full rounded-lg border border-border-strong bg-background px-3 py-2.5 text-[15px] text-foreground placeholder:text-subtle',
        className,
      )}
      {...props}
    />
  ),
)
Textarea.displayName = 'Textarea'

export const SelectInput = React.forwardRef<HTMLSelectElement, React.SelectHTMLAttributes<HTMLSelectElement>>(
  ({ className, ...props }, ref) => (
    <select
      ref={ref}
      className={cn(
        'control-focus h-11 w-full rounded-lg border border-border-strong bg-background px-3 text-[15px] text-foreground',
        className,
      )}
      {...props}
    />
  ),
)
SelectInput.displayName = 'SelectInput'
