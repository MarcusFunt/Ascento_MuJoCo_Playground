import * as React from 'react'
import { Tooltip as BaseTooltip } from '@base-ui/react/tooltip'
import { cn } from '../../lib/utils'

export const TooltipProvider = BaseTooltip.Provider
export const Tooltip = BaseTooltip.Root
export const TooltipTrigger = BaseTooltip.Trigger

export function TooltipContent({
  className,
  children,
}: {
  className?: string
  children: React.ReactNode
}) {
  return (
    <BaseTooltip.Portal>
      <BaseTooltip.Positioner sideOffset={8}>
        <BaseTooltip.Popup
          className={cn(
            'z-[80] max-w-xs rounded-lg border border-border-strong bg-raised px-3 py-2 text-xs leading-relaxed text-secondary shadow-xl shadow-black/40',
            className,
          )}
        >
          {children}
        </BaseTooltip.Popup>
      </BaseTooltip.Positioner>
    </BaseTooltip.Portal>
  )
}
