import * as React from 'react'
import { Dialog as BaseDialog } from '@base-ui/react/dialog'
import { X } from 'lucide-react'
import { cn } from '../../lib/utils'

export const Dialog = BaseDialog.Root
export const DialogTrigger = BaseDialog.Trigger
export const DialogTitle = BaseDialog.Title
export const DialogDescription = BaseDialog.Description
export const DialogClose = BaseDialog.Close

export function DialogContent({
  className,
  children,
}: {
  className?: string
  children: React.ReactNode
}) {
  return (
    <BaseDialog.Portal>
      <BaseDialog.Backdrop className="fixed inset-0 z-50 bg-black/70 backdrop-blur-[2px] transition-opacity" />
      <BaseDialog.Viewport className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto overscroll-contain px-3 py-4 [padding-top:max(1rem,env(safe-area-inset-top))] [padding-bottom:max(1rem,env(safe-area-inset-bottom))] sm:px-4 sm:py-[8vh]">
        <BaseDialog.Popup
          className={cn(
            'relative max-h-[calc(100dvh_-_2rem_-_env(safe-area-inset-top)_-_env(safe-area-inset-bottom))] w-full max-w-2xl overflow-y-auto overscroll-contain rounded-xl border border-border-strong bg-panel p-4 shadow-2xl shadow-black/50 outline-none sm:max-h-[84vh] sm:p-6',
            className,
          )}
        >
          {children}
          <BaseDialog.Close
            aria-label="Close"
            className="control-focus absolute right-3 top-3 inline-flex h-11 w-11 items-center justify-center rounded-lg text-muted hover:bg-hover hover:text-foreground sm:right-4 sm:top-4"
          >
            <X size={18} />
          </BaseDialog.Close>
        </BaseDialog.Popup>
      </BaseDialog.Viewport>
    </BaseDialog.Portal>
  )
}
