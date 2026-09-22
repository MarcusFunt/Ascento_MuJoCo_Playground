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
      <BaseDialog.Viewport className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto px-4 py-[8vh]">
        <BaseDialog.Popup
          className={cn(
            'relative w-full max-w-2xl rounded-xl border border-border-strong bg-panel p-6 shadow-2xl shadow-black/50 outline-none',
            className,
          )}
        >
          {children}
          <BaseDialog.Close
            aria-label="Close"
            className="control-focus absolute right-4 top-4 inline-flex h-9 w-9 items-center justify-center rounded-lg text-muted hover:bg-hover hover:text-foreground"
          >
            <X size={18} />
          </BaseDialog.Close>
        </BaseDialog.Popup>
      </BaseDialog.Viewport>
    </BaseDialog.Portal>
  )
}
