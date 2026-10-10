import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { LockKeyhole, LogOut, ShieldCheck } from 'lucide-react'
import { api } from '../api'
import { Button } from './ui/button'
import { Dialog, DialogContent, DialogDescription, DialogTitle } from './ui/dialog'
import { Input } from './ui/input'

export function ControlAccess() {
  const queryClient = useQueryClient()
  const [open, setOpen] = useState(false)
  const [token, setToken] = useState('')
  const session = useQuery({ queryKey: ['control-session'], queryFn: api.controlSession, staleTime: 15_000, refetchInterval: 60_000 })
  const unlock = useMutation({
    mutationFn: () => api.openControlSession(token),
    onSuccess: async () => {
      setToken('')
      setOpen(false)
      await queryClient.invalidateQueries({ queryKey: ['control-session'] })
    },
  })
  const lock = useMutation({
    mutationFn: api.closeControlSession,
    onSuccess: async () => { await queryClient.invalidateQueries({ queryKey: ['control-session'] }) },
  })
  const state = session.data
  const label = !state?.configured ? 'Read-only mode' : state.authenticated ? 'Controls unlocked' : 'Controls locked'

  return <>
    <div className="flex items-center gap-2">
      <span className={`hidden items-center gap-1.5 text-xs sm:flex ${state?.authenticated ? 'text-success' : state?.configured ? 'text-warning' : 'text-muted'}`} aria-live="polite">
        {state?.authenticated ? <ShieldCheck size={14} /> : <LockKeyhole size={14} />}{label}
      </span>
      {state?.authenticated ? <Button size="sm" variant="ghost" onClick={() => lock.mutate()} disabled={lock.isPending}><LogOut size={14} /> Lock</Button> : <Button size="sm" variant="secondary" onClick={() => setOpen(true)}><LockKeyhole size={14} /> {state && !state.configured ? 'Setup' : 'Unlock'}</Button>}
    </div>
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent className="max-w-md">
        <DialogTitle className="text-xl font-semibold">Unlock dashboard controls</DialogTitle>
        <DialogDescription className="mt-2 text-sm leading-relaxed text-muted">Enter the operator control token configured on the dashboard host. The token is exchanged for a short-lived, HttpOnly browser session and is not saved in browser storage.</DialogDescription>
        <form className="mt-5 space-y-4" onSubmit={(event) => { event.preventDefault(); unlock.mutate() }}>
          <label className="block text-xs font-bold uppercase tracking-[0.06em] text-muted">Control token
            <Input autoFocus type="password" autoComplete="current-password" className="mt-2" value={token} onChange={(event) => setToken(event.target.value)} />
          </label>
          {unlock.error ? <p className="rounded-lg border border-danger/35 bg-danger/10 p-3 text-sm text-danger" role="alert">{unlock.error.message}</p> : null}
          <div className="flex justify-end gap-2"><Button type="button" variant="ghost" onClick={() => setOpen(false)}>Cancel</Button><Button type="submit" variant="primary" disabled={!token || unlock.isPending}>{unlock.isPending ? 'Unlocking…' : 'Unlock controls'}</Button></div>
        </form>
        {!state?.configured ? <p className="mt-4 text-sm text-warning">Controls are disabled. Configure ASCENTO_CONTROL_TOKEN on the dashboard host with a value at least 32 characters long.</p> : null}
      </DialogContent>
    </Dialog>
  </>
}
