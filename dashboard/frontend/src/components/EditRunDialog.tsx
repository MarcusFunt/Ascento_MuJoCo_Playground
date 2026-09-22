import { useEffect } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useForm } from 'react-hook-form'
import { api } from '../api'
import type { RunDetail, RunIndexRow } from '../types'
import { Button } from './ui/button'
import { Dialog, DialogContent, DialogDescription, DialogTitle } from './ui/dialog'
import { Input, SelectInput, Textarea } from './ui/input'

type Values = {
  display_name: string
  purpose: string
  tags: string
  parent_run_id: string
  parent_checkpoint: string
  notes: string
}

export function EditRunDialog({
  detail,
  runs,
  open,
  onOpenChange,
}: {
  detail: RunDetail
  runs: RunIndexRow[]
  open: boolean
  onOpenChange: (open: boolean) => void
}) {
  const queryClient = useQueryClient()
  const form = useForm<Values>()
  useEffect(() => {
    if (!open) return
    form.reset({
      display_name: detail.display_name || detail.name || '',
      purpose: String(detail.metadata?.purpose || ''),
      tags: (detail.tags || []).join(', '),
      parent_run_id: detail.lineage?.parent_run_id || '',
      parent_checkpoint: detail.lineage?.parent_checkpoint || '',
      notes: detail.notes || '',
    })
  }, [detail, form, open])

  const mutation = useMutation({
    mutationFn: (values: Values) =>
      api.updateRun(detail.id, {
        ...values,
        tags: values.tags.split(',').map((tag) => tag.trim()).filter(Boolean),
        parent_run_id: values.parent_run_id || null,
        parent_checkpoint: values.parent_checkpoint || null,
      }),
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['run', detail.id] }),
        queryClient.invalidateQueries({ queryKey: ['runs'] }),
        queryClient.invalidateQueries({ queryKey: ['overview'] }),
      ])
      onOpenChange(false)
    },
  })

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogTitle className="text-2xl font-semibold tracking-[-0.03em]">Run metadata</DialogTitle>
        <DialogDescription className="mt-2 text-sm text-muted">
          Change the human-facing description without moving or rewriting training artifacts.
        </DialogDescription>
        <form className="mt-6 space-y-4" onSubmit={form.handleSubmit((values) => mutation.mutate(values))}>
          <Field label="Human name"><Input {...form.register('display_name')} /></Field>
          <div className="grid gap-4 sm:grid-cols-2">
            <Field label="Purpose"><Input {...form.register('purpose')} /></Field>
            <Field label="Tags"><Input {...form.register('tags')} /></Field>
            <Field label="Parent run">
              <SelectInput {...form.register('parent_run_id')}>
                <option value="">None</option>
                {runs.filter((run) => run.id !== detail.id).map((run) => (
                  <option key={run.id} value={run.id}>{run.display_name}</option>
                ))}
              </SelectInput>
            </Field>
            <Field label="Parent checkpoint"><Input {...form.register('parent_checkpoint')} /></Field>
          </div>
          <Field label="Notes"><Textarea rows={4} {...form.register('notes')} /></Field>
          {mutation.error ? <div className="rounded-lg border border-danger/40 bg-danger/10 px-4 py-3 text-sm text-danger">{mutation.error.message}</div> : null}
          <div className="flex justify-end gap-3 border-t border-border pt-5">
            <Button type="button" variant="ghost" onClick={() => onOpenChange(false)}>Cancel</Button>
            <Button type="submit" variant="primary" disabled={mutation.isPending}>{mutation.isPending ? 'Saving…' : 'Save metadata'}</Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  )
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1.5 block text-xs font-bold uppercase tracking-[0.06em] text-muted">{label}</span>
      {children}
    </label>
  )
}
