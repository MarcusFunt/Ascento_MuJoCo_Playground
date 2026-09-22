import { useEffect } from 'react'
import { zodResolver } from '@hookform/resolvers/zod'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate } from '@tanstack/react-router'
import { useForm } from 'react-hook-form'
import { z } from 'zod'
import { api } from '../api'
import { Button } from './ui/button'
import { Dialog, DialogContent, DialogDescription, DialogTitle } from './ui/dialog'
import { Input, SelectInput, Textarea } from './ui/input'

const schema = z.object({
  display_name: z.string().trim().min(1, 'Give the run a name.'),
  task: z.string().min(1),
  purpose: z.string(),
  tags: z.string(),
  parent_run_id: z.string(),
  parent_checkpoint: z.string(),
  episode_horizon_s: z.number().optional(),
  notes: z.string(),
  num_envs: z.number().int().min(1).max(8192),
  max_iterations: z.number().int().min(1),
  seed: z.string(),
  extra_args: z.string(),
  allow_dirty_provenance: z.boolean(),
})

type FormValues = z.infer<typeof schema>

const defaults: FormValues = {
  display_name: '',
  task: 'Ascento-Balance-Flat',
  purpose: 'exploratory',
  tags: '',
  parent_run_id: '',
  parent_checkpoint: '',
  episode_horizon_s: 20,
  notes: '',
  num_envs: 512,
  max_iterations: 10_000,
  seed: '',
  extra_args: '',
  allow_dirty_provenance: false,
}

export function NewRunDialog({
  open,
  onOpenChange,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
}) {
  const queryClient = useQueryClient()
  const navigate = useNavigate()
  const tasks = useQuery({ queryKey: ['tasks'], queryFn: api.tasks, staleTime: 60_000 })
  const runs = useQuery({ queryKey: ['runs'], queryFn: api.runs, staleTime: 10_000 })
  const form = useForm<FormValues>({ resolver: zodResolver(schema), defaultValues: defaults })
  const taskId = form.watch('task')
  const task = tasks.data?.tasks.find((item) => item.id === taskId)

  useEffect(() => {
    if (!open) form.reset(defaults)
  }, [open, form])

  const create = useMutation({
    mutationFn: async (values: FormValues) => {
      const args = [
        '--env.scene.num-envs',
        String(values.num_envs),
        '--agent.max-iterations',
        String(values.max_iterations),
      ]
      if (values.seed.trim()) args.push('--agent.seed', values.seed.trim())
      if (values.extra_args.trim()) {
        args.push(...values.extra_args.split('\n').map((line) => line.trim()).filter(Boolean))
      }
      return api.createRun({
        display_name: values.display_name,
        task: values.task,
        purpose: values.purpose,
        tags: values.tags.split(',').map((tag) => tag.trim()).filter(Boolean),
        parent_run_id: values.parent_run_id || null,
        parent_checkpoint: values.parent_checkpoint || null,
        episode_horizon_s: task?.supports_horizon ? values.episode_horizon_s : null,
        notes: values.notes,
        training_args: args,
        allow_dirty_provenance: values.allow_dirty_provenance,
      })
    },
    onSuccess: async (created) => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['runs'] }),
        queryClient.invalidateQueries({ queryKey: ['overview'] }),
      ])
      onOpenChange(false)
      if (created.id) {
        await navigate({ to: '/runs/$runId', params: { runId: created.id } })
      }
    },
  })

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl">
        <DialogTitle className="text-2xl font-semibold tracking-[-0.03em]">Start training</DialogTitle>
        <DialogDescription className="mt-2 text-sm leading-relaxed text-muted">
          Create a managed run. The dashboard records source provenance, lineage, metadata and lifecycle state automatically.
        </DialogDescription>

        <form onSubmit={form.handleSubmit((values) => create.mutate(values))} className="mt-6 space-y-6">
          <div className="grid gap-4 md:grid-cols-2">
            <Field label="Human name" error={form.formState.errors.display_name?.message}>
              <Input autoFocus placeholder="Locomotion continuation from recovery 6250" {...form.register('display_name')} />
            </Field>
            <Field label="Task">
              <SelectInput {...form.register('task')}>
                {(tasks.data?.tasks || []).map((option) => (
                  <option key={option.id} value={option.id}>{option.label} · {option.id}</option>
                ))}
              </SelectInput>
              {task?.description ? <p className="mt-1 text-xs leading-relaxed text-muted">{task.description}</p> : null}
            </Field>
            <Field label="Purpose">
              <SelectInput {...form.register('purpose')}>
                {['exploratory', 'baseline', 'tuning', 'validation', 'regression'].map((value) => <option key={value}>{value}</option>)}
              </SelectInput>
            </Field>
            <Field label="Tags">
              <Input placeholder="locomotion, warm-start, recovery" {...form.register('tags')} />
            </Field>
            <Field label="Parallel environments" error={form.formState.errors.num_envs?.message}>
              <Input type="number" min={1} max={8192} {...form.register('num_envs', { valueAsNumber: true })} />
            </Field>
            <Field label="Maximum PPO iterations" error={form.formState.errors.max_iterations?.message}>
              <Input type="number" min={1} {...form.register('max_iterations', { valueAsNumber: true })} />
            </Field>
            {task?.supports_horizon ? (
              <Field label="Initial episode horizon">
                <SelectInput {...form.register('episode_horizon_s', { valueAsNumber: true })}>
                  <option value={20}>20 seconds · frequent reset practice</option>
                  <option value={60}>60 seconds</option>
                  <option value={120}>120 seconds</option>
                  <option value={300}>300 seconds · final protected stage</option>
                </SelectInput>
              </Field>
            ) : null}
            <Field label="Seed">
              <Input inputMode="numeric" placeholder="Optional" {...form.register('seed')} />
            </Field>
            <Field label="Parent run">
              <SelectInput {...form.register('parent_run_id')}>
                <option value="">None</option>
                {(runs.data?.runs || []).map((run) => <option key={run.id} value={run.id}>{run.display_name}</option>)}
              </SelectInput>
            </Field>
            <Field label="Parent checkpoint">
              <Input placeholder="model_6250.pt" {...form.register('parent_checkpoint')} />
            </Field>
          </div>

          <Field label="Notes">
            <Textarea rows={3} placeholder="What should this run prove or improve?" {...form.register('notes')} />
          </Field>

          <details className="rounded-lg border border-border bg-background/35 p-4">
            <summary className="cursor-pointer text-sm font-semibold">Advanced</summary>
            <div className="mt-4 space-y-4">
              <Field label="Additional argument tokens">
                <Textarea
                  rows={6}
                  className="font-mono text-xs"
                  placeholder={"--agent.learning-rate\n0.00001\n--agent.save-interval\n250"}
                  {...form.register('extra_args')}
                />
                <p className="mt-1 text-xs text-muted">One CLI token per line. Common environment count and iteration arguments are generated above.</p>
              </Field>
              <label className="flex items-start gap-3 text-sm text-secondary">
                <input className="mt-1" type="checkbox" {...form.register('allow_dirty_provenance')} />
                <span>Allow dirty source provenance. Use this only when intentionally archiving an uncommitted experiment.</span>
              </label>
            </div>
          </details>

          {create.error ? (
            <div className="rounded-lg border border-danger/40 bg-danger/10 px-4 py-3 text-sm text-[#ff9a9a]">
              {create.error.message}
            </div>
          ) : null}

          <div className="flex justify-end gap-3 border-t border-border pt-5">
            <Button type="button" variant="ghost" onClick={() => onOpenChange(false)}>Cancel</Button>
            <Button type="submit" variant="primary" disabled={create.isPending}>
              {create.isPending ? 'Starting…' : 'Start training'}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  )
}

function Field({
  label,
  error,
  children,
}: {
  label: string
  error?: string
  children: React.ReactNode
}) {
  return (
    <label className="block">
      <span className="mb-1.5 block text-xs font-bold uppercase tracking-[0.06em] text-muted">{label}</span>
      {children}
      {error ? <span className="mt-1 block text-xs text-danger">{error}</span> : null}
    </label>
  )
}
