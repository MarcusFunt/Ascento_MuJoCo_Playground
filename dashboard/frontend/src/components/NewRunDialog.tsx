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
  experiment_id: z.string(),
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
  max_speed_mps: z.string(),
}).superRefine((values, context) => {
  if (values.task !== 'Ascento-Locomotion-Speed-Flat') return
  const speed = Number(values.max_speed_mps)
  if (!values.max_speed_mps.trim() || !Number.isFinite(speed) || speed <= 0) {
    context.addIssue({
      code: z.ZodIssueCode.custom,
      path: ['max_speed_mps'],
      message: 'Enter a positive maximum speed for the selectable range.',
    })
  }
})

type FormValues = z.infer<typeof schema>

const defaults: FormValues = {
  display_name: '',
  task: 'Ascento-Balance-Flat',
  purpose: 'exploratory',
  experiment_id: '',
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
  max_speed_mps: '',
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
  const experiments = useQuery({ queryKey: ['experiments'], queryFn: api.experiments, enabled: open, staleTime: 60_000 })
  const controlSession = useQuery({ queryKey: ['control-session'], queryFn: api.controlSession, enabled: open, staleTime: 15_000 })
  const form = useForm<FormValues>({ resolver: zodResolver(schema), defaultValues: defaults })
  const taskId = form.watch('task')
  const parentRunId = form.watch('parent_run_id')
  const parentCheckpoint = form.watch('parent_checkpoint')
  const extraArgsText = form.watch('extra_args')
  const requestedDevice = deviceFromAdvancedArgs(extraArgsText)
  const task = tasks.data?.tasks.find((item) => item.id === taskId)
  const parentCheckpoints = useQuery({
    queryKey: ['run-checkpoints', parentRunId],
    queryFn: () => api.checkpoints(parentRunId),
    enabled: open && Boolean(parentRunId),
    staleTime: 15_000,
  })
  const checkpointCompatibility = useQuery({
    queryKey: ['checkpoint-compatibility', parentRunId, parentCheckpoint, taskId],
    queryFn: () => api.checkpointCompatibility(parentRunId, parentCheckpoint, taskId),
    enabled: open && Boolean(parentRunId && parentCheckpoint && taskId),
    staleTime: 60_000,
    retry: false,
  })
  const stableParentCheckpoints = (parentCheckpoints.data?.checkpoints || []).filter((item) => item.stable === true)
  const preflight = useQuery({
    queryKey: ['runtime-preflight', taskId, requestedDevice],
    queryFn: () => api.runtimePreflight(taskId, requestedDevice),
    enabled: open && Boolean(taskId),
    staleTime: 0,
    refetchOnWindowFocus: false,
  })

  useEffect(() => {
    if (!open) form.reset(defaults)
  }, [open, form])

  useEffect(() => {
    form.setValue('parent_checkpoint', '')
  }, [form, parentRunId])

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
        experiment_id: values.experiment_id || null,
        tags: values.tags.split(',').map((tag) => tag.trim()).filter(Boolean),
        parent_run_id: values.parent_run_id || null,
        parent_checkpoint: values.parent_checkpoint || null,
        episode_horizon_s: task?.supports_horizon ? values.episode_horizon_s : null,
        max_speed_mps: task?.supports_speed_command ? Number(values.max_speed_mps) : null,
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
      <DialogContent className="max-h-[90vh] max-w-3xl overflow-y-auto">
        <DialogTitle className="text-2xl font-semibold tracking-[-0.03em]">Start training</DialogTitle>
        <DialogDescription className="mt-2 text-sm leading-relaxed text-muted">
          Create a managed run. The dashboard records source provenance, lineage, metadata and lifecycle state automatically.
        </DialogDescription>

        <form onSubmit={form.handleSubmit((values) => create.mutate(values))} className="mt-6 space-y-6">
          <section className="space-y-3">
            <div><h2 className="text-sm font-semibold">1 · Task and curriculum</h2><p className="mt-1 text-xs text-muted">Choose the registered task; its declared capabilities control the curriculum fields below.</p></div>
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
            {task?.supports_speed_command ? (
              <Field label="Maximum selectable speed (m/s)" error={form.formState.errors.max_speed_mps?.message}>
                <Input type="number" min={0.01} step={0.05} placeholder="Choose a validated training cap" {...form.register('max_speed_mps')} />
                <p className="mt-1 text-xs leading-relaxed text-muted">
                  One policy will train across 0 to this cap; the Viser slider selects speed at runtime.
                  Choose a conservative cap and validate it before relying on the upper end.
                </p>
              </Field>
            ) : null}
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
          </div>
          </section>

          <section className="space-y-3 border-t border-border pt-5">
            <div><h2 className="text-sm font-semibold">2 · Experiment and lineage</h2><p className="mt-1 text-xs text-muted">Link this run to a declared experiment and optional parent attempt/checkpoint.</p></div>
            <div className="grid gap-4 md:grid-cols-2">
              <Field label="Experiment plan">
                <SelectInput {...form.register('experiment_id')}>
                  <option value="">Unlinked run</option>
                  {(experiments.data?.programs || []).map((item) => <option key={item.id} value={item.id}>{item.title} · {item.id}</option>)}
                </SelectInput>
                {experiments.isError ? <p className="mt-1 text-xs text-warning">Experiment list unavailable: {experiments.error.message}</p> : null}
              </Field>
              <Field label="Purpose">
                <SelectInput {...form.register('purpose')}>
                  {['exploratory', 'baseline', 'tuning', 'validation', 'regression'].map((value) => <option key={value}>{value}</option>)}
                </SelectInput>
              </Field>
              <Field label="Tags">
                <Input placeholder="locomotion, warm-start, recovery" {...form.register('tags')} />
              </Field>
              <Field label="Parent run">
                <SelectInput {...form.register('parent_run_id')}>
                  <option value="">None</option>
                  {(runs.data?.runs || []).map((run) => <option key={run.id} value={run.id}>{run.display_name}</option>)}
                </SelectInput>
              </Field>
              <Field label="Parent checkpoint">
                <SelectInput {...form.register('parent_checkpoint')} disabled={!parentRunId || parentCheckpoints.isLoading}>
                  <option value="">No checkpoint selected</option>
                  {stableParentCheckpoints.map((checkpoint) => <option key={checkpoint.relative_path} value={checkpoint.relative_path}>
                    {checkpoint.relative_path}{checkpoint.iteration == null ? '' : ` · iteration ${checkpoint.iteration}`}
                  </option>)}
                </SelectInput>
              </Field>
            </div>
            {parentRunId && parentCheckpoints.isError ? <p className="text-xs text-danger">Could not list parent checkpoints: {parentCheckpoints.error.message}</p> : null}
            {parentCheckpoint ? (
              <div className={`rounded-lg border p-3 text-sm ${checkpointCompatibility.data?.compatible ? 'border-success/30 bg-success/5 text-success' : checkpointCompatibility.isError || checkpointCompatibility.data ? 'border-danger/35 bg-danger/5 text-danger' : 'border-border bg-background/35 text-muted'}`} aria-live="polite">
                {checkpointCompatibility.isFetching ? 'Checking the exact checkpoint hash and plant/action/task contracts…' : null}
                {checkpointCompatibility.data ? <>{checkpointCompatibility.data.status} · {checkpointCompatibility.data.reason} · SHA-256 {checkpointCompatibility.data.checkpoint_sha256.slice(0, 12)}</> : null}
                {checkpointCompatibility.isError ? `Compatibility could not be verified: ${checkpointCompatibility.error.message}` : null}
              </div>
            ) : null}
          </section>

          <section className="space-y-3 border-t border-border pt-5">
            <div><h2 className="text-sm font-semibold">3 · Compute and preflight</h2><p className="mt-1 text-xs text-muted">Resources, seed, optional CLI arguments, and canonical runtime readiness.</p></div>

          <div className="grid gap-4 md:grid-cols-3">
            <Field label="Parallel environments" error={form.formState.errors.num_envs?.message}>
              <Input type="number" min={1} max={8192} {...form.register('num_envs', { valueAsNumber: true })} />
            </Field>
            <Field label="Maximum PPO iterations" error={form.formState.errors.max_iterations?.message}>
              <Input type="number" min={1} {...form.register('max_iterations', { valueAsNumber: true })} />
            </Field>
            <Field label="Seed">
              <Input inputMode="numeric" placeholder="Optional" {...form.register('seed')} />
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
          </section>

          <section className="space-y-3 border-t border-border pt-5">
            <div><h2 className="text-sm font-semibold">4 · Review and start</h2><p className="mt-1 text-xs text-muted">Confirm the exact task, source revision, device, and run lineage before launch.</p></div>

          <div className="grid gap-2 rounded-lg border border-border bg-background/35 p-4 text-sm sm:grid-cols-2">
            <ReviewItem label="Run" value={form.watch('display_name') || 'Unnamed run'} />
            <ReviewItem label="Task" value={task?.id || taskId} />
            <ReviewItem label="Experiment" value={form.watch('experiment_id') || 'Unlinked'} />
            <ReviewItem label="Purpose" value={form.watch('purpose') || 'Unspecified'} />
            <ReviewItem label="Parent" value={form.watch('parent_run_id') || 'None'} />
            <ReviewItem label="Compute" value={`${form.watch('num_envs')} environments · ${form.watch('max_iterations')} iterations · ${requestedDevice}`} />
          </div>

          <section className={`rounded-lg border p-4 ${preflight.data?.allowed ? 'border-success/30 bg-success/5' : 'border-warning/35 bg-warning/5'}`} aria-live="polite">
            <div className="text-xs font-bold uppercase tracking-[0.08em] text-muted">Canonical runtime preflight</div>
            {preflight.isFetching ? <p className="mt-2 text-sm text-secondary">Checking source revision, checkout state and {requestedDevice} availability…</p> : null}
            {preflight.data?.allowed ? (
              <p className="mt-2 break-words text-sm text-success">
                Ready · {String(preflight.data.runtime.source_branch || 'source')} @ {String(preflight.data.runtime.source_commit || 'unknown commit').slice(0, 12)} · {String(preflight.data.runtime.device || requestedDevice)}
              </p>
            ) : null}
            {preflight.data && !preflight.data.allowed ? (
              <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-warning">{preflight.data.blockers.map((blocker) => <li key={blocker}>{blocker}</li>)}</ul>
            ) : null}
            {preflight.isError ? <p className="mt-2 text-sm text-danger">Preflight could not complete: {preflight.error.message}</p> : null}
            {!preflight.data && !preflight.isFetching && !preflight.isError ? <p className="mt-2 text-sm text-muted">Open this dialog to check the selected task and device before launch.</p> : null}
          </section>
          </section>

          {create.error ? (
            <div className="rounded-lg border border-danger/40 bg-danger/10 px-4 py-3 text-sm text-[#ff9a9a]">
              {create.error.message}
            </div>
          ) : null}

          <div className="flex justify-end gap-3 border-t border-border pt-5">
            <Button type="button" variant="ghost" onClick={() => onOpenChange(false)}>Cancel</Button>
            <Button type="submit" variant="primary" disabled={create.isPending || preflight.isFetching || !preflight.data?.allowed || !controlSession.data?.authenticated || Boolean(parentCheckpoint && !checkpointCompatibility.data?.compatible)}>
              {create.isPending ? 'Starting…' : !controlSession.data?.authenticated ? 'Unlock controls to start' : parentCheckpoint && !checkpointCompatibility.data?.compatible ? 'Resolve checkpoint compatibility to start' : preflight.data?.allowed ? 'Start training' : 'Resolve preflight to start'}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  )
}

function deviceFromAdvancedArgs(raw: string): string {
  const args = raw.split(/\r?\n/).map((value) => value.trim()).filter(Boolean)
  const names = ['--device', '--env.device', '--agent.device']
  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index]
    if (names.includes(argument)) return args[index + 1] || 'cuda:0'
    const option = names.find((name) => argument.startsWith(`${name}=`))
    if (option) return argument.slice(option.length + 1)
  }
  return 'cuda:0'
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

function ReviewItem({ label, value }: { label: string; value: string }) {
  return <div className="min-w-0"><span className="block text-[10px] font-bold uppercase tracking-[0.07em] text-muted">{label}</span><strong className="mt-1 block break-all text-secondary">{value}</strong></div>
}
