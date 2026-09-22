import { useMemo, useState } from 'react'
import {
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  type ColumnDef,
  type SortingState,
  useReactTable,
} from '@tanstack/react-table'
import { useNavigate } from '@tanstack/react-router'
import { ArrowDown, ArrowUp, ArrowUpDown } from 'lucide-react'
import type { RunIndexRow } from '../types'
import { fmtNumber, fmtPercent, shortCommit } from '../lib/utils'
import { Progress } from './ui/progress'
import { StateBadge } from './StateBadge'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table'

function taskLabel(run: RunIndexRow): string {
  const source = run.task || run.stage || '—'
  return source.replace(/^Ascento-/, '').replace(/-Flat$/, '').replaceAll('-', ' ')
}

export function RunTable({
  runs,
  compareIds,
  onToggleCompare,
}: {
  runs: RunIndexRow[]
  compareIds: string[]
  onToggleCompare: (id: string) => void
}) {
  const navigate = useNavigate()
  const [sorting, setSorting] = useState<SortingState>([{ id: 'modified_at', desc: true }])

  const columns = useMemo<ColumnDef<RunIndexRow>[]>(() => [
    {
      id: 'compare',
      enableSorting: false,
      header: '',
      cell: ({ row }) => (
        <input
          aria-label={`Compare ${row.original.display_name}`}
          type="checkbox"
          checked={compareIds.includes(row.original.id)}
          onClick={(event) => event.stopPropagation()}
          onChange={() => onToggleCompare(row.original.id)}
        />
      ),
      size: 32,
    },
    {
      id: 'display_name',
      accessorFn: (run) => run.display_name,
      header: 'Run',
      cell: ({ row }) => (
        <div className="min-w-[250px] max-w-[430px]">
          <strong className="block truncate text-[15px] font-semibold text-foreground">{row.original.display_name}</strong>
          <span className="mt-1 block truncate font-mono text-[11px] text-subtle">{row.original.name}</span>
        </div>
      ),
    },
    {
      accessorKey: 'state',
      header: 'State',
      cell: ({ row }) => <StateBadge state={row.original.state} stale={row.original.stale} />,
    },
    {
      id: 'task',
      accessorFn: (run) => run.task || run.stage || '',
      header: 'Task',
      cell: ({ row }) => (
        <div className="min-w-[160px]">
          <span className="block text-sm capitalize text-secondary">{taskLabel(row.original)}</span>
          {row.original.tags?.length ? (
            <span className="mt-1 block truncate text-[11px] text-muted">{row.original.tags.slice(0, 2).join(' · ')}</span>
          ) : null}
        </div>
      ),
    },
    {
      accessorKey: 'iteration',
      header: 'Iteration',
      cell: ({ row }) => <span className="numeric text-foreground">{fmtNumber(row.original.iteration, 0)}</span>,
    },
    {
      accessorKey: 'percent_complete',
      header: 'Progress',
      cell: ({ row }) => (
        <div className="min-w-[115px]">
          <div className="numeric text-sm text-foreground">{fmtPercent(row.original.percent_complete, 1)}</div>
          <Progress value={Number(row.original.percent_complete || 0)} className="mt-2" />
        </div>
      ),
    },
    {
      accessorKey: 'reward',
      header: 'Reward',
      cell: ({ row }) => <span className="numeric text-foreground">{fmtNumber(row.original.reward, 4)}</span>,
    },
    {
      id: 'version',
      accessorFn: (run) => run.repository_version?.status || '',
      header: 'Version',
      cell: ({ row }) => (
        <div>
          <span className={row.original.repository_version?.is_outdated ? 'text-warning' : 'text-secondary'}>
            {row.original.repository_version?.status || 'unknown'}
          </span>
          <span className="mt-1 block font-mono text-[11px] text-muted">{shortCommit(row.original.repository_version?.run_commit)}</span>
        </div>
      ),
    },
    {
      accessorKey: 'modified_at',
      header: 'Modified',
      cell: ({ row }) => {
        const value = row.original.modified_at
        return <span className="whitespace-nowrap text-xs text-muted">{value ? new Date(value * 1000).toLocaleString() : '—'}</span>
      },
    },
  ], [compareIds, onToggleCompare])

  const table = useReactTable({
    data: runs,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  })

  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          {table.getHeaderGroups().map((group) => (
            <TableRow key={group.id} className="hover:bg-transparent">
              {group.headers.map((header) => (
                <TableHead key={header.id} style={{ width: header.getSize() }}>
                  {header.isPlaceholder ? null : header.column.getCanSort() ? (
                    <button
                      className="control-focus inline-flex items-center gap-1.5 rounded text-left hover:text-secondary"
                      onClick={header.column.getToggleSortingHandler()}
                    >
                      {flexRender(header.column.columnDef.header, header.getContext())}
                      {header.column.getIsSorted() === 'asc' ? <ArrowUp size={12} /> : header.column.getIsSorted() === 'desc' ? <ArrowDown size={12} /> : <ArrowUpDown size={12} className="opacity-45" />}
                    </button>
                  ) : flexRender(header.column.columnDef.header, header.getContext())}
                </TableHead>
              ))}
            </TableRow>
          ))}
        </TableHeader>
        <TableBody>
          {table.getRowModel().rows.map((row) => (
            <TableRow
              key={row.id}
              className="cursor-pointer"
              onClick={() => void navigate({ to: '/runs/$runId', params: { runId: row.original.id } })}
            >
              {row.getVisibleCells().map((cell) => (
                <TableCell key={cell.id}>{flexRender(cell.column.columnDef.cell, cell.getContext())}</TableCell>
              ))}
            </TableRow>
          ))}
          {runs.length === 0 ? (
            <TableRow className="hover:bg-transparent">
              <TableCell colSpan={columns.length} className="py-16 text-center text-muted">No runs match these filters.</TableCell>
            </TableRow>
          ) : null}
        </TableBody>
      </Table>
    </div>
  )
}
