import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function fmtNumber(value: unknown, digits = 2): string {
  const number = Number(value)
  if (value === null || value === undefined || !Number.isFinite(number)) return '—'
  return number.toLocaleString(undefined, { maximumFractionDigits: digits })
}

export function fmtPercent(value: unknown, digits = 1): string {
  const number = Number(value)
  if (value === null || value === undefined || !Number.isFinite(number)) return '—'
  return `${number.toFixed(digits)}%`
}

export function fmtRatioPercent(value: unknown, digits = 1): string {
  const number = Number(value)
  if (value === null || value === undefined || !Number.isFinite(number)) return '—'
  return `${(number * 100).toFixed(digits)}%`
}

export function fmtDuration(seconds: unknown): string {
  const value = Number(seconds)
  if (!Number.isFinite(value)) return '—'
  const total = Math.max(0, Math.round(value))
  const days = Math.floor(total / 86400)
  const hours = Math.floor((total % 86400) / 3600)
  const minutes = Math.floor((total % 3600) / 60)
  if (days) return `${days}d ${hours}h`
  if (hours) return `${hours}h ${minutes}m`
  const secs = total % 60
  if (minutes) return `${minutes}m ${secs}s`
  return `${secs}s`
}

export function fmtDate(value: unknown): string {
  if (!value) return '—'
  const date = new Date(String(value))
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString()
}

export function shortCommit(value: unknown): string {
  return value ? String(value).slice(0, 10) : 'unknown'
}

export function activeState(state: unknown): boolean {
  return ['starting', 'running', 'stopping'].includes(String(state || ''))
}
