import * as echarts from 'echarts/core'
import { LineChart } from 'echarts/charts'
import { DataZoomComponent, GridComponent, TooltipComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import ReactEChartsCore from 'echarts-for-react/lib/core'
import type { TelemetryRecord } from '../types'

echarts.use([LineChart, DataZoomComponent, GridComponent, TooltipComponent, CanvasRenderer])
import { fmtNumber } from '../lib/utils'

export function TelemetryChart({
  records,
  metric,
  title,
  description,
}: {
  records: TelemetryRecord[]
  metric: string
  title: string
  description?: string
}) {
  const points = records
    .map((record) => {
      const canonical = record.canonical_metrics || {}
      const metrics = record.metrics || {}
      const value = canonical[metric] ?? metrics[metric]
      return [record.iteration, value] as const
    })
    .filter(([iteration, value]) => Number.isFinite(Number(iteration)) && Number.isFinite(Number(value)))

  const latest = points.length ? points[points.length - 1][1] : null
  const option = {
    animation: false,
    backgroundColor: 'transparent',
    grid: { top: 18, left: 58, right: 18, bottom: 58 },
    tooltip: {
      trigger: 'axis',
      backgroundColor: '#14171a',
      borderColor: '#343a40',
      textStyle: { color: '#f1f3f5', fontSize: 12 },
      valueFormatter: (value: unknown) => fmtNumber(value, 6),
    },
    xAxis: {
      type: 'value',
      axisLine: { lineStyle: { color: '#343a40' } },
      axisTick: { show: false },
      axisLabel: { color: '#6e767f', fontSize: 11 },
      splitLine: { show: false },
    },
    yAxis: {
      type: 'value',
      scale: true,
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { color: '#6e767f', fontSize: 11 },
      splitLine: { lineStyle: { color: '#25292e' } },
    },
    dataZoom: [
      { type: 'inside', filterMode: 'none' },
      {
        type: 'slider',
        height: 16,
        bottom: 12,
        borderColor: 'transparent',
        backgroundColor: '#0b0d0f',
        fillerColor: 'rgba(241,243,245,0.10)',
        handleStyle: { color: '#a8afb7', borderColor: '#a8afb7' },
        moveHandleStyle: { color: '#6e767f' },
        textStyle: { color: '#6e767f' },
      },
    ],
    series: [
      {
        type: 'line',
        showSymbol: false,
        connectNulls: false,
        sampling: 'lttb',
        lineStyle: { width: 2, color: '#d7dadd' },
        emphasis: { lineStyle: { width: 2 } },
        data: points,
      },
    ],
  }

  return (
    <section className="rounded-xl border border-border bg-panel p-5">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h3 className="text-[15px] font-semibold">{title}</h3>
          {description ? <p className="mt-1 max-w-xl text-xs leading-relaxed text-muted">{description}</p> : null}
        </div>
        <div className="numeric text-right text-lg font-semibold">{fmtNumber(latest, 5)}</div>
      </div>
      <ReactEChartsCore echarts={echarts} option={option} style={{ height: 300, marginTop: 12 }} notMerge lazyUpdate />
    </section>
  )
}
