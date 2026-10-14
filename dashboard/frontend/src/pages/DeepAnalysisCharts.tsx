import { useState } from 'react'
import ReactECharts from 'echarts-for-react'
import type { DeepEpisode } from '../api'

export type Pair = { a: DeepEpisode | null; b: DeepEpisode | null }
export const CYAN = '#53d6dc'
export const AMBER = '#f2b66b'
export const fmt = (x: unknown, places = 2) =>
  typeof x === 'number' && Number.isFinite(x) ? x.toFixed(places) : '-'
const config = {
  animation: false, backgroundColor: 'transparent',
  textStyle: { color: '#acb8c4', fontSize: 11 },
  grid: { left: 60, top: 38, right: 24, bottom: 46 },
  legend: { textStyle: { color: '#abb8c5' }, top: 0 },
  tooltip: { trigger: 'axis', backgroundColor: '#161b21', borderColor: '#364049', textStyle: { color: '#eef2f7' } },
  xAxis: { type: 'value', axisLabel: { color: '#8896a4' }, splitLine: { lineStyle: { color: '#20272d' } } },
  yAxis: { type: 'value', axisLabel: { color: '#8896a4' }, splitLine: { lineStyle: { color: '#20272d' } } },
}
export function Chart({ title, option, caption, height = 295 }: { title: string; option: object; caption?: string; height?: number }) {
  return <section className="min-w-0 rounded-2xl border border-border bg-panel p-4 sm:p-5">
    <div className="mb-3 text-sm font-semibold">{title}</div>
    <ReactECharts option={option} notMerge lazyUpdate style={{ height, width: '100%' }} />
    {caption && <div className="mt-2 text-xs leading-relaxed text-muted">{caption}</div>}
  </section>
}
const scalar = (ep: DeepEpisode, key: string) => {
  const value = ep.channels[key]
  return Array.isArray(value) && value.every((v) => !Array.isArray(v)) ? value as Array<number | null> : []
}
const vectors = (ep: DeepEpisode, key: string) => {
  const value = ep.channels[key]
  return Array.isArray(value) && Array.isArray(value[0]) ? value as Array<Array<number | null>> : []
}
export function timeSeries(pair: Pair, key: string, ylabel: string, threshold?: number, factor = 1) {
  const series: object[] = ([pair.a,pair.b] as const).flatMap((ep,i) =>
    ep && scalar(ep,key).length ? [{
      name: i===0?'A - Primary':'B - Comparison', type:'line', showSymbol:false,
      lineStyle:{width:2.3},itemStyle:{color:i===0?CYAN:AMBER},
      data:ep.time.map((t,j) => [t,(scalar(ep,key)[j] ?? NaN)*factor]),
    }] : [])
  if(threshold !== undefined) series.push({
    name:'Reference threshold',type:'line',showSymbol:false,
    lineStyle:{width:1.3,type:'dashed'},itemStyle:{color:'#e0c28a'},
    data:[[0,threshold],[Math.max(pair.a?.duration_s??0,pair.b?.duration_s??0,1),threshold]],
  })
  return {...config,xAxis:{...config.xAxis,name:'Time (s)'},yAxis:{...config.yAxis,name:ylabel},series}
}
export function vectorSeries(pair: Pair,key:string,ylabel:string,indices:number[]=[0,1,2]){
  const colors=[[CYAN,'#68aee8','#97d69e'],[AMBER,'#ef8b82','#cca3ef']]
  const series=([pair.a,pair.b] as const).flatMap((ep,i)=>{
    if(!ep)return []
    const rows=vectors(ep,key)
    return rows.length?indices.filter(j=>j<rows[0].length).map(j=>({
      name:(i===0?'A':'B')+' - '+['left/x','right/y','z'][j],type:'line',showSymbol:false,
      data:rows.map((row,k)=>[ep.time[k],row[j]]),itemStyle:{color:colors[i][j]},lineStyle:{width:1.7},
    })):[]
  })
  return {...config,xAxis:{...config.xAxis,name:'Time (s)'},yAxis:{...config.yAxis,name:ylabel},series}
}
export function SpatialCharts({pair}:{pair:Pair}){
  const [metric,setMetric]=useState('tilt_rad')
  const pathSeries=([pair.a,pair.b] as const).flatMap((ep,i)=>{
    if(!ep)return []
    const row=vectors(ep,'root_pos'),goal=vectors(ep,'target_xy'),color=i===0?CYAN:AMBER
    return [
      {name:(i===0?'A':'B')+' path',type:'line',showSymbol:false,lineStyle:{color,width:2.5},data:row.map(r=>[r[0],r[1]])},
      {name:(i===0?'A':'B')+' endpoint',type:'scatter',symbolSize:15,itemStyle:{color,borderColor:'#fff',borderWidth:1},
        data:row.length?[[row.at(-1)?.[0],row.at(-1)?.[1]]]:[]},
      {name:(i===0?'A':'B')+' target',type:'scatter',symbol:'diamond',symbolSize:12,itemStyle:{color:'#fff'},
        data:goal.length?[[goal.at(-1)?.[0],goal.at(-1)?.[1]]]:[]},
    ]
  })
  const relative=([pair.a,pair.b] as const).flatMap((ep,i)=>{
    if(!ep)return []
    const pos=vectors(ep,'root_pos'),target=vectors(ep,'target_xy')
    return [{name:(i===0?'A':'B')+' relative',type:'scatter',symbolSize:7,
      itemStyle:{color:i===0?CYAN:AMBER,opacity:.55},
      data:pos.map((r,j)=>[Number(r[0]??0)-Number(target[j]?.[0]??0),Number(r[1]??0)-Number(target[j]?.[1]??0)])}]
  })
  const rings=[.035,.10].map((radius,i)=>({
    name:i===0?'3.5 cm arrival':'10 cm final stop',type:'line',showSymbol:false,
    lineStyle:{color:i===0?'#7bd6f0':'#e8b46e',width:1.5,type:'dashed'},
    data:Array.from({length:80},(_,j)=>[radius*Math.cos(j*2*Math.PI/79),radius*Math.sin(j*2*Math.PI/79)]),
  }))
  // Values are mapped to colors in geographic space, not only in a time chart.
  const spatialSignals=([pair.a,pair.b] as const).flatMap((ep,i)=>{
    if(!ep)return []
    const pos=vectors(ep,'root_pos'),values=scalar(ep,metric)
    const finite=values.filter((v):v is number=>typeof v==='number'&&Number.isFinite(v))
    const high=Math.max(...finite,.00001)
    const points=pos.map((row,j)=>{
      const ratio=Math.min(1,Math.max(0,(values[j]??0)/high))
      const red=Math.round(255*ratio),green=Math.round(203*(1-ratio))
      const color='rgb('+red+','+green+',105)'
      return {value:[row[0],row[1],values[j]],itemStyle:{color,opacity:.8}}
    })
    return [{name:(i===0?'A':'B')+' spatial '+metric,type:'scatter',symbolSize:7,data:points}]
  })
  const xy={...config,xAxis:{...config.xAxis,name:'X (m)',scale:true},yAxis:{...config.yAxis,name:'Y (m)',scale:true}}
  return <div className="space-y-4">
    <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-border bg-panel p-4 text-xs text-secondary">
      <span>Observed world position, target-centered footprint and endpoint. Target commands may change during the episode.</span>
      <label>Signal <select className="ml-2 rounded-lg border border-border bg-raised p-2 text-foreground" value={metric} onChange={e=>setMetric(e.target.value)}>
        <option value="tilt_rad">Tilt</option><option value="target_error_m">Distance</option><option value="planar_speed_mps">Speed</option>
      </select></label>
    </div>
    <div className="grid gap-4 xl:grid-cols-2">
      <Chart title="Top-down world trajectory" option={{...xy,series:pathSeries}} height={390}/>
      <Chart title="Target-relative positions" option={{...xy,series:[...relative,...rings]}} height={390}
        caption="The tolerance rings are visual references only, not new evaluator verdicts."/>
    </div>
    <div className="grid gap-4 xl:grid-cols-2">
      <Chart title="Spatial signal hotspots" option={{...xy,series:spatialSignals}} height={310}
        caption="Green: lower signal values. Red: higher within each episode. Colors are normalized per recording; absolute values are in the time chart."/>
      <Chart title="Selected diagnostic over time" option={timeSeries(pair,metric,metric==='tilt_rad'?'Radians':metric==='target_error_m'?'Metres':'m/s')} height={310}/>
    </div>
  </div>
}
export function SignalsCharts({pair}:{pair:Pair}){
 const charts=[
  ['Body tilt',timeSeries(pair,'tilt_rad','Degrees',4.58,180/Math.PI)],
  ['Target distance',timeSeries(pair,'target_error_m','Metres',.035)],
  ['Planar speed',timeSeries(pair,'planar_speed_mps','m/s',.08)],
  ['Heading error',timeSeries(pair,'heading_error_rad','Radians',.2)],
  ['Angular rates',vectorSeries(pair,'root_ang_vel_b','rad/s')],
  ['Wheel contacts',vectorSeries(pair,'contacts','Flag 0 / 1',[0,1])],
  ['Actuator effort',vectorSeries(pair,'effort','Nm')],
  ['Policy action',vectorSeries(pair,'action','Action')],
  ['Joint positions',vectorSeries(pair,'joint_pos','Radians')],
  ['Joint velocities',vectorSeries(pair,'joint_vel','rad/s')],
 ] as const
 return <div className="grid gap-4 xl:grid-cols-2">{charts.map(([name,option])=><Chart key={name} title={name} option={option}/>)}</div>
}
export function RewardsCharts({pair}:{pair:Pair}){
 const [source,setSource]=useState<'A'|'B'>('A')
 const episode=(source==='A'?pair.a:pair.b)??pair.a
 const terms=Object.entries(episode?.reward_terms??{}).map(([name,values])=>({
  name,total:values.reduce<number>((acc,v)=>acc+(v??0),0),magnitude:values.reduce<number>((acc,v)=>acc+Math.abs(v??0),0),
 })).sort((a,b)=>b.magnitude-a.magnitude)
 return <div className="space-y-4">
  <Chart title="Actual step reward" option={timeSeries(pair,'reward_total','Reward / step')}/>
  <div className="rounded-2xl border border-border bg-panel p-5">
   <div className="mb-4 flex flex-wrap justify-between gap-3"><h3 className="font-semibold">Recorded reward terms</h3>
    {pair.b&&<div className="flex gap-2">{(['A','B'] as const).map(id=><button onClick={()=>setSource(id)}
     className={'rounded-lg border px-3 py-2 text-xs '+(id===source?'border-info text-info':'border-border text-secondary')} key={id}>{id} checkpoint</button>)}</div>}
   </div>
   {terms.length?terms.slice(0,16).map(term=><div key={term.name} className="mb-3 grid items-center gap-3 sm:grid-cols-[minmax(0,1fr)_160px_80px]">
    <span title={term.name} className="truncate font-mono text-xs text-secondary">{term.name}</span>
    <div className="h-2 rounded-full bg-raised"><div className="h-full rounded-full bg-info" style={{width:String(Math.round(100*term.magnitude/Math.max(.001,terms[0].magnitude)))+'%'}}/></div>
    <span className="text-right font-mono text-xs">{fmt(term.total,2)}</span></div>):
    <p className="text-sm text-muted">No term-level reward recording exists for this episode.</p>}
   <p className="mt-3 text-xs text-muted">Bars: absolute cumulative contribution. Numbers: signed recorded totals. Descriptive analysis only.</p>
  </div>
 </div>
}
export function EventsCharts({pair}:{pair:Pair}){
 const keys=[...new Set([...(pair.a?Object.keys(pair.a.channels):[]),...(pair.b?Object.keys(pair.b.channels):[])].filter(k=>k.startsWith('evaluator_')))]
 const [current,setCurrent]=useState('evaluator_stop_all_pass')
 return <div className="space-y-4">
  <div className="rounded-xl border border-info/30 bg-info/5 p-4 text-sm text-secondary"><strong className="text-info">Recorded evaluator flags - read-only.</strong> Displaying these does not run acceptance checks or produce evaluation verdicts.</div>
  {keys.length?<><label className="block text-xs text-secondary">Recorded condition
   <select value={keys.includes(current)?current:keys[0]} onChange={e=>setCurrent(e.target.value)}
    className="mt-2 block w-full rounded-lg border border-border bg-panel p-3 text-sm text-foreground">
    {keys.map(k=><option value={k} key={k}>{k.replaceAll('_',' ')}</option>)}
   </select></label>
   <Chart title="Recorded condition timeline" option={timeSeries(pair,keys.includes(current)?current:keys[0],'Recorded flag')}/>
   <div className="grid gap-4 xl:grid-cols-2">
    <Chart title="Consecutive settled dwell" option={timeSeries(pair,'evaluator_stop_dwell_s','Seconds',1)}/>
    <Chart title="Recovery dwell" option={timeSeries(pair,'evaluator_recovery_stable_dwell_s','Seconds')}/>
   </div></>:<p className="rounded-xl border border-dashed border-border p-7 text-secondary">No evaluator flags were captured for this playback.</p>}
 </div>
}
export function StatisticsCharts({pair}:{pair:Pair}){
 const [metric,setMetric]=useState('target_error_m')
 const names=[['target_error_m','Target error'],['tilt_rad','Tilt'],['planar_speed_mps','Speed'],['heading_error_rad','Heading error'],['reward_total','Step reward']]
 const series=([pair.a,pair.b] as const).flatMap((ep,i)=>{
  if(!ep)return []
  const values=scalar(ep,metric).filter((v):v is number=>v!==null&&Number.isFinite(v))
  if(!values.length)return []
  const min=Math.min(...values),max=Math.max(...values),step=Math.max((max-min)/20,1e-6)
  const bins=Array(20).fill(0)
  values.forEach(v=>bins[Math.min(19,Math.max(0,Math.floor((v-min)/step)))]+=1)
  return [{name:i===0?'A':'B',type:'bar',itemStyle:{color:i===0?CYAN:AMBER},data:bins.map((count,j)=>[min+(j+.5)*step,count])}]
 })
 return <div className="space-y-4">
  <label className="text-xs text-secondary">Distribution signal <select value={metric} onChange={e=>setMetric(e.target.value)} className="ml-2 rounded-lg border border-border bg-panel p-2 text-foreground">{names.map(([id,name])=><option key={id} value={id}>{name}</option>)}</select></label>
  <Chart title="Distribution of sampled episode frames" option={{...config,xAxis:{...config.xAxis,name:metric},yAxis:{...config.yAxis,name:'Frames'},series}}/>
  <div className="grid gap-4 xl:grid-cols-2">{([pair.a,pair.b] as const).map((ep,i)=>ep&&<section key={ep.checkpoint_id} className="rounded-2xl border border-border bg-panel p-5">
    <h3 className="mb-3 font-semibold">{i===0?'A - Primary':'B - Comparison'} / descriptive statistics</h3>
    <div className="grid grid-cols-[minmax(0,1fr)_repeat(3,64px)] gap-2 border-b border-border pb-2 text-[11px] text-muted"><span>Signal</span><span>Mean</span><span>P95</span><span>Max</span></div>
    {names.map(([id,name])=><div key={id} className="grid grid-cols-[minmax(0,1fr)_repeat(3,64px)] gap-2 border-b border-border py-2 text-xs">
     <span className="text-secondary">{name}</span><span>{fmt(ep.stats[id]?.mean,3)}</span><span>{fmt(ep.stats[id]?.p95,3)}</span><span>{fmt(ep.stats[id]?.max,3)}</span>
    </div>)}
  </section>)}</div>
  <p className="text-xs text-muted">Full-resolution summary statistics; decimated interactive charts. This is not a full benchmark population evaluation.</p>
 </div>
}
