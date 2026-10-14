import { lazy, Suspense, useEffect, useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Link } from '@tanstack/react-router'
import {
  ArrowLeftRight, ArrowRight, BarChart3, CheckCircle2, ChevronRight, Database, Eye,
  GitCompareArrows, Map, Microscope, Radio, RefreshCw, Search, ShieldCheck, SlidersHorizontal,
  Target, Waves, Activity,
} from 'lucide-react'
import type { DeepCatalog, DeepCheckpoint, DeepEpisode, DeepRecording } from '../api'
import { api } from '../api'
import { CYAN, AMBER, fmt, type Pair, SpatialCharts, SignalsCharts, RewardsCharts, EventsCharts, StatisticsCharts } from './DeepAnalysisCharts'
import { DeepAnalysisLive } from './DeepAnalysisLive'

const RerunWebViewer=lazy(()=>import('../components/RerunWebViewer'))
type Mode='single'|'compare'|'live'
type Tab='spatial'|'signals'|'rewards'|'events'|'statistics'|'rerun'|'provenance'
type Choice={mode:'single'|'compare'; a:string; b:string|null}|{mode:'live';runId:string}
const views: Array<{id:Tab;label:string;icon:typeof Map}>=[
 {id:'spatial',label:'Spatial',icon:Map},{id:'signals',label:'Signals',icon:Waves},
 {id:'rewards',label:'Rewards',icon:BarChart3},{id:'events',label:'Events',icon:Activity},
 {id:'statistics',label:'Statistics',icon:SlidersHorizontal},{id:'rerun',label:'Rerun 3D',icon:Eye},
 {id:'provenance',label:'Provenance',icon:Database},
]
export function DeepTabs({tab,setTab}:{tab:Tab;setTab:(tab:Tab)=>void}){
 return <nav aria-label="Deep analysis sub-tabs" className="flex flex-wrap gap-2 rounded-2xl border border-border bg-panel p-2">
  {views.map(({id,label,icon:Icon})=><button type="button" key={id} aria-current={id===tab?'page':undefined}
   className={'flex min-h-10 items-center gap-2 rounded-lg px-3 py-2 text-xs font-semibold transition-colors sm:text-sm '+(id===tab?'bg-foreground text-background':'text-secondary hover:bg-raised hover:text-foreground')}
   onClick={()=>setTab(id)}><Icon size={15}/>{label}</button>)}
 </nav>
}
function Chooser({ catalog, label, checkpoint, exclude, onChange }:{
 catalog:DeepCatalog;label:string;checkpoint:string;exclude:string;onChange:(value:string)=>void
}){
 const [run,setRun]=useState('')
 const [filter,setFilter]=useState('')
 const selected=catalog.checkpoints.find(c=>c.id===checkpoint)
 const runs=useMemo(()=>[...new Set(catalog.checkpoints.map(c=>c.run))].sort(),[catalog])
 const activeRun=run||selected?.run||runs[0]||''
 const options=catalog.checkpoints.filter(c=>c.run===activeRun&&(String(c.iteration).includes(filter)||c.path.toLowerCase().includes(filter.toLowerCase())))
 return <section className="min-w-0 rounded-2xl border border-border bg-panel p-4 sm:p-5">
  <p className="text-xs font-bold uppercase tracking-widest text-info">{label}</p>
  <label className="mt-4 block text-xs font-semibold text-secondary">Training lineage
   <select aria-label={label+' training run'} className="mt-2 w-full truncate rounded-lg border border-border bg-raised p-3 text-sm text-foreground"
     value={activeRun} onChange={e=>{
      setRun(e.target.value);setFilter('')
      onChange(catalog.checkpoints.find(c=>c.run===e.target.value&&c.id!==exclude)?.id??'')
     }}>{runs.map(r=><option key={r} value={r}>{r}</option>)}</select>
  </label>
  <label className="mt-4 block text-xs font-semibold text-secondary">Filter step or path
   <div className="mt-2 flex items-center gap-2 rounded-lg border border-border bg-raised px-3">
    <Search size={16} className="text-muted"/><input className="min-w-0 flex-1 bg-transparent py-3 text-sm text-foreground outline-none" placeholder="e.g. 20400" value={filter} onChange={e=>setFilter(e.target.value)}/>
   </div>
  </label>
  <label className="mt-4 block text-xs font-semibold text-secondary">Exact checkpoint
   <select aria-label={label+' checkpoint'} className="mt-2 w-full truncate rounded-lg border border-border bg-raised p-3 text-sm text-foreground"
     value={options.some(c=>c.id===checkpoint)?checkpoint:''} onChange={e=>onChange(e.target.value)}>
     <option value="">Choose a checkpoint</option>
     {options.map(c=><option disabled={c.id===exclude} key={c.id} value={c.id}>
      Step {c.iteration.toLocaleString()} - {c.recording_count} captures - {c.path.split('/').slice(-3).join('/')}
     </option>)}
   </select>
  </label>
  {selected&&<div className="mt-4 rounded-xl border border-info/25 bg-info/5 p-3">
    <p className="text-[10px] font-bold uppercase tracking-wider text-info">Selected identity</p>
    <p className="mt-2 font-semibold">Step {selected.iteration.toLocaleString()}</p>
    <p className="mt-2 break-all font-mono text-[11px] leading-relaxed text-secondary">{selected.path}</p>
    <p className="mt-2 text-[11px] text-muted">{selected.recording_count} matching captured episodes - {new Date(selected.modified_at*1000).toLocaleString()}</p>
   </div>}
 </section>
}
function Landing({catalog,onSelect}:{catalog:DeepCatalog;onSelect:(choice:Choice)=>void}){
 const [mode,setMode]=useState<Mode>('single')
 const [a,setA]=useState('')
 const [b,setB]=useState('')
 const [liveId,setLiveId]=useState('')
 const ca=catalog.checkpoints.find(c=>c.id===a),cb=catalog.checkpoints.find(c=>c.id===b)
 const okay=mode==='live'?catalog.live_runs.some(r=>r.id===liveId):!!ca&&(mode==='single'||!!cb&&a!==b)
 const modes=[
  {id:'single',label:'Single checkpoint',caption:'Inspect one frozen policy and its recordings.',icon:Target},
  {id:'compare',label:'Compare checkpoints',caption:'Two different saved policies with explicit identities.',icon:GitCompareArrows},
  {id:'live',label:'Live checkpoint / PPO',caption:'Subscribe to actual in-memory training samples.',icon:Radio},
 ] as const
 return <div className="space-y-5">
  <div className="grid gap-3 lg:grid-cols-3">{modes.map(({id,label,caption,icon:Icon})=><button key={id} type="button"
    aria-pressed={mode===id} onClick={()=>setMode(id)}
    className={'min-h-36 rounded-2xl border p-5 text-left transition-colors '+(mode===id?'border-info/70 bg-info/[.08]':'border-border bg-panel hover:border-border-strong')}>
    <div className="mb-3 flex justify-between"><Icon size={23} className={mode===id?'text-info':'text-secondary'}/>{mode===id&&<CheckCircle2 size={18} className="text-info"/>}</div>
    <h2 className="font-semibold">{label}</h2><p className="mt-2 text-xs leading-relaxed text-secondary">{caption}</p>
   </button>)}</div>
  {mode==='live'?<section className="rounded-2xl border border-border bg-panel p-5">
   <h3 className="font-semibold">Select an active training run</h3>
   <p className="mt-2 text-xs text-secondary">A live run is an in-memory policy, not necessarily identical to its latest saved .pt checkpoint. Streaming requires the existing PPO monitor.</p>
   {catalog.live_runs.length?<label className="mt-4 block text-xs text-secondary">Source run
     <select aria-label="Live training source" value={liveId} onChange={e=>setLiveId(e.target.value)}
      className="mt-2 w-full rounded-lg border border-border bg-raised p-3 text-sm text-foreground">
      <option value="">Choose an active run</option>
      {catalog.live_runs.map(r=><option key={r.id} value={r.id}>{r.label} - {r.state} - step {r.iteration??'?'} - {r.id}</option>)}
     </select></label>:<div className="mt-4 rounded-lg border border-dashed border-border p-5 text-sm text-muted">No non-stale active dashboard run is available. Live samples cannot be fabricated; choose historical analysis instead.</div>}
  </section>:<div className={'grid gap-4 '+(mode==='compare'?'xl:grid-cols-2':'')}>
   <Chooser catalog={catalog} label="A - Primary checkpoint" checkpoint={a} exclude={b} onChange={setA}/>
   {mode==='compare'&&<Chooser catalog={catalog} label="B - Comparison checkpoint" checkpoint={b} exclude={a} onChange={setB}/>}
  </div>}
  <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-border bg-panel p-4">
   <p className="max-w-3xl text-xs leading-relaxed text-muted">Exact checkpoint provenance is verified before any recording is displayed. Exploration never starts or scores a benchmark.</p>
   <button type="button" disabled={!okay} className="flex items-center gap-2 rounded-lg bg-foreground px-5 py-3 text-sm font-bold text-background disabled:cursor-not-allowed disabled:opacity-40"
    onClick={()=>{
     if(mode==='live')onSelect({mode:'live',runId:liveId})
     else if(ca)onSelect({mode, a:ca.id, b:mode==='compare'?cb!.id:null})
    }}>Open deep analysis <ArrowRight size={16}/></button>
  </div>
 </div>
}
function DetailCards({cp,episodes,side,selected,onChange}:{
 cp:DeepCheckpoint;episodes:DeepRecording[];side:'A'|'B';selected:string;onChange:(val:string)=>void
}){
 const current=episodes.find(r=>r.id===selected)||episodes[0]
 return <section className="min-w-0 rounded-2xl border border-border bg-panel p-4">
  <div className="flex items-start gap-3"><span className="mt-1 h-3 w-3 shrink-0 rounded-full" style={{background:side==='A'?CYAN:AMBER}}/>
   <div className="min-w-0">
    <h3 className="text-sm font-bold">{side} - {side==='A'?'PRIMARY':'COMPARISON'} - Step {cp.iteration.toLocaleString()}</h3>
    <p className="mt-1 break-all font-mono text-[11px] text-secondary">{cp.path}</p>
   </div>
  </div>
  <label className="mt-4 block text-xs text-secondary">Exact captured scenario
   <select aria-label={side+' captured episode'} className="mt-2 block w-full rounded-lg border border-border bg-raised p-3 text-sm text-foreground"
    value={current?.id??''} onChange={e=>onChange(e.target.value)} disabled={!episodes.length}>
    {!episodes.length?<option value="">No recordings for this checkpoint</option>:episodes.map(r=><option key={r.id} value={r.id}>{r.scenario??r.id} - {r.frames} frames - {r.rrd_url?'Rerun':'NPZ'}</option>)}
   </select>
  </label>
 </section>
}
const observations='Explore recorded values only. Algorithmic evaluation against requirements is handled separately by the benchmark evaluator.'
function Missing({name}:{name:string}){
 return <div className="rounded-2xl border border-dashed border-border bg-panel p-8 text-sm text-secondary">
  No checkpoint-verified capture exists for {name}. This page will not silently use a recording from a different policy. Record an exact scenario in the capture workflow, then refresh the catalog.
 </div>
}
function Provenance({pair,cps,recordings}:{pair:Pair;cps:DeepCheckpoint[];recordings:Array<DeepRecording|null>}){
 return <div className="grid gap-4 xl:grid-cols-2">{cps.map((cp,i)=>{
  const ep=i===0?pair.a:pair.b
  const entries=[
   ['Role',i===0?'A - Primary':'B - Comparison'],['Training lineage',cp.run],
   ['Checkpoint step',String(cp.iteration)],['Checkpoint file',cp.path],
   ['Selected capture',recordings[i]?.id??'Not recorded'],['Scenario',ep?.scenario??'Unknown'],
   ['Task',ep?.task??'Unknown'],['Frames',String(ep?.frames??'Unknown')],
   ['Checkpoint SHA-256',ep?.checkpoint_sha256??'Unknown'],['Suite SHA-256',ep?.suite_sha256??'Unknown'],
  ]
  return <section className="rounded-2xl border border-border bg-panel p-5" key={cp.id}>
   <h3 className="mb-4 font-semibold">{i===0?'Primary':'Comparison'} / source audit</h3>
   <div className="space-y-3">{entries.map(([key,value])=><div key={key} className="rounded-lg bg-raised p-3">
    <div className="text-[10px] font-bold uppercase tracking-wide text-muted">{key}</div>
    <div className="mt-1 break-all font-mono text-xs text-foreground">{value}</div></div>)}</div>
  </section>
 })}</div>
}
function Recorded({catalog,choice}:{catalog:DeepCatalog;choice:Extract<Choice,{mode:'single'|'compare'}>}){
 const cpA=catalog.checkpoints.find(c=>c.id===choice.a)!
 const cpB=catalog.checkpoints.find(c=>c.id===choice.b)
 const [recordingA,setRecordingA]=useState('')
 const [recordingB,setRecordingB]=useState('')
 const [tab,setTab]=useState<Tab>('spatial')
 const [showViewer,setShowViewer]=useState(false)
 const [renderer,setRenderer]=useState<'webgl'|'webgpu'>('webgl')
 const [viewSource,setViewSource]=useState<'A'|'B'>('A')
 const rowsA=catalog.recordings[cpA.id]??[],rowsB=cpB?catalog.recordings[cpB.id]??[]:[]
 const a=rowsA.find(r=>r.id===recordingA)??rowsA[0]??null
 const b=rowsB.find(r=>r.id===recordingB)??rowsB.find(r=>r.scenario && r.scenario===a?.scenario)??rowsB[0]??null
 const qa=useQuery({queryKey:['deep-episode',cpA.id,a?.id],queryFn:()=>api.deepEpisode(cpA.id,a!.id),enabled:!!a,staleTime:300_000})
 const qb=useQuery({queryKey:['deep-episode',cpB?.id,b?.id],queryFn:()=>api.deepEpisode(cpB!.id,b!.id),enabled:!!cpB&&!!b,staleTime:300_000})
 const pair:Pair={a:qa.data??null,b:qb.data??null}
 const mismatch=qa.data?.task&&qb.data?.task&&qa.data.task!==qb.data.task
 const unmatched=qa.data?.scenario&&qb.data?.scenario&&qa.data.scenario!==qb.data.scenario
 const currentViewer=(viewSource==='A'?a:b)??a
 return <div className="space-y-5">
  <div className={'grid gap-4 '+(cpB?'xl:grid-cols-2':'')}>
   <DetailCards cp={cpA} side="A" episodes={rowsA} selected={recordingA} onChange={setRecordingA}/>
   {cpB&&<DetailCards cp={cpB} side="B" episodes={rowsB} selected={recordingB} onChange={setRecordingB}/>}
  </div>
  {mismatch&&<p role="alert" className="rounded-xl border border-danger/50 bg-danger/10 p-4 text-sm text-danger">Task mismatch between selected recordings - comparison may be invalid.</p>}
  {unmatched&&<p className="rounded-xl border border-warning/50 bg-warning/10 p-4 text-sm text-warning">Selected captures have different scenario IDs. These visual overlays are exploratory, not a controlled A/B benchmark.</p>}
  <DeepTabs tab={tab} setTab={setTab}/>
  {!a?<Missing name="A - Primary"/>:cpB&&!b?<Missing name="B - Comparison"/>:
   (qa.isPending||qb.isPending&&!!b)?<div className="rounded-2xl border border-border bg-panel p-12 text-center text-secondary">Loading checkpoint-verified telemetry...</div>:
   qa.error||qb.error?<p role="alert" className="rounded-2xl border border-danger p-6 text-danger">{qa.error?.message??qb.error?.message}</p>:
   tab==='spatial'?<SpatialCharts pair={pair}/>:
   tab==='signals'?<SignalsCharts pair={pair}/>:
   tab==='rewards'?<RewardsCharts pair={pair}/>:
   tab==='events'?<EventsCharts key={(a?.id??'')+(b?.id??'')} pair={pair}/>:
   tab==='statistics'?<StatisticsCharts pair={pair}/>:
   tab==='provenance'?<Provenance pair={pair} cps={[cpA,...(cpB?[cpB]:[])]} recordings={[a,b]}/>:
   <section className="rounded-2xl border border-border bg-panel p-5">
    <h3 className="text-lg font-semibold">Rerun 3D episode viewer</h3>
    <p className="mt-2 text-sm text-muted">Select the exact checkpoint source below; the native WebAssembly viewer loads only when requested.</p>
    <div className="my-4 flex flex-wrap items-center gap-3">
     <button className={'rounded-lg border px-3 py-2 text-sm '+(viewSource==='A'?'border-info text-info':'border-border text-secondary')}
      onClick={()=>{setViewSource('A');setShowViewer(false)}}>A - Primary</button>
     {cpB&&<button className={'rounded-lg border px-3 py-2 text-sm '+(viewSource==='B'?'border-warning text-warning':'border-border text-secondary')}
      onClick={()=>{setViewSource('B');setShowViewer(false)}}>B - Comparison</button>}
     <label className="text-xs text-secondary">Renderer <select className="ml-2 rounded-lg border border-border bg-raised p-2 text-foreground" value={renderer} onChange={e=>setRenderer(e.target.value as typeof renderer)}>
      <option value="webgl">WebGL</option><option value="webgpu">WebGPU</option></select></label>
     {currentViewer?.rrd_url&&<button onClick={()=>setShowViewer(v=>!v)} className="rounded-lg border border-info bg-info/10 px-4 py-2 text-sm text-info">{showViewer?'Close viewer':'Open Rerun'}</button>}
    </div>
    {showViewer&&currentViewer?.rrd_url?<div className="h-[480px] min-w-0 overflow-hidden rounded-xl border border-border bg-black lg:h-[650px]">
     <Suspense fallback={<p className="p-5 text-sm text-muted">Loading Rerun viewer...</p>}>
      <RerunWebViewer url={currentViewer.rrd_url} backend={renderer}/>
     </Suspense>
    </div>:<div className="rounded-xl border border-dashed border-border p-10 text-center text-sm text-muted">{currentViewer?.rrd_url?'Viewer available. Open it above.':'No RRD generated for this exact capture. The other tabs remain available.'}</div>}
   </section>}
  <div className="rounded-xl border border-border bg-panel px-4 py-3 text-xs text-muted">{observations}</div>
 </div>
}
export function DeepAnalysisPage(){
 const catalogQuery=useQuery({queryKey:['deep-catalog'],queryFn:api.deepCatalog,staleTime:20_000,refetchInterval:120_000})
 const [choice,setChoice]=useState<Choice|null>(null)
 const catalog=catalogQuery.data
 return <div className="space-y-6" data-testid="deep-analysis-page">
  <header className="flex flex-wrap items-start justify-between gap-4">
   <div><div className="flex items-center gap-2 text-xs font-bold uppercase tracking-[.14em] text-info"><Microscope size={16}/> Read-only research workspace</div>
    <h1 className="mt-2 text-3xl font-bold tracking-tight sm:text-4xl">Deep Analysis</h1>
    <p className="mt-3 max-w-3xl text-sm leading-relaxed text-secondary">Explore checkpoint behavior through Rerun, recorded signals, target-relative trajectories and rewards. <strong className="text-foreground">Analysis observes; Evaluation checks requirements.</strong></p>
   </div>
   <div className="flex items-center gap-2">
    {choice&&<button className="flex items-center gap-2 rounded-lg border border-border px-3 py-2 text-sm" onClick={()=>setChoice(null)}><ArrowLeftRight size={16}/> Change checkpoint</button>}
    <button aria-label="Refresh checkpoint catalog" className="rounded-lg border border-border p-3 text-secondary" onClick={()=>catalogQuery.refetch()}><RefreshCw size={17}/></button>
   </div>
  </header>
  <div className="flex flex-wrap items-center gap-4 border-b border-border pb-4 text-xs font-semibold text-secondary">
   <span className="flex items-center gap-2 text-info"><Microscope size={15}/> Analysis / visual inspection</span>
   <Link to="/analyze" className="flex items-center gap-2 hover:text-foreground"><ShieldCheck size={15}/> Evaluation / algorithmic acceptance <ChevronRight size={13}/></Link>
  </div>
  {catalogQuery.error?<p role="alert" className="rounded-xl border border-danger/50 p-5 text-danger">Could not load checkpoints: {catalogQuery.error.message}</p>:
   !catalog?<div className="min-h-72 animate-pulse rounded-xl border border-border bg-panel p-8 text-muted">Discovering checkpoint artifacts...</div>:
   !choice?<Landing catalog={catalog} onSelect={setChoice}/>:
   choice.mode==='live'?<DeepAnalysisLive key={choice.runId} runId={choice.runId} catalog={catalog}/>:
   <Recorded key={choice.a+choice.b} catalog={catalog} choice={choice}/>}
  <footer className="border-t border-border pt-3 text-xs text-muted">{observations} Live PPO data come only from actual monitor samples; they are not frozen checkpoint replays.</footer>
 </div>
}
