import { useEffect, useState } from 'react'
import ReactECharts from 'echarts-for-react'
import type { DeepCatalog, LivePpoEnvironment, LivePpoFrame } from '../api'
import { api } from '../api'
import { Chart, fmt } from './DeepAnalysisCharts'

type Tab='spatial'|'signals'|'rewards'|'events'|'statistics'|'rerun'|'provenance'
const tabs:Tab[]=['spatial','signals','rewards','events','statistics','rerun','provenance']
export function DeepAnalysisLive({runId,catalog}:{runId:string;catalog:DeepCatalog}){
 const [frame,setFrame]=useState<LivePpoFrame|null>(null)
 const [history,setHistory]=useState<LivePpoFrame[]>([])
 const [status,setStatus]=useState<'connecting'|'connected'|'reconnecting'>('connecting')
 const [envId,setEnvId]=useState<number|null>(null)
 const [tab,setTab]=useState<Tab>('spatial')
 useEffect(()=>{
  let alive=true
  setFrame(null);setHistory([]);setStatus('connecting')
  void api.livePpoLatest(runId).then(f=>{
   if(alive&&f.available&&f.source==='actual_ppo_rollout'){setFrame(f);setHistory([f])}
  }).catch(()=>{})
  const s=new EventSource(api.livePpoStreamUrl(runId))
  s.addEventListener('frame',event=>{
   if(!alive)return
   try{
    const f=JSON.parse((event as MessageEvent<string>).data) as LivePpoFrame
    if(!f.available||f.source!=='actual_ppo_rollout')return
    setFrame(f);setStatus('connected')
    setHistory(old=>old.at(-1)?.sequence===f.sequence?old:[...old.slice(-239),f])
   }catch{setStatus('reconnecting')}
  })
  s.onerror=()=>{if(alive)setStatus('reconnecting')}
  return ()=>{alive=false;s.close()}
 },[runId])
 const liveRun=catalog.live_runs.find(r=>r.id===runId)
 const envs=frame?.environments??[]
 const current=envs.find(e=>e.environment_id===envId)??envs[0]
 const samples=history.map(f=>f.environments.find(e=>e.environment_id===current?.environment_id))
  .filter((x):x is LivePpoEnvironment=>x!==undefined)
  .reduce<LivePpoEnvironment[]>((list,env)=>{
   if(env.reset||(list.length&&env.episode_step<list.at(-1)!.episode_step))return [env]
   return [...list,env]
  },[])
 const stale=!frame||Date.now()/1000-frame.sampled_at>10
 const latest=current?.target_xy
 const trajectory=samples.map(e=>[e.position_xyz[0],e.position_xyz[1]])
 const series=[{name:'PPO spatial trail',type:'line',showSymbol:false,
  data:trajectory,lineStyle:{width:3,color:'#53d6dc'}},
  ...(latest?[{name:'Target',type:'scatter',symbol:'diamond',symbolSize:18,
   data:[latest],itemStyle:{color:'#f2b66b'}}]:[])]
 const chartBase={
  animation:false,legend:{textStyle:{color:'#b7c3ce'}},
  grid:{left:50,right:25,top:35,bottom:43},
  xAxis:{type:'value',name:'X',axisLabel:{color:'#8191a1'},splitLine:{lineStyle:{color:'#222a32'}}},
  yAxis:{type:'value',name:'Y',axisLabel:{color:'#8191a1'},splitLine:{lineStyle:{color:'#222a32'}}},
 }
 const trend=(name:string,signal:(e:LivePpoEnvironment)=>number,unit:string)=>({
  ...chartBase,
  xAxis:{...chartBase.xAxis,name:'Display sample',type:'value'},
  yAxis:{...chartBase.yAxis,name:unit},
  tooltip:{trigger:'axis'},
  series:[{name,type:'line',showSymbol:false,data:samples.map((e,i)=>[i,signal(e)]),
   lineStyle:{width:2,color:'#53d6dc'}}],
 })
 return <div className="space-y-5">
  <div className="rounded-xl border border-info/35 bg-info/5 p-4 text-sm">
   <p className="font-semibold text-info">Live PPO / {liveRun?.label??runId} / {stale?'No fresh sample':status}</p>
   <p className="mt-2 text-xs leading-relaxed text-secondary">This is an actual sampled in-memory policy state. It is not a frozen checkpoint, does not launch an evaluator, and is not a real-time MuJoCo rendering stream. Display samples can be dropped without affecting training.</p>
  </div>
  <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-border bg-panel p-4 text-xs text-secondary">
   <span>Run ID {runId} / PPO global step {frame?.global_step.toLocaleString()??'-'} / capture overhead {fmt(frame?.capture_overhead_ms)} ms</span>
   {envs.length>0&&<label>Sampled environment
    <select className="ml-2 rounded-lg border border-border bg-raised p-2 text-foreground"
     value={current?.environment_id??0} onChange={e=>setEnvId(Number(e.target.value))}>
     {envs.map(e=><option key={e.environment_id} value={e.environment_id}>#{e.environment_id}</option>)}
    </select></label>}
  </div>
  <nav className="flex flex-wrap gap-2 rounded-xl border border-border bg-panel p-2" aria-label="Live deep analysis sub-tabs">
   {tabs.map(name=><button key={name} type="button" aria-current={tab===name?'page':undefined}
    className={'rounded-lg px-3 py-2 text-xs font-semibold capitalize '+(tab===name?'bg-foreground text-background':'text-secondary hover:bg-raised')}
    onClick={()=>setTab(name)}>{name}</button>)}
  </nav>
  {!current?<div className="rounded-2xl border border-dashed border-border bg-panel p-10 text-sm text-secondary">
   Waiting for actual PPO monitor frames. No historical telemetry will be passed off as a live feed.
  </div>:tab==='spatial'?<div className="grid gap-4 xl:grid-cols-2">
   <Chart title="Live sampled world XY" height={410}
    option={{...chartBase,xAxis:{...chartBase.xAxis,scale:true},yAxis:{...chartBase.yAxis,scale:true},series}}/>
   <div className="rounded-2xl border border-border bg-panel p-5">
    <h3 className="mb-4 text-base font-semibold">Current observed robot</h3>
    <div className="grid grid-cols-2 gap-3">{([
     ['Body tilt',fmt(current.tilt_deg,1)+' degrees'],
     ['Target distance',fmt(current.target_distance_m,3)+' m'],
     ['Actual speed',fmt(current.speed_mps,3)+' m/s'],
     ['Closing speed',fmt(current.closing_speed_mps,3)+' m/s'],
     ['Step reward',fmt(current.step_reward,4)],
     ['Wheel contacts',(current.wheel_contacts.left?'yes':'no')+' / '+(current.wheel_contacts.right?'yes':'no')],
     ]).map(([key,value])=><div className="rounded-lg bg-raised p-3" key={key}>
      <p className="text-[11px] text-muted">{key}</p><p className="mt-2 text-sm font-semibold tabular-nums">{value}</p>
     </div>)}</div>
   </div>
  </div>:tab==='signals'?<div className="grid gap-4 xl:grid-cols-2">
   <Chart title="Body tilt" option={trend('Tilt',f=>f.tilt_deg,'Degrees')}/>
   <Chart title="Planar speed" option={trend('Speed',f=>f.speed_mps,'m/s')}/>
   <Chart title="Goal distance" option={trend('Goal distance',f=>f.target_distance_m??NaN,'m')}/>
   <Chart title="Step reward" option={trend('Step reward',f=>f.step_reward,'Weighted rate')}/>
  </div>:tab==='rewards'?<div className="rounded-2xl border border-border bg-panel p-5">
   <h3 className="mb-3 font-semibold">Weighted live reward term rates</h3>
   <div className="grid gap-2 sm:grid-cols-2">
    {Object.entries(current.reward_terms_rate).sort((a,b)=>Math.abs(b[1])-Math.abs(a[1])).map(([key,value])=><div key={key}
     className="flex min-w-0 justify-between gap-3 rounded-lg bg-raised p-3 text-xs">
     <span className="truncate text-secondary">{key}</span><span className="tabular-nums">{fmt(value,4)}</span>
    </div>)}
   </div>
  </div>:tab==='provenance'?<div className="rounded-2xl border border-border bg-panel p-5 text-sm">
   <h3 className="mb-4 font-semibold">Live source metadata</h3>
   <p className="text-secondary">Run ID: {runId}</p>
   <p className="mt-2 text-secondary">Source: {frame?.source??'-'}</p>
   <p className="mt-2 text-secondary">Sampled at: {new Date((frame?.sampled_at??0)*1000).toLocaleString()}</p>
   <p className="mt-2 text-secondary">Dropped display frames: {frame?.dropped_display_frames??0}</p>
  </div>:<div className="rounded-2xl border border-dashed border-border bg-panel p-8 text-sm leading-relaxed text-secondary">
   These analyses require a saved, checkpoint-verified capture. The live PPO stream does not emit a full reward distribution, evaluator pass/fail flags, Rerun file, or high-rate 3D playback. Capture the policy and switch to a frozen checkpoint for those views.
  </div>}
 </div>
}
