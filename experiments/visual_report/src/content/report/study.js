export const TASKS={fexofenadine:'Fexofenadine',osimertinib:'Osimertinib'};
export const METHODS={frozen:{name:'冻结采样',short:'冻结',color:'#77828d'},reinforce:{name:'REINFORCE',short:'RL',color:'#176bb0'},reinforce_kl:{name:'REINFORCE + KL',short:'RL + KL',color:'#168266'},rf:{name:'REINFORCE',short:'RL',color:'#176bb0'},kl_0003:{name:'KL β = 0.003',color:'#67a78c'},kl_001:{name:'KL β = 0.01',short:'RL + KL',color:'#168266'},kl_003:{name:'KL β = 0.03',color:'#376e66'},kl_01:{name:'KL β = 0.1',color:'#244c48'},length_0001:{name:'长度成本 0.001',color:'#b87c28'},length_0003:{name:'长度成本 0.003',color:'#a45927'},upper_tpsa150:{name:'TPSA 软约束',color:'#8a5b9b'}};
export const CORE=['frozen','reinforce','reinforce_kl'];
export const PHASES={core:'基础对照',kl_dose:'KL 强度扫描',mechanism:'机制干预',long_budget:'长预算验证'};
export const F=(x,n=4)=>Number.isFinite(x)?x.toFixed(n):'—';
export const mean=xs=>xs.length?xs.reduce((a,b)=>a+b,0)/xs.length:null;
export const sd=xs=>xs.length>1?Math.sqrt(xs.reduce((a,b)=>a+(b-mean(xs))**2,0)/(xs.length-1)):0;
export const color=c=>METHODS[c]?.color||'#176bb0';
export const name=c=>METHODS[c]?.name||c;
export function averageCurve(rows,x,y){const groups=new Map();for(const r of rows){const k=r.condition+'|'+r[x];if(!groups.has(k))groups.set(k,[]);groups.get(k).push(r);}return [...groups.values()].map(g=>({condition:g[0].condition,[x]:g[0][x],[y]:mean(g.map(r=>r[y])),n:g.length})).sort((a,b)=>a[x]-b[x]);}
export function download(name,body,type='text/plain;charset=utf-8'){const u=URL.createObjectURL(new Blob([body],{type}));const a=document.createElement('a');a.href=u;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(u),2000);}
export function csv(rows){if(!rows.length)return '';const keys=[...new Set(rows.flatMap(Object.keys))].filter(k=>!k.toLowerCase().includes('source')&&!k.toLowerCase().includes('sha256'));const e=v=>'"'+String(v??'').replaceAll('"','""')+'"';return '\ufeff'+[keys.map(e).join(','),...rows.map(r=>keys.map(k=>e(typeof r[k]==='object'?JSON.stringify(r[k]):r[k])).join(','))].join('\n');}
export const componentNames=task=>task==='fexofenadine'?['AP 结构相似度','TPSA 下限奖励','logP 上限奖励']:['FCFP4 结构相似度','ECFP6 相似度上限','TPSA 下限奖励','logP 上限奖励'];
