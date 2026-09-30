import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads';
import fs from 'node:fs';
import crypto from 'node:crypto';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const dir = path.dirname(fileURLToPath(import.meta.url));
const originalSource = path.resolve(dir, '../../dist/rl.js');
const snapshot = path.join(dir,'rl.snapshot.mjs');
const source = fs.existsSync(snapshot)?snapshot:originalSource;
const { PPO } = await import(pathToFileURL(source));
const hash = crypto.createHash('sha256').update(fs.readFileSync(source)).digest('hex');
const N = 128, UPDATES = 80, WORKERS = 3, EVAL_COUNT = 20;
const EVAL_AT = new Set([10,20,40,80]);
const quantile = (xs,q) => { if(!xs.length)return null; const a=[...xs].sort((x,y)=>x-y),p=(a.length-1)*q,i=Math.floor(p); return a[i]+(a[Math.min(a.length-1,i+1)]-a[i])*(p-i); };
const evalSeed = (seed,update) => crypto.createHash('sha256').update(`cartpole-curve-evaluation:v1:${seed}:${update}`).digest().readUInt32LE(0);

function run(seed) {
  const started=performance.now(), agent=new PPO(seed), history=[];
  const initialEval=agent.evaluate(evalSeed(seed,0),EVAL_COUNT,false);
  const evaluation=[{update:0,steps:0,seed:evalSeed(seed,0),...initialEval}];
  let prevEpisodes=0;
  for(let update=1;update<=UPDATES;update++) {
    agent.learn();
    const recent=agent.episodes.slice(-20),from=agent.episodes.length-recent.length;
    history.push({update,steps:agent.totalSteps,score:recent.reduce((s,x)=>s+x,0)/recent.length,
      episodeCount:agent.episodes.length,recentEpisodeIds:recent.map((_,i)=>from+i+1),recentEpisodeScores:recent,
      newlyCompletedEpisodeIds:agent.episodes.slice(prevEpisodes).map((_,i)=>prevEpisodes+i+1),
      newlyCompletedEpisodeScores:agent.episodes.slice(prevEpisodes)});
    prevEpisodes=agent.episodes.length;
    if(EVAL_AT.has(update))evaluation.push({update,steps:agent.totalSteps,seed:evalSeed(seed,update),...agent.evaluate(evalSeed(seed,update),EVAL_COUNT,false)});
  }
  return {seed,durationMs:performance.now()-started,history,evaluation,episodes:agent.episodes};
}

function describeDips(history) {
  const dips=[];let start=null;
  for(let i=1;i<history.length;i++) {
    if(history[i].score<history[i-1].score-1e-12) {if(start===null)start=i-1;}
    else if(start!==null) {dips.push({startUpdate:history[start].update,troughUpdate:history[i-1].update,startScore:history[start].score,troughScore:history[i-1].score,magnitude:history[start].score-history[i-1].score});start=null;}
  }
  if(start!==null)dips.push({startUpdate:history[start].update,troughUpdate:history.at(-1).update,startScore:history[start].score,troughScore:history.at(-1).score,magnitude:history[start].score-history.at(-1).score});
  let peak=history[0].score,maxDrawdown=0;
  for(const p of history){peak=Math.max(peak,p.score);maxDrawdown=Math.max(maxDrawdown,peak-p.score);}
  return {decreasingTransitions:history.slice(1).filter((p,i)=>p.score<history[i].score-1e-12).length,contiguousDips:dips.length,substantialDips25:dips.filter(p=>p.magnitude>=25).length,maxContiguousDip:Math.max(0,...dips.map(p=>p.magnitude)),maxDrawdown,dips};
}

function wilson(hits,total) {const z=1.959963984540054,p=hits/total,d=1+z*z/total,c=(p+z*z/(2*total))/d,w=z*Math.sqrt(p*(1-p)/total+z*z/(4*total*total))/d;return [c-w,c+w];}

function cluster(runs) {
  const curves=runs.map(r=>r.history.map(h=>h.score/500));
  const dist=curves.map(a=>curves.map(b=>Math.sqrt(a.reduce((s,x,i)=>s+(x-b[i])**2,0))));
  let prng=12345;const rand=()=>{prng=(Math.imul(prng,1664525)+1013904223)>>>0;return prng/4294967296;};
  function assignment(medoids){return dist.map(row=>{let choices=medoids.map((m,c)=>({c,m,d:row[m]})).sort((a,b)=>a.d-b.d||a.m-b.m);return {cluster:choices[0].c,d1:choices[0].d,d2:choices[1].d};});}
  let best=null;const restarts=[];
  for(let start=0;start<32;start++) {
    let medoids=[Math.floor(rand()*N)];
    // Random weighted seeding, followed by full best-swap PAM refinement.
    while(medoids.length<3){const weights=dist.map((row,i)=>medoids.includes(i)?0:Math.min(...medoids.map(m=>row[m]))**2);let r=rand()*weights.reduce((s,x)=>s+x,0),next=N-1;for(let i=0;i<N;i++){r-=weights[i];if(r<=0){next=i;break;}}medoids.push(next);}
    medoids.sort((a,b)=>a-b);let assignments=assignment(medoids),cost=assignments.reduce((s,a)=>s+a.d1,0),iterations=0;
    while(true) {
      let chosen=null,deltaBest=-1e-10;
      for(let c=0;c<3;c++)for(let candidate=0;candidate<N;candidate++)if(!medoids.includes(candidate)) {
        let delta=0;
        for(let i=0;i<N;i++){const a=assignments[i],d=dist[i][candidate];delta+=(a.cluster===c?Math.min(a.d2,d):Math.min(a.d1,d))-a.d1;}
        if(delta<deltaBest){deltaBest=delta;chosen={c,candidate};}
      }
      if(!chosen)break;
      medoids[chosen.c]=chosen.candidate;medoids.sort((a,b)=>a-b);assignments=assignment(medoids);cost=assignments.reduce((s,a)=>s+a.d1,0);iterations++;
    }
    restarts.push({medoidSeeds:medoids.map(i=>runs[i].seed),cost,iterations});
    if(best===null||cost<best.cost-1e-10)best={medoids,assignments,cost};
  }
  // Display clusters ordered by median score at 20 updates.
  const ordered=best.medoids.map((m,c)=>({m,c,members:runs.filter((_,i)=>best.assignments[i].cluster===c)})).sort((a,b)=>quantile(b.members.map(r=>r.history[19].score),.5)-quantile(a.members.map(r=>r.history[19].score),.5));
  const groups=ordered.map((g,index)=> {
    const memberSet=new Set(g.members.map(r=>r.seed));
    const firstHits=g.members.map(r=>r.history.find(h=>h.score>=475)?.steps??null);
    const observed=firstHits.filter(x=>x!==null).sort((a,b)=>a-b);
    const kmMedian=observed.length>=Math.ceil(g.members.length/2)?observed[Math.ceil(g.members.length/2)-1]:null;
    const dips=g.members.map(r=>describeDips(r.history));
    return {id:index+1,medoidSeed:runs[g.m].seed,memberSeeds:[...memberSet],n:g.members.length,share:g.members.length/N,shareWilson95:wilson(g.members.length,N),
      curve:runs[g.m].history.map(h=>({update:h.update,steps:h.steps,score:h.score})),
      quantiles:runs[0].history.map((h,i)=>({update:h.update,steps:h.steps,p10:quantile(g.members.map(r=>r.history[i].score),.1),p25:quantile(g.members.map(r=>r.history[i].score),.25),median:quantile(g.members.map(r=>r.history[i].score),.5),p75:quantile(g.members.map(r=>r.history[i].score),.75),p90:quantile(g.members.map(r=>r.history[i].score),.9)})),
      threshold475:{reached:observed.length,notReached:g.members.length-observed.length,kmMedianSteps:kmMedian,medianStepsAmongReached:quantile(observed,.5),finalBelowThreshold:g.members.filter(r=>r.history.at(-1).score<475).length},
      dips:{decreasingTransitions:{min:Math.min(...dips.map(x=>x.decreasingTransitions)),median:quantile(dips.map(x=>x.decreasingTransitions),.5),max:Math.max(...dips.map(x=>x.decreasingTransitions))},contiguousDips:{min:Math.min(...dips.map(x=>x.contiguousDips)),median:quantile(dips.map(x=>x.contiguousDips),.5),max:Math.max(...dips.map(x=>x.contiguousDips))},substantialDips25:{min:Math.min(...dips.map(x=>x.substantialDips25)),median:quantile(dips.map(x=>x.substantialDips25),.5),max:Math.max(...dips.map(x=>x.substantialDips25))},maxDrawdown:{min:Math.min(...dips.map(x=>x.maxDrawdown)),median:quantile(dips.map(x=>x.maxDrawdown),.5),max:Math.max(...dips.map(x=>x.maxDrawdown))},medoid:describeDips(runs[g.m].history)}};
  });
  return {criterion:'PAM k-medoids, k=3, Euclidean distance over 80 aligned recent20 scores / 500; 32 deterministic weighted-initialization restarts, best improving full swaps; forced three groups are a descriptive summary, not evidence of exactly three natural types.',objective:best.cost,restarts,groups};
}

if(!isMainThread) {
  for(const seed of workerData.seeds)parentPort.postMessage(run(seed));
} else {
  fs.mkdirSync(dir,{recursive:true});
  const seedsFile=path.join(dir,'seeds.json');let seeds;
  if(fs.existsSync(seedsFile)){seeds=JSON.parse(fs.readFileSync(seedsFile));if(seeds.length!==N)throw Error('Expected 128 seeds');}
  else {const unique=new Set();while(unique.size<N)unique.add(crypto.randomBytes(4).readUInt32LE(0));seeds=[...unique];fs.writeFileSync(seedsFile,JSON.stringify(seeds,null,2));}
  const started=performance.now(), timestamp=new Date().toISOString(),runs=[];
  const stream=fs.createWriteStream(path.join(dir,'runs.jsonl'));
  const manifest={timestamp,node:process.version,source:path.relative(path.resolve(dir,'../..'),source),sourceSha256:hash,n:N,updates:UPDATES,stepsPerRun:81920,workers:WORKERS,seedSampling:'128 unique uniformly sampled 32-bit seeds from crypto.randomBytes; reproducible values in seeds.json',settings:new PPO(0).options,score:'Arithmetic mean of most recent 20 completed training episode returns after each PPO update; completion score capped at 500; episodes in progress excluded.',evaluation:'20 fresh sampled-policy episodes at update 0, 10, 20, 40, 80, separate RNG from training; SHA-256-derived evaluation seed for each training seed/update.',dipDefinition:'Decrease between adjacent update scores; contiguous decreasing updates form one dip; substantial dip means contiguous drop >=25 points. Drawdown is peak-to-subsequent-trough over full curve.'};
  fs.writeFileSync(path.join(dir,'manifest.json'),JSON.stringify(manifest,null,2));
  if(source!==snapshot)fs.copyFileSync(source,snapshot);
  await Promise.all(Array.from({length:WORKERS},(_,worker)=>new Promise((resolve,reject)=>{
    const thread=new Worker(new URL(import.meta.url),{workerData:{seeds:seeds.filter((_,i)=>i%WORKERS===worker)}});
    thread.on('message',r=>{runs.push(r);stream.write(JSON.stringify(r)+'\n');if(runs.length%8===0)console.log(JSON.stringify({completed:runs.length,total:N,elapsedSeconds:(performance.now()-started)/1000}));});
    thread.on('error',reject);thread.on('exit',code=>code?reject(Error(`Worker exited ${code}`)):resolve());
  })));
  stream.end();runs.sort((a,b)=>seeds.indexOf(a.seed)-seeds.indexOf(b.seed));
  const overall=runs[0].history.map((h,i)=>({update:h.update,steps:h.steps,p10:quantile(runs.map(r=>r.history[i].score),.1),median:quantile(runs.map(r=>r.history[i].score),.5),p90:quantile(runs.map(r=>r.history[i].score),.9)}));
  const checkpoint=runs.map(r=>r.history[19].score),histogram=Array.from({length:10},(_,i)=>({from:i*50,to:(i+1)*50,n:checkpoint.filter(x=>x>=i*50&&(x<(i+1)*50||i===9&&x<=500)).length}));
  const clustering=cluster(runs),finished=performance.now();
  const results={manifest:{...manifest,runtimeSeconds:(finished-started)/1000},runs,overall,checkpointHistogram:{update:20,steps:20480,bins:histogram,scores:checkpoint,mean:checkpoint.reduce((s,x)=>s+x,0)/N,median:quantile(checkpoint,.5),p10:quantile(checkpoint,.1),p90:quantile(checkpoint,.9)},clustering,
    threshold475:{everReached:runs.filter(r=>r.history.some(h=>h.score>=475)).length,neverReached:runs.filter(r=>r.history.every(h=>h.score<475)).length,finalBelowThreshold:runs.filter(r=>r.history.at(-1).score<475).length},
    shareIntervalCaution:'Wilson binomial intervals condition on the chosen fixed cluster assignment and do not include uncertainty from fitting cluster boundaries to this same sample.'};
  fs.writeFileSync(path.join(dir,'results.json'),JSON.stringify(results));
  fs.writeFileSync(path.join(dir,'summary.json'),JSON.stringify({...results,runs:undefined,clustering:{...clustering,restarts:undefined}},null,2));
  console.log(JSON.stringify({done:true,runtimeSeconds:results.manifest.runtimeSeconds,threshold475:results.threshold475,groups:clustering.groups.map(g=>({id:g.id,seed:g.medoidSeed,n:g.n,share:g.share,wilson:g.shareWilson95,threshold:g.threshold475,dips:g.dips}))}));
}
