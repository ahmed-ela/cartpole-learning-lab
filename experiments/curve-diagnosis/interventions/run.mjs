import fs from 'node:fs';
import crypto from 'node:crypto';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const dir=path.dirname(fileURLToPath(import.meta.url));
const sourcePath=path.resolve(dir,'../../curve-analysis/rl.snapshot.mjs');
const source=fs.readFileSync(sourcePath,'utf8');
const hash=s=>crypto.createHash('sha256').update(s).digest('hex');
const end=source.indexOf('\n}\n\nexport class CartPole');
if(end<0 || !source.startsWith('export function random(seed) {\n  return () =>'))throw Error('Snapshot RNG shape changed');
// Add an RNG-state accessor only. The numeric body and every PPO operation stay identical.
const randomSource=source.slice(0,end+2).replace('  return () =>','  const fn = () =>').replace(/\n}$/, '\n  fn.getState = () => seed;\n  return fn;\n}');
const instrumented=randomSource+source.slice(end+2);
const instrumentedPath=path.join(dir,'rl.rng-accessor.mjs');
fs.writeFileSync(instrumentedPath,instrumented);
const {PPO,Network,CartPole,random}=await import(pathToFileURL(instrumentedPath));
const {PPO:OriginalPPO}=await import(pathToFileURL(sourcePath));

const seed=162066966, counts=1000, start=12, endUpdate=24, learningRates=[.001,.0005,.00025,0];
const expectedRun=JSON.parse(fs.readFileSync(path.resolve(dir,'../../curve-analysis/results.json'),'utf8')).runs.find(r=>r.seed===seed);
if(!expectedRun)throw Error('Expected selected original run not found');
const snapshotNetwork=n=>({hidden:n.hidden,w:Array.from(n.w),g:Array.from(n.g),m:Array.from(n.m),v:Array.from(n.v),t:n.t});
const networkHash=n=>hash(JSON.stringify(snapshotNetwork(n)));
function cloneNetwork(n){const c=Object.create(Network.prototype);c.hidden=n.hidden;for(const k of ['w','g','m','v'])c[k]=new Float64Array(n[k]);c.t=n.t;return c;}
function cloneAgent(a){const c=Object.create(PPO.prototype);c.rng=random(a.rng.getState());c.actor=cloneNetwork(a.actor);c.critic=cloneNetwork(a.critic);c.env=Object.create(CartPole.prototype);c.env.rng=c.rng;c.env.state=[...a.env.state];c.env.steps=a.env.steps;c.options={...a.options};for(const k of ['totalSteps','updates'])c[k]=a[k];c.episodes=[...a.episodes];c.buffer=a.buffer.map(d=>({...d,x:[...d.x]}));return c;}
function sameState(a,b){return networkHash(a.actor)===networkHash(b.actor)&&networkHash(a.critic)===networkHash(b.critic)&&JSON.stringify(a.episodes)===JSON.stringify(b.episodes)&&JSON.stringify(a.env.state)===JSON.stringify(b.env.state)&&a.totalSteps===b.totalSteps&&a.updates===b.updates;}
function recent(a){const r=a.episodes.slice(-20);return r.reduce((s,x)=>s+x,0)/r.length;}
function verifyExpected(a){const exp=expectedRun.history[a.updates-1];if(exp.score!==recent(a)||exp.episodeCount!==a.episodes.length||exp.steps!==a.totalSteps)throw Error(`Replay differs at update ${a.updates}`);}
function metrics(a,data,oldP){let kl=0,clipFraction=0,activeClipFraction=0,surr=0,criticMSE=0,entropy=0,maxRatio=0,minRatio=Infinity,clampCount=0;const o=a.options;
 for(let i=0;i<data.length;i++){const d=data[i];let p=1/(1+Math.exp(-a.actor.forward(d.x).y));const raw=p;p=Math.max(1e-15,Math.min(1-1e-15,p));if(raw!==p)clampCount++;const op=oldP[i];kl+=op*Math.log(op/p)+(1-op)*Math.log((1-op)/(1-p));const ratio=Math.exp(Math.log(d.a?p:1-p)-d.logp);const cr=Math.max(1-o.clip,Math.min(1+o.clip,ratio));clipFraction+=Math.abs(ratio-1)>o.clip;activeClipFraction+=(d.adv>=0&&ratio>1+o.clip)||(d.adv<0&&ratio<1-o.clip);surr+=Math.min(ratio*d.adv,cr*d.adv);criticMSE+=(a.critic.forward(d.x).y-d.ret)**2;entropy-=p*Math.log(p)+(1-p)*Math.log(1-p);minRatio=Math.min(minRatio,ratio);maxRatio=Math.max(maxRatio,ratio);}
 return {klOldToNew:kl/data.length,clipFraction:clipFraction/data.length,activeClipFraction:activeClipFraction/data.length,surrogate:surr/data.length,criticMSE:criticMSE/data.length,entropy:entropy/data.length,minRatio,maxRatio,probabilityClampCount:clampCount};}
function describeScores(scores){const mean=scores.reduce((s,x)=>s+x,0)/scores.length;const variance=scores.reduce((s,x)=>s+(x-mean)**2,0)/(scores.length-1);return {n:scores.length,mean,sd:Math.sqrt(variance),se:Math.sqrt(variance/scores.length),success500:scores.filter(x=>x===500).length,min:Math.min(...scores),max:Math.max(...scores)};}
function bootstrapDifference(after,before,bootstrapSeed=991231){const diffs=after.map((x,i)=>x-before[i]),mean=diffs.reduce((s,x)=>s+x,0)/diffs.length,rng=random(bootstrapSeed),means=[];
 for(let b=0;b<2000;b++){let m=0;for(let i=0;i<diffs.length;i++)m+=diffs[Math.floor(rng()*diffs.length)];means.push(m/diffs.length);}means.sort((a,b)=>a-b);return {mean,bootstrapPaired95:[means[49],means[1949]],episodes:diffs.length};}
const out={manifest:{timestamp:new Date().toISOString(),seed,sourcePath:path.relative(path.resolve(dir,'../../..'),sourcePath),sourceSha256:hash(source),instrumentedSha256:hash(instrumented),node:process.version,updatesTested:Array.from({length:endUpdate-start+1},(_,i)=>start+i),interpretation:'Each tested update forks exactly the same complete original trainer state, collects exactly the same 1024-step rollout batch, and uses the same eight epoch/minibatch permutations. Actor learning rate is the only intervention. Critic learning rate stays .001. Independent evaluation uses common start states and per-episode, per-timestep action uniforms. Single-update results are conditional on the one selected seed and batch, not an algorithm-wide effect. Confidence intervals are pointwise paired bootstrap intervals, not simultaneous familywise intervals.',evaluationEpisodes:counts,bootstrapReplicates:2000,learningRates},validation:{},singleUpdates:[],branches:[],targetedBranches:[]};
fs.writeFileSync(path.join(dir,'manifest.json'),JSON.stringify(out.manifest,null,2));

// Compare unmodified and accessor-only implementations through the whole original prefix.
const replay=new PPO(seed),original=new OriginalPPO(seed);
for(let u=1;u<start;u++){replay.learn();original.learn();verifyExpected(replay);if(!sameState(replay,original))throw Error(`Accessor changed source behavior at update ${u}`);}
out.validation.originalSourcePrefixExactThrough=start-1;
const fork=cloneAgent(replay);fork.learn();const control=cloneAgent(replay);control.learn();if(!sameState(fork,control)||fork.rng.getState()!==control.rng.getState())throw Error('Clone not exact');
out.validation.cloneNextUpdateExactlyReproduced=true;
console.log(JSON.stringify({phase:'prefix-validated',update:replay.updates,trainingScore:recent(replay)}));

// The reusable evaluator is authored independently by the checkpoint-evaluation agent.
const {createDataset,evaluatePolicy}=await import(pathToFileURL(path.resolve(dir,'../evaluation/eval.mjs')));
const dataset=createDataset({count:counts,seed:'cartpole-intervention-independent-v1',maxSteps:500});
out.manifest.evaluationDatasetSeed='cartpole-intervention-independent-v1';
const evaluate=a=>{const result=evaluatePolicy(s=>a.probability(s),dataset,{deterministic:false});const scores=result.episodes.map(e=>e.reward);if(scores.some(x=>!Number.isFinite(x)))throw Error('Unrecognized evaluation scores');return {...describeScores(scores),failureCauses:result.summary.causes,scores};};
let firstSignificantRegression=null;
const prefixes=new Map();
for(let update=start;update<=endUpdate;update++){
 prefixes.set(update-1,cloneAgent(replay));
 const before=evaluate(replay),batchState=cloneAgent(replay),data=batchState.collect();
 const oldP=data.map(d=>1/(1+Math.exp(-replay.actor.forward(d.x).y)));
 const beforeBatchMetrics=metrics(replay,data,oldP);
 const variants=[];
 for(const actorLR of learningRates){const a=cloneAgent(batchState);a.options.actorLR=actorLR;a.learn(data);const after=evaluate(a),m=metrics(a,data,oldP);variants.push({actorLR,trainingSteps:a.totalSteps,actorHash:networkHash(a.actor),criticHash:networkHash(a.critic),evaluation:after,changeFromBefore:bootstrapDifference(after.scores,before.scores),metrics:{...m,surrogateGain:m.surrogate-beforeBatchMetrics.surrogate,criticMSEBefore:beforeBatchMetrics.criticMSE,criticMSEChange:m.criticMSE-beforeBatchMetrics.criticMSE},agent:a});}
 const def=variants[0];if(firstSignificantRegression===null&&def.changeFromBefore.bootstrapPaired95[1]<0)firstSignificantRegression=update;
 for(const v of variants){v.changeFromDefault=bootstrapDifference(v.evaluation.scores,def.evaluation.scores,777231);delete v.agent;}
 replay.learn();verifyExpected(replay);if(networkHash(replay.actor)!==def.actorHash||networkHash(replay.critic)!==def.criticHash)throw Error(`Control exactness failed at update ${update}`);
 out.singleUpdates.push({update,checkpointBefore:update-1,trainingScoreAfter:recent(replay),before,variants});
 fs.writeFileSync(path.join(dir,'results.json'),JSON.stringify(out));
 console.log(JSON.stringify({phase:'single-update',update,beforeMean:before.mean,variants:variants.map(v=>({actorLR:v.actorLR,mean:v.evaluation.mean,change:v.changeFromBefore.mean,ci:v.changeFromBefore.bootstrapPaired95,KL:v.metrics.klOldToNew,surrGain:v.metrics.surrogateGain}))}));
}
out.validation.originalTrainingHistoryExactThrough=endUpdate;
out.validation.allDefaultUpdatesActorAndCriticHashesExact=true;
out.validation.firstSignificantRegression=firstSignificantRegression;
// Choose the earliest independently evaluated, single-update regression; no tuning to a favored outcome.
const branchUpdate=firstSignificantRegression??17;
const prefix=prefixes.get(branchUpdate-1);if(!prefix)throw Error('Chosen branch prefix unavailable');
out.manifest.branchCheckpoint=branchUpdate-1;out.manifest.branchSelection=firstSignificantRegression===null?'No significant single-update regression found; prespecified training-dip start checkpoint16 fallback.':'Earliest default single-update regression with paired95 interval entirely below zero in prespecified updates12..24.';
const branchRates=[.001,.0005,.00025],branchAgents=branchRates.map(actorLR=>{const a=cloneAgent(prefix);a.options.actorLR=actorLR;return a;});
for(let update=branchUpdate;update<=30;update++){
 const points=[];
 for(let i=0;i<branchRates.length;i++){const a=branchAgents[i];a.learn();if(i===0)verifyExpected(a);points.push({actorLR:branchRates[i],evaluation:evaluate(a),trainingScore:recent(a),trainingEpisodes:a.episodes.length});}
 for(const point of points)point.changeFromDefault=bootstrapDifference(point.evaluation.scores,points[0].evaluation.scores,662341);
 out.branches.push({update,steps:update*1024,variants:points});fs.writeFileSync(path.join(dir,'results.json'),JSON.stringify(out));
 console.log(JSON.stringify({phase:'branch',update,variants:points.map(v=>({actorLR:v.actorLR,mean:v.evaluation.mean,trainingScore:v.trainingScore,change:v.changeFromDefault.mean,ci:v.changeFromDefault.bootstrapPaired95}))}));
}
out.validation.defaultBranchReproducesOriginalHistoryThrough=30;
// The large regression independently identified at14→15 is also studied directly,
// preserving the earlier11-prefix branch and all intermediate outcomes above.
const targetCheckpoint=14,targetAgents=branchRates.map(actorLR=>{const a=cloneAgent(prefixes.get(targetCheckpoint));a.options.actorLR=actorLR;return a;});
out.manifest.targetedBranchCheckpoint=targetCheckpoint;
out.manifest.targetedBranchSelection='Independently identified main early regression14→15; fixed exact original checkpoint14, all outcomes through30 retained. An additional conditional case study, not a best-branch search.';
for(let update=targetCheckpoint+1;update<=30;update++){
 const points=[];
 for(let i=0;i<branchRates.length;i++){const a=targetAgents[i];a.learn();if(i===0)verifyExpected(a);points.push({actorLR:branchRates[i],evaluation:evaluate(a),trainingScore:recent(a),trainingEpisodes:a.episodes.length});}
 for(const point of points)point.changeFromDefault=bootstrapDifference(point.evaluation.scores,points[0].evaluation.scores,662341);
 out.targetedBranches.push({checkpoint:targetCheckpoint,update,steps:update*1024,variants:points});fs.writeFileSync(path.join(dir,'results.json'),JSON.stringify(out));
 console.log(JSON.stringify({phase:'targeted-branch',checkpoint:targetCheckpoint,update,variants:points.map(v=>({actorLR:v.actorLR,mean:v.evaluation.mean,trainingScore:v.trainingScore,change:v.changeFromDefault.mean,ci:v.changeFromDefault.bootstrapPaired95}))}));
}
// Exact source-vs-accessor regression check over the whole prefix and branch horizon.
const checkSource=new OriginalPPO(seed),checkAccessor=new PPO(seed);
for(let u=1;u<=30;u++){checkSource.learn();checkAccessor.learn();if(!sameState(checkSource,checkAccessor))throw Error(`Full source exactness failed at update${u}`);}
out.validation.originalSourceAndAccessorExactlyAgreeThrough=30;
fs.writeFileSync(path.join(dir,'manifest.json'),JSON.stringify(out.manifest,null,2));
fs.writeFileSync(path.join(dir,'results.json'),JSON.stringify(out));
const summarizeBranch=p=>({checkpoint:p.checkpoint??branchUpdate-1,update:p.update,variants:p.variants.map(v=>({actorLR:v.actorLR,mean:v.evaluation.mean,trainingScore:v.trainingScore,changeFromDefault:v.changeFromDefault}))});
const summary={manifest:out.manifest,validation:out.validation,singleUpdates:out.singleUpdates.map(p=>({update:p.update,before:p.before.mean,trainingScore:p.trainingScoreAfter,variants:p.variants.map(v=>({actorLR:v.actorLR,mean:v.evaluation.mean,change:v.changeFromBefore,changeFromDefault:v.changeFromDefault,metrics:v.metrics}))})),branches:out.branches.map(summarizeBranch),targetedBranches:out.targetedBranches.map(summarizeBranch)};
fs.writeFileSync(path.join(dir,'summary.json'),JSON.stringify(summary,null,2));
console.log(JSON.stringify({done:true,firstSignificantRegression,branchCheckpoint:branchUpdate-1}));
