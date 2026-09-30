import fs from 'node:fs';
import crypto from 'node:crypto';
import {PPO,Network,features,random} from '../../curve-analysis/rl.snapshot.mjs';
import {createDataset,evaluatePolicy,pairedDifference} from '../evaluation/eval.mjs';
const dir=new URL('./',import.meta.url), seed=162066966, source=new URL('../../curve-analysis/rl.snapshot.mjs',import.meta.url);
const expected=JSON.parse(fs.readFileSync(new URL('../../curve-analysis/results.json',import.meta.url))).runs.find(r=>r.seed===seed);
const dataset=createDataset({count:750,seed:'cartpole-diagnosis-independent-v1',maxSteps:500});
const sigmoid=z=>1/(1+Math.exp(-z));
const recent=a=>a.episodes.slice(-20).reduce((a,b)=>a+b,0)/Math.min(20,a.episodes.length);
const snapshot=n=>{const copy=new Network(random(0),n.hidden);copy.w.set(n.w);return copy;};
function batchMetrics(actor,data,oldP){
 let surr=0,kl=0,entropy=0,clipFraction=0,activeClipFraction=0,maxKL=0,maxRatio=0,minRatio=Infinity;
 const kls=[];
 for(let i=0;i<data.length;i++){
  const d=data[i],p=Math.max(1e-15,Math.min(1-1e-15,sigmoid(actor.forward(d.x).y))),op=oldP[i];
  const ratio=Math.exp(Math.log(d.a?p:1-p)-d.logp),clippedRatio=Math.max(.8,Math.min(1.2,ratio));
  surr+=Math.min(ratio*d.adv,clippedRatio*d.adv);
  const stateKL=op*Math.log(op/p)+(1-op)*Math.log((1-op)/(1-p));
  kl+=stateKL;kls.push(stateKL);maxKL=Math.max(maxKL,stateKL);
  entropy-=p*Math.log(p)+(1-p)*Math.log(1-p);
  clipFraction+=+(Math.abs(ratio-1)>.2);
  activeClipFraction+=+((d.adv>=0&&ratio>1.2)||(d.adv<0&&ratio<.8));
  minRatio=Math.min(minRatio,ratio);maxRatio=Math.max(maxRatio,ratio);
 }
 const n=data.length;kls.sort((a,b)=>a-b);
 return {n,surrogate:surr/n,meanEntropy:entropy/n,fullMaximizedObjective:surr/n+.005*entropy/n,exactKLoldToNewMean:kl/n,exactKLoldToNewMax:maxKL,exactKLoldToNewP95:kls[Math.floor(.95*(n-1))],clipFraction:clipFraction/n,activeClipFraction:activeClipFraction/n,minRatio,maxRatio};
}
const sourceHash=crypto.createHash('sha256').update(fs.readFileSync(source)).digest('hex');
const output={manifest:{seed,node:process.version,sourceSha256:sourceHash,evaluation:{episodes:750,seed:dataset.seed,maxSteps:500,matching:'Identical presampled four-component initial states and per-step action uniforms for all frozen snapshots.',primary:'sampled policy',secondary:'deterministic argmax'},updates:[15,16],scope:'Only actor.update is wrapped for copying weights after every16 original minibatch updates; no arithmetic/RNG changes. All frozen evaluations are run after original learn(data) completes. Surrogate and exact Bernoulli KL are measured on that unchanged update rollout batch.'},validation:{},updates:[]};
const a=new PPO(seed), control=new PPO(seed);
for(let u=1;u<=14;u++){a.learn();control.learn();if(recent(a)!==expected.history[u-1].score)throw Error(`Prefix mismatch${u}`);}
output.validation.originalPrefixTrainingScoreExactThrough14=true;
for(const update of [15,16]){
 const data=a.collect(),oldActor=snapshot(a.actor),oldP=data.map(d=>sigmoid(oldActor.forward(d.x).y)),captures=[{epoch:0,actor:oldActor}];
 let calls=0;const originalUpdate=a.actor.update;
 a.actor.update=function(...args){originalUpdate.apply(this,args);calls++;if(calls%16===0)captures.push({epoch:calls/16,actor:snapshot(this)});};
 a.learn(data);a.actor.update=originalUpdate;
 control.learn();
 for(const net of ['actor','critic'])for(const attr of ['w','g','m','v'])if([...a[net][attr]].some((v,i)=>v!==control[net][attr][i]))throw Error(`Wrapped source mismatch at update${update} ${net}.${attr}`);
 if(a.actor.t!==control.actor.t||a.critic.t!==control.critic.t||JSON.stringify(a.env.state)!==JSON.stringify(control.env.state)||JSON.stringify(a.episodes)!==JSON.stringify(control.episodes))throw Error(`Wrapped state mismatch${update}`);
 if(recent(a)!==expected.history[update-1].score)throw Error(`Expected score mismatch${update}`);
 const evaluations=captures.map(({epoch,actor})=>{
  const probability=s=>sigmoid(actor.forward(features(s)).y);
  const sampled=evaluatePolicy(probability,dataset),deterministic=evaluatePolicy(probability,dataset,{deterministic:true});
  return {epoch,actorWeights:Array.from(actor.w),batch:batchMetrics(actor,data,oldP),sampled,deterministic};
 });
 for(let i=0;i<evaluations.length;i++){
  const row=evaluations[i],prior=evaluations[i-1]??row;
  row.pairedChangeFromBefore=pairedDifference(evaluations[0].sampled.episodes,row.sampled.episodes);
  row.pairedChangeFromPreviousEpoch=pairedDifference(prior.sampled.episodes,row.sampled.episodes);
 }
 output.updates.push({update,checkpointBefore:update-1,originalTrainingRollingAfter:recent(a),minibatchesPerEpoch:16,totalActorUpdateCalls:calls,rows:evaluations});
 fs.writeFileSync(new URL('epoch-results.json',dir),JSON.stringify(output));
 console.log(JSON.stringify({update,validated:true,rows:evaluations.map(r=>({epoch:r.epoch,mean:r.sampled.summary.returns.mean,meanCI:r.sampled.summary.returns.ci95,pairedChange:r.pairedChangeFromBefore.mean,pairedCI:r.pairedChangeFromBefore.ci95,detMean:r.deterministic.summary.returns.mean,cartFailure:r.sampled.summary.causes.cart.fraction,poleFailure:r.sampled.summary.causes.pole.fraction,surrogate:r.batch.surrogate,objective:r.batch.fullMaximizedObjective,kl:r.batch.exactKLoldToNewMean,clipFraction:r.batch.clipFraction}))}));
}
output.validation.wrappedActorArithmeticExactThrough16=true;
fs.writeFileSync(new URL('epoch-results.json',dir),JSON.stringify(output));
const summary={manifest:output.manifest,validation:output.validation,updates:output.updates.map(u=>({...u,rows:u.rows.map(r=>({epoch:r.epoch,batch:r.batch,sampled:r.sampled.summary,deterministic:r.deterministic.summary,pairedChangeFromBefore:{...r.pairedChangeFromBefore,differences:undefined},pairedChangeFromPreviousEpoch:{...r.pairedChangeFromPreviousEpoch,differences:undefined}}))}))};
fs.writeFileSync(new URL('epoch-summary.json',dir),JSON.stringify(summary,null,2));
