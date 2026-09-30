import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import crypto from 'node:crypto';
import { PPO, Network, features } from '../../curve-analysis/rl.snapshot.mjs';
import { createDataset, evaluatePolicy, pairedDifference, trajectoryExamples } from './eval.mjs';

const dir=path.dirname(fileURLToPath(import.meta.url)),seed=162066966;
fs.mkdirSync(dir,{recursive:true});
const expected=JSON.parse(fs.readFileSync(path.resolve(dir,'../../curve-analysis/results.json'))).runs.find(r=>r.seed===seed);
const updates=[0,...Array.from({length:23},(_,i)=>i+8),40,80],selected=new Set(updates),agent=new PPO(seed),checkpoints=[],scores=[];
const serialize=network=>({hidden:network.hidden,w:Array.from(network.w),m:Array.from(network.m),v:Array.from(network.v),t:network.t});
const save=()=>checkpoints.push({update:agent.updates,steps:agent.totalSteps,episodes:agent.episodes.length,options:agent.options,actor:serialize(agent.actor),critic:serialize(agent.critic)});
save();
for(let update=1;update<=80;update++) {
  agent.learn();const recent=agent.episodes.slice(-20),mean=recent.reduce((s,x)=>s+x,0)/recent.length;
  scores.push(mean);if(mean!==expected.history[update-1].score)throw Error(`Score mismatch ${update}`);if(selected.has(update))save();
}
if(JSON.stringify(agent.episodes)!==JSON.stringify(expected.episodes))throw Error('Full episode history mismatch');
fs.writeFileSync(path.join(dir,'checkpoints.json'),JSON.stringify({seed,sourceSha256:crypto.createHash('sha256').update(fs.readFileSync(path.resolve(dir,'../../curve-analysis/rl.snapshot.mjs'))).digest('hex'),checkpoints}));
fs.writeFileSync(path.join(dir,'replay-verification.json'),JSON.stringify({seed,all80TrainingScoresMatch:true,allEpisodeScoresMatch:true,trainingHistory:expected.history,updates,totalSteps:agent.totalSteps},null,2));

const dataset=createDataset({count:1500,seed:'cartpole-diagnosis-independent-v1'});
fs.writeFileSync(path.join(dir,'dataset.json'),JSON.stringify({count:dataset.count,seed:dataset.seed,maxSteps:dataset.maxSteps,scheme:'Each episode has SHA256(seed + :episode: + id), first32bits little endian, as Mulberry32 seed; draw4 start values then500 timestep uniforms; shared exactly across checkpoints.',episodes:dataset.episodes.map(e=>({id:e.id,episodeSeed:e.episodeSeed,initialState:e.initialState,uniformsSha256:crypto.createHash('sha256').update(Buffer.from(e.uniforms.buffer)).digest('hex')}))}));

const started=performance.now(),results=[],trajectoryRecords=[];
for(const checkpoint of checkpoints) {
  const actor=new Network(()=>.5,checkpoint.actor.hidden);actor.w.set(checkpoint.actor.w);
  const probability=s=>1/(1+Math.exp(-actor.forward(features(s)).y));
  const sampled=evaluatePolicy(probability,dataset),deterministic=evaluatePolicy(probability,dataset,{deterministic:true});
  const record={update:checkpoint.update,steps:checkpoint.steps,sampled,deterministic};results.push(record);
  fs.writeFileSync(path.join(dir,`checkpoint-${checkpoint.update}-episodes.json`),JSON.stringify(record));
  const examples=trajectoryExamples(probability,dataset,sampled.episodes);trajectoryRecords.push({update:checkpoint.update,episodes:examples});
  console.log(JSON.stringify({update:checkpoint.update,mean:sampled.summary.returns.mean,ci95:sampled.summary.returns.ci95,cap:sampled.summary.hit500.fraction,cart:sampled.summary.causes.cart.fraction,pole:sampled.summary.causes.pole.fraction,both:sampled.summary.causes.both.fraction,elapsedSeconds:(performance.now()-started)/1000}));
}
const at=u=>results.find(r=>r.update===u),paired=(a,b)=>({from:a,to:b,...pairedDifference(at(a).sampled.episodes,at(b).sampled.episodes)});
const adjacent=results.filter(r=>r.update>=9&&r.update<=30).map(r=>paired(r.update-1,r.update));
const preDip=results.filter(r=>r.update>=12&&r.update<=16),dipWindow=results.filter(r=>r.update>=16&&r.update<=23);
const peak=preDip.reduce((a,b)=>a.sampled.summary.returns.mean>b.sampled.summary.returns.mean?a:b);
const trough=dipWindow.reduce((a,b)=>a.sampled.summary.returns.mean<b.sampled.summary.returns.mean?a:b);
const withoutDifferences=p=>({...p,differences:undefined});
const summary={seed,count:dataset.count,datasetSeed:dataset.seed,updates,runtimeSeconds:(performance.now()-started)/1000,
  metric:'Independent sampled-policy episode return, up to500steps, same1500 initial states and timestep uniforms per checkpoint; primary app action sampling preserved.',
  ciCaution:'Normal-approximation95% mean intervals from sample variance; Wilson95% cap/cause intervals. These intervals estimate frozen-policy performance on random resets/action draws, not variability across training seeds.',
  checkpoints:results.map(r=>({update:r.update,steps:r.steps,sampled:r.sampled.summary,deterministic:r.deterministic.summary})),
  fixedTrainingDipComparison:withoutDifferences(paired(16,19)),
  observedEvaluationPeakTrough:{selection:'Post hoc: largest frozen mean among updates12..16 versus smallest among16..23; unadjusted paired CI does not correct selection.',peakUpdate:peak.update,troughUpdate:trough.update,comparison:withoutDifferences(paired(peak.update,trough.update))},
  adjacentDifferences:adjacent.map(withoutDifferences),evaluationResultsFilePattern:'checkpoint-{update}-episodes.json',weightsFile:'checkpoints.json',trajectoryExamplesFile:'trajectory-examples.json'};
fs.writeFileSync(path.join(dir,'summary.json'),JSON.stringify(summary,null,2));
fs.writeFileSync(path.join(dir,'paired-differences.json'),JSON.stringify({fixedTrainingDip:paired(16,19),observedEvaluationPeakTrough:paired(peak.update,trough.update),adjacent}));
fs.writeFileSync(path.join(dir,'trajectory-examples.json'),JSON.stringify(trajectoryRecords));
console.log(JSON.stringify({done:true,runtimeSeconds:summary.runtimeSeconds,fixed16to19:summary.fixedTrainingDipComparison,observedPeakTrough:summary.observedEvaluationPeakTrough}));
