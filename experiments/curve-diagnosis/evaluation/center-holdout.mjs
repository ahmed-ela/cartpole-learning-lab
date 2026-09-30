import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Network, features } from '../../curve-analysis/rl.snapshot.mjs';
import { createDataset, evaluatePolicy, pairedDifference, meanCI } from './eval.mjs';

const dir=path.dirname(fileURLToPath(import.meta.url));
const baseline=JSON.parse(fs.readFileSync(path.join(dir,'holdout-raw.json')));
const checkpoints=JSON.parse(fs.readFileSync(path.join(dir,'checkpoints.json'))).checkpoints;
const dataset=createDataset({count:2000,seed:'cartpole-diagnosis-holdout-v1'}),results=[];
if(baseline.dataset.count!==dataset.count||baseline.dataset.seed!==dataset.seed)throw Error('Holdout dataset mismatch');
for(const update of [14,15]) {
  const checkpoint=checkpoints.find(c=>c.update===update),actor=new Network(()=>.5,checkpoint.actor.hidden);actor.w.set(checkpoint.actor.w);
  const offset=actor.forward([0,0,0,0]).y;
  const result=evaluatePolicy(s=>1/(1+Math.exp(-(actor.forward(features(s)).y-offset))),dataset);
  results.push({update,removedLogitOffset:offset,originalRestProbability:1/(1+Math.exp(-offset)),centeredRestProbability:.5,...result});
}
const original14=baseline.results.find(r=>r.update===14),original15=baseline.results.find(r=>r.update===15),centered14=results.find(r=>r.update===14),centered15=results.find(r=>r.update===15);
const originalChange=pairedDifference(original14.episodes,original15.episodes),centeredChange=pairedDifference(centered14.episodes,centered15.episodes);
const interactionDifferences=dataset.episodes.map((e,i)=>(centered15.episodes[i].reward-centered14.episodes[i].reward)-(original15.episodes[i].reward-original14.episodes[i].reward));
const interaction={direction:'(centered15 - centered14) - (original15 - original14)',...meanCI(interactionDifferences),differences:interactionDifferences};
const withinCheckpoint=results.map(result=>({update:result.update,...pairedDifference(baseline.results.find(r=>r.update===result.update).episodes,result.episodes)}));
const summary={dataset:{count:dataset.count,seed:dataset.seed,maxSteps:dataset.maxSteps},contrastDeclaredBeforeEvaluation:'At fixed checkpoints14 and15, remove only the constant zero-state logit: sigmoid(z(s)-z(0)); compare new-minus-old checkpoint performance under centering versus original. This contrast was chosen using an earlier independent cohort, before this holdout centering evaluation.',
  intervention:'Inference-time subtraction of the logit at state [0,0,0,0]; all weights and training history held fixed. This is a controller intervention, not further training.',
  original:baseline.results.filter(r=>[14,15].includes(r.update)).map(r=>({update:r.update,summary:r.summary})),
  centered:results.map(r=>({update:r.update,removedLogitOffset:r.removedLogitOffset,originalRestProbability:r.originalRestProbability,centeredRestProbability:r.centeredRestProbability,summary:r.summary})),
  originalChange:{...originalChange,differences:undefined},centeredChange:{...centeredChange,differences:undefined},interaction:{...interaction,differences:undefined},withinCheckpoint:withinCheckpoint.map(p=>({...p,differences:undefined})),
  limitation:'Evidence that the changed constant logit is a major causal component of this particular performance drop on these resets. It does not establish all optimizer-level causes or prove centering is an optimal fix for all checkpoints/seeds.'};
fs.writeFileSync(path.join(dir,'holdout-validation.json'),JSON.stringify(summary,null,2));
fs.writeFileSync(path.join(dir,'centered-holdout-raw.json'),JSON.stringify({results,originalChange,centeredChange,interaction,withinCheckpoint}));
console.log(JSON.stringify({original:summary.original.map(r=>({update:r.update,mean:r.summary.returns.mean})),centered:summary.centered.map(r=>({update:r.update,mean:r.summary.returns.mean})),originalChange:summary.originalChange,centeredChange:summary.centeredChange,interaction:summary.interaction}));
