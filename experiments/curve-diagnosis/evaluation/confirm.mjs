import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Network, features } from '../../curve-analysis/rl.snapshot.mjs';
import { createDataset, evaluatePolicy, pairedDifference } from './eval.mjs';

const dir=path.dirname(fileURLToPath(import.meta.url));
const selected=[14,15,16,18,19,20,40];
const checkpoints=JSON.parse(fs.readFileSync(path.join(dir,'checkpoints.json'))).checkpoints.filter(c=>selected.includes(c.update));
const comparisons=[[14,15],[14,16],[16,18],[19,20]];
const dataset=createDataset({count:2000,seed:'cartpole-diagnosis-holdout-v1'}),results=[];
for(const checkpoint of checkpoints) {
  const actor=new Network(()=>.5,checkpoint.actor.hidden);actor.w.set(checkpoint.actor.w);
  const result=evaluatePolicy(s=>1/(1+Math.exp(-actor.forward(features(s)).y)),dataset);
  results.push({update:checkpoint.update,...result});
}
const paired=comparisons.map(([from,to])=>({from,to,...pairedDifference(results.find(r=>r.update===from).episodes,results.find(r=>r.update===to).episodes)}));
fs.writeFileSync(path.join(dir,'holdout-raw.json'),JSON.stringify({dataset:{count:dataset.count,seed:dataset.seed,maxSteps:dataset.maxSteps},results,paired}));
const summary={count:dataset.count,datasetSeed:dataset.seed,selection:'Checkpoint contrasts fixed before collecting this independent holdout cohort; main cohort used to identify the contrasts.',checkpoints:results.map(r=>({update:r.update,...r.summary})),comparisons:paired.map(p=>({...p,differences:undefined}))};
fs.writeFileSync(path.join(dir,'holdout-summary.json'),JSON.stringify(summary,null,2));
console.log(JSON.stringify(summary.comparisons));
