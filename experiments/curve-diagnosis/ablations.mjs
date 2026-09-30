import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { PPO, features } from '../curve-analysis/rl.snapshot.mjs';
import { createDataset, evaluatePolicy, pairedDifference } from './evaluation/eval.mjs';

const dir=path.dirname(fileURLToPath(import.meta.url));
const agent=new PPO(162066966);
const dataset=createDataset({count:1500,seed:'cartpole-diagnosis-independent-v1'});
const selected=new Set([14,15,16,18,19,20,24,30,40]);
const conditions=['original','center-logit','ignore-cart-position','center-and-ignore-cart-position'];
const rows=[];
const sigmoid=y=>1/(1+Math.exp(-y));
const began=performance.now();
for(let update=0;update<=40;update++) {
  if(update)agent.learn();
  if(!selected.has(update))continue;
  const restLogit=agent.actor.forward([0,0,0,0]).y;
  let reference;
  for(const condition of conditions) {
    const center=condition.includes('center'),ignorePosition=condition.includes('ignore');
    const probability=state=> {
      const x=features(state);
      if(ignorePosition)x[0]=0;
      return sigmoid(agent.actor.forward(x).y-(center?restLogit:0));
    };
    const evaluation=evaluatePolicy(probability,dataset);
    if(condition==='original')reference=evaluation;
    const delta=pairedDifference(reference.episodes,evaluation.episodes);
    const compactEpisodes=evaluation.episodes.map(e=>({id:e.id,episodeSeed:e.episodeSeed,reward:e.reward,cause:e.cause,finalState:e.finalState}));
    rows.push({update,trainingSteps:agent.totalSteps,condition,restLogit,originalProbabilityRightAtRest:sigmoid(restLogit),summary:evaluation.summary,
      gainVsOriginal:{...delta,differences:undefined},episodes:compactEpisodes});
    console.log(JSON.stringify({update,condition,mean:evaluation.summary.returns.mean,gain:delta.mean,gainCI95:delta.ci95,cartFailures:evaluation.summary.causes.cart.n,poleFailures:evaluation.summary.causes.pole.n}));
  }
}
const output={seed:162066966,dataset:{count:dataset.count,seed:dataset.seed,maxSteps:dataset.maxSteps},conditions,
  interpretation:'Frozen-policy interventions, no learning. Center-logit subtracts the policy logit at state zero, setting p(right|zero)=0.5 at every state with a constant logit shift. Ignore-cart-position sets only the cart-position network input to zero. Other weights and all physical state transitions remain unchanged. Paired common random numbers match starts and per-timestep action uniforms. These interventions test causal behavior contributions, not the origin of training errors.',runtimeSeconds:(performance.now()-began)/1000,rows};
fs.writeFileSync(path.join(dir,'ablations.json'),JSON.stringify(output));
fs.writeFileSync(path.join(dir,'ablation-summary.json'),JSON.stringify({...output,rows:rows.map(({episodes,...rest})=>rest)},null,2));
