import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { PPO } from '../curve-analysis/rl.snapshot.mjs';

const dir=path.dirname(fileURLToPath(import.meta.url));
const agent=new PPO(162066966), rows=[];
function localControl() {
  const cache=agent.actor.forward([0,0,0,0]);
  const scale=[1/2.4,1/2,4.7746483,1/2];
  const slope=scale.map((s,i)=>s*Array.from({length:agent.actor.hidden},(_,j)=>
    agent.actor.w[agent.actor.hidden*5+j]*(1-cache.h[j]**2)*agent.actor.w[j*5+i]).reduce((a,b)=>a+b,0));
  const grid=[];
  for(const x of [-2,-1,0,1,2]) for(const angle of [-.06,-.03,0,.03,.06]) {
    const probability=agent.probability([x,0,angle,0]);
    grid.push({x,angle,probability,meanForce:10*(2*probability-1)});
  }
  return {restLogit:cache.y,rightProbabilityAtRest:1/(1+Math.exp(-cache.y)),logitGradientPerPhysicalState:slope,grid};
}
for(let update=0;update<=40;update++) {
  if(update) agent.learn();
  rows.push({update,steps:agent.totalSteps,...localControl()});
}
fs.mkdirSync(dir,{recursive:true});
fs.writeFileSync(path.join(dir,'control-field.json'),JSON.stringify({seed:162066966,description:'Local analytic derivative of policy logit at zero physical state; counterfactual force field at zero cart and angular velocities. Diagnostic description, not a causal intervention.',stateOrder:['cartPosition','cartVelocity','poleAngle','poleAngularVelocity'],rows},null,2));
console.log(JSON.stringify(rows.filter(x=>x.update>=10&&x.update<=24).map(x=>({update:x.update,restP:x.rightProbabilityAtRest,gradient:x.logitGradientPerPhysicalState}))));
