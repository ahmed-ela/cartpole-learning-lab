import crypto from 'node:crypto';
import { CartPole, random } from '../../curve-analysis/rl.snapshot.mjs';

export const ANGLE_LIMIT=12*Math.PI/180;
export const quantile=(values,q)=>{if(!values.length)return null;const a=[...values].sort((x,y)=>x-y),p=(a.length-1)*q,i=Math.floor(p);return a[i]+(a[Math.ceil(p)]-a[i])*(p-i);};
export function meanCI(values,{bounds=null}={}) {
  if(!values.length)return {n:0,mean:null,sd:null,se:null,ci95:null};
  const n=values.length,mean=values.reduce((s,x)=>s+x,0)/n;
  const variance=n>1?values.reduce((s,x)=>s+(x-mean)**2,0)/(n-1):0;
  const sd=Math.sqrt(variance),se=sd/Math.sqrt(n),z=1.959963984540054;
  let ci95=[mean-z*se,mean+z*se];if(bounds)ci95=[Math.max(bounds[0],ci95[0]),Math.min(bounds[1],ci95[1])];
  return {n,mean,sd,se,ci95,method:'Normal approximation for the mean, from the sample variance; not a claim that returns or learning curves are Gaussian.'};
}
export function wilson(hits,total) {const z=1.959963984540054,p=hits/total,d=1+z*z/total,c=(p+z*z/(2*total))/d,w=z*Math.sqrt(p*(1-p)/total+z*z/(4*total*total))/d;return [c-w,c+w];}

export function createDataset({count=1500,seed='cartpole-diagnosis-independent-v1',maxSteps=500}={}) {
  const episodes=Array.from({length:count},(_,id)=> {
    const episodeSeed=crypto.createHash('sha256').update(`${seed}:episode:${id}`).digest().readUInt32LE(0),rng=random(episodeSeed);
    const initialState=Array.from({length:4},()=> (rng()-.5)*.1);
    const uniforms=Float64Array.from({length:maxSteps},()=>rng());
    return {id,episodeSeed,initialState,uniforms};
  });
  return {count,seed,maxSteps,episodes};
}

export function evaluatePolicy(probability,dataset,{deterministic=false,recordTrajectories=false,trajectoryEvery=1}={}) {
  if(typeof probability!=='function') {
    const policy=probability;probability=s=>policy.probability(s);
  }
  const episodes=dataset.episodes.map(source=> {
    const env=new CartPole(()=>.5);env.state=[...source.initialState];env.steps=0;
    let maxAbsX=0,maxAbsTheta=0,maxAbsVelocity=0,maxAbsAngularVelocity=0,sumAbsX=0,sumAbsTheta=0,sumAbsVelocity=0,sumAbsAngularVelocity=0;
    let rightActions=0,nearRailSteps=0,nearRailOutwardActions=0,nearAngleSteps=0,firstNearRailStep=null;
    let cart=false,pole=false,capped=false,terminal=false;
    const trajectory=recordTrajectories?[{step:0,state:[...env.state]}]:null;
    for(let t=0;t<dataset.maxSteps;t++) {
      const state=env.state,p=probability(state),action=deterministic?+(p>=.5):+(source.uniforms[t]<p);
      rightActions+=action;
      if(Math.abs(state[0])>1.8){nearRailSteps++;nearRailOutwardActions+=+(state[0]>0?action===1:action===0);if(firstNearRailStep===null)firstNearRailStep=t;}
      if(Math.abs(state[2])>9*Math.PI/180)nearAngleSteps++;
      const result=env.step(action),s=result.state;
      maxAbsX=Math.max(maxAbsX,Math.abs(s[0]));maxAbsTheta=Math.max(maxAbsTheta,Math.abs(s[2]));maxAbsVelocity=Math.max(maxAbsVelocity,Math.abs(s[1]));maxAbsAngularVelocity=Math.max(maxAbsAngularVelocity,Math.abs(s[3]));
      sumAbsX+=Math.abs(s[0]);sumAbsTheta+=Math.abs(s[2]);sumAbsVelocity+=Math.abs(s[1]);sumAbsAngularVelocity+=Math.abs(s[3]);
      if(recordTrajectories&&(env.steps%trajectoryEvery===0||result.done))trajectory.push({step:env.steps,state:[...s],action,probability:p,uniform:source.uniforms[t]});
      if(result.done){cart=Math.abs(s[0])>2.4;pole=Math.abs(s[2])>ANGLE_LIMIT;terminal=result.terminal;capped=result.truncated&&!terminal;break;}
    }
    const cause=cart&&pole?'both':cart?'cart':pole?'pole':'cap';
    return {id:source.id,episodeSeed:source.episodeSeed,initialState:[...source.initialState],reward:env.steps,cause,cartFailure:cart,poleFailure:pole,cartFailureDirection:cart?(env.state[0]>0?'right':'left'):null,poleFailureDirection:pole?(env.state[2]>0?'right':'left'):null,terminal,capped,hit500:env.steps===500,
      finalState:[...env.state],finalAbsAngleDegrees:Math.abs(env.state[2])*180/Math.PI,finalAbsX:Math.abs(env.state[0]),
      maxAbsX,maxAbsAngleDegrees:maxAbsTheta*180/Math.PI,maxAbsVelocity,maxAbsAngularVelocity,
      meanAbsX:sumAbsX/env.steps,meanAbsAngleDegrees:sumAbsTheta/env.steps*180/Math.PI,meanAbsVelocity:sumAbsVelocity/env.steps,meanAbsAngularVelocity:sumAbsAngularVelocity/env.steps,
      rightActionFraction:rightActions/env.steps,nearRailSteps,nearRailOutwardActionFraction:nearRailSteps?nearRailOutwardActions/nearRailSteps:null,nearAngleSteps,firstNearRailStep,...(trajectory?{trajectory}: {})};
  });
  return {episodes,summary:summarizeEpisodes(episodes),deterministic,dataset:{count:dataset.count,seed:dataset.seed,maxSteps:dataset.maxSteps}};
}

export function summarizeEpisodes(episodes) {
  const n=episodes.length,returns=episodes.map(e=>e.reward),hit500=episodes.filter(e=>e.hit500).length,capped=episodes.filter(e=>e.capped).length;
  const causes=Object.fromEntries(['cart','pole','both','cap'].map(cause=>{
    const group=episodes.filter(e=>e.cause===cause),metric=k=>{const values=group.map(e=>e[k]).filter(Number.isFinite);return {n:values.length,mean:meanCI(values).mean,median:quantile(values,.5),p10:quantile(values,.1),p90:quantile(values,.9)};};
    return [cause,{n:group.length,fraction:group.length/n,ci95:wilson(group.length,n),cartFailureDirections:{left:group.filter(e=>e.cartFailureDirection==='left').length,right:group.filter(e=>e.cartFailureDirection==='right').length},duration:meanCI(group.map(e=>e.reward),{bounds:[0,500]}),durationMedian:quantile(group.map(e=>e.reward),.5),finalAbsX:metric('finalAbsX'),finalAbsAngleDegrees:metric('finalAbsAngleDegrees'),meanAbsX:metric('meanAbsX'),meanAbsAngleDegrees:metric('meanAbsAngleDegrees'),maxAbsX:metric('maxAbsX'),maxAbsVelocity:metric('maxAbsVelocity'),meanAbsVelocity:metric('meanAbsVelocity'),firstNearRailStep:metric('firstNearRailStep'),nearRailOutwardActionFraction:metric('nearRailOutwardActionFraction')}];
  }));
  return {returns:meanCI(returns,{bounds:[0,500]}),returnQuantiles:{p10:quantile(returns,.1),median:quantile(returns,.5),p90:quantile(returns,.9)},hit500:{n:hit500,fraction:hit500/n,ci95:wilson(hit500,n)},nonterminalCap:{n:capped,fraction:capped/n,ci95:wilson(capped,n)},causes};
}

export function pairedDifference(episodesA,episodesB) {
  if(episodesA.length!==episodesB.length)throw Error('Different episode count');
  const differences=episodesA.map((a,i)=>{const b=episodesB[i];if(a.id!==b.id||a.episodeSeed!==b.episodeSeed)throw Error('Unmatched episodes');return b.reward-a.reward;});
  return {direction:'B minus A',...meanCI(differences),differences};
}

export function trajectoryExamples(probability,dataset,episodes,{perCause=1}={}) {
  const ids=new Set();
  for(const cause of ['cart','pole','both','cap']) {
    const group=episodes.filter(e=>e.cause===cause).sort((a,b)=>a.reward-b.reward||a.id-b.id);
    for(let i=0;i<perCause;i++)if(group.length)ids.add(group[Math.floor((i+1)*group.length/(perCause+1))].id);
  }
  const subset={...dataset,count:ids.size,episodes:dataset.episodes.filter(e=>ids.has(e.id))};
  return evaluatePolicy(probability,subset,{recordTrajectories:true}).episodes;
}
