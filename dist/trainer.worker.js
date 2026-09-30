import { PPO } from './rl.js';
let learner, speed=20, paused=false, credit=0, previous=performance.now(), lastReport=0, totalEpisodes=0, best=0, recent=[], pending=[], timer;
function report(includePolicy=false) {
  const mean=recent.length ? recent.reduce((sum,n)=>sum+n,0)/recent.length : 0;
  postMessage({type:'progress',steps:learner.totalSteps,episodes:totalEpisodes,updates:learner.updates,best,average:mean,points:pending.splice(0),...(includePolicy?{weights:Array.from(learner.actor.w),hidden:learner.actor.hidden}:{})});
  lastReport=performance.now();
}
function tick() {
  const now=performance.now(), dt=Math.min(100,now-previous); previous=now;
  if(paused || !learner)return;
  credit+=dt/1000*50*speed;
  let policyChanged=false;
  const count=Math.min(1000,Math.floor(credit));credit-=count;
  for(let i=0;i<count;i++) {
    const result=learner.stepExperience();
    if(result.episodeScore!==null) {
      totalEpisodes++;best=Math.max(best,result.episodeScore);recent.push(result.episodeScore);if(recent.length>20)recent.shift();
      pending.push({episode:totalEpisodes,reward:result.episodeScore,average:recent.reduce((a,b)=>a+b,0)/recent.length});
      if(learner.episodes.length>1000)learner.episodes.splice(0,500);
    }
    if(result.rolloutReady){learner.learn(learner.finishRollout());policyChanged=true;}
  }
  if(policyChanged || now-lastReport>=200)report(policyChanged);
}
onmessage=({data})=>{
  if(data.type==='start') {
    clearInterval(timer);learner=new PPO(data.seed);speed=data.speed;paused=false;credit=0;totalEpisodes=0;best=0;recent=[];pending=[];previous=performance.now();report(true);timer=setInterval(tick,20);
  }else if(data.type==='pause'){paused=Boolean(data.paused);credit=0;previous=performance.now();report();postMessage({type:'pause-ack',paused});}
  else if(data.type==='speed' && [1,5,20,100].includes(data.speed)){speed=data.speed;credit=0;previous=performance.now();}
};
