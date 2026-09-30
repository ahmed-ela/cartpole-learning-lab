import { CartPole, Network, features, random } from './rl.js';
const $=id=>document.getElementById(id), speeds=[1,5,20,100];
const world=$('world'), chart=$('chart'), ctx=world.getContext('2d'), chartCtx=chart.getContext('2d');
let worker, policy, env=new CartPole(random(731)), viewerRng=random(832), paused=false, speedIndex=2, history=[], stats={steps:0,episodes:0,updates:0,best:0,average:0};
let testing=false, checkpoint=null, deferredProgress=[], dragging=null;
let previous=0, accumulator=0, hold=0, action=1, displayState=env.state.slice(), stateBefore=env.state.slice(), eventUntil=0, worldSize={w:0,h:0}, chartSize={w:0,h:0};
const number=new Intl.NumberFormat('en-US');
function seed(){const values=new Uint32Array(1);crypto.getRandomValues(values);return values[0];}
function startTraining(trainingSeed=42){
  worker?.terminate();history=[];stats={steps:0,episodes:0,updates:0,best:0,average:0};policy=new Network(random(trainingSeed),32,.01);
  env=new CartPole(random(trainingSeed^731));viewerRng=random(trainingSeed^832);displayState=env.state.slice();stateBefore=env.state.slice();hold=0;accumulator=0;eventUntil=0;$('arena-event').classList.remove('visible');
  testing=false;checkpoint=null;deferredProgress=[];dragging=null;paused=false;updatePauseUI();updateStats();drawChart();
  const activeWorker=new Worker(new URL('./trainer.worker.js',import.meta.url),{type:'module'});worker=activeWorker;
  activeWorker.onmessage=({data})=>{
    if(worker!==activeWorker)return;
    if(data.type==='progress'){
      if(testing)deferredProgress.push(data);else applyProgress(data);
    }
  };
  activeWorker.onerror=()=>{paused=true;updatePauseUI();$('status-text').textContent='Training error';$('score-note').textContent='Training could not run. Refresh the page to try again.';};
  activeWorker.postMessage({type:'start',seed:trainingSeed,speed:speeds[speedIndex]});
}
function applyProgress(data){
  stats={steps:data.steps,episodes:data.episodes,updates:data.updates,best:data.best,average:data.average};
  if(data.weights)policy.w.set(data.weights);
  history.push(...data.points);
  if(history.length>2000)history=history.filter((_,i)=>i%2===0 || i===history.length-1);
  updateStats();if(data.points.length)drawChart();
}
function beginCheckpoint(){
  if(testing)return;
  testing=true;checkpoint={update:stats.updates,steps:stats.steps};paused=false;accumulator=0;
  worker.postMessage({type:'pause',paused:true});updatePauseUI();
}
function leaveCheckpoint(){
  testing=false;checkpoint=null;dragging=null;
  for(const progress of deferredProgress)applyProgress(progress);deferredProgress=[];
  setPaused(false);updatePauseUI();
}
function updateStats(){
  $('episodes').textContent=number.format(stats.episodes);$('steps').textContent=number.format(stats.steps);$('updates').textContent=number.format(stats.updates);
  $('average').textContent=stats.episodes?Math.round(stats.average):'—';$('best').textContent=stats.episodes?stats.best:'—';$('score-meter').style.width=`${stats.average/5}%`;
  $('chart-empty').style.display=stats.episodes?'none':'flex';$('chart-episodes').textContent=`${number.format(stats.episodes)} training episode${stats.episodes===1?'':'s'}`;
  $('learning-stage').textContent=stats.average>=475?'Balancing':stats.average>=100?'Improving':'Exploring';
  $('score-note').textContent=stats.episodes?`Average of the last ${Math.min(20,stats.episodes)} training episodes.`:'Every upright step earns a reward of +1.';
}
function updatePauseUI(){
  document.documentElement.dataset.paused=paused;
  $('status-pill').classList.toggle('paused',paused);$('status-pill').classList.toggle('testing',testing);
  $('status-text').textContent=testing?'Testing checkpoint':paused?'Paused':'Learning';
  const resume=paused||testing;
  $('pause').querySelector('span').textContent=resume?'Resume learning':'Pause learning';
  $('pause').querySelector('svg').innerHTML=resume?'<path d="m7 4 9 6-9 6Z"/>':'<path d="M7 5v10M13 5v10"/>';
  $('pause').setAttribute('aria-label',testing?'Resume learning and leave checkpoint':paused?'Resume learning and live view':'Pause learning and live view');
  $('policy-label').textContent=testing?`Checkpoint ${checkpoint.update} · policy frozen`:'Current policy · live test';
  $('checkpoint').disabled=testing;$('checkpoint').textContent=testing?`Checkpoint ${checkpoint.update} locked`:'Test current checkpoint';
  $('interaction-hint').classList.toggle('testing',testing);$('interaction-hint').classList.toggle('dragging',Boolean(dragging));
  $('interaction-text').textContent=dragging?(hold?'Run ended. Release the cart to try again.':'Wiggle the cart. Release it to let the agent respond.'):testing?'Policy frozen. Drag again, or resume learning.':'Drag the cart to test its balance at this checkpoint.';
}
function setPaused(value){paused=Boolean(value);accumulator=0;worker.postMessage({type:'pause',paused:paused||testing});updatePauseUI();}
function setSpeed(index){if(!Number.isInteger(index)||index<0||index>=speeds.length)throw new Error('Choose a supported learning speed.');speedIndex=index;$('speed').value=index;$('speed-value').textContent=`${speeds[index]}×`;$('speed').setAttribute('aria-valuetext',`${speeds[index]} times real time`);document.querySelectorAll('[data-speed]').forEach(button=>{button.classList.toggle('selected',Number(button.dataset.speed)===index);button.setAttribute('aria-pressed',String(Number(button.dataset.speed)===index));});worker?.postMessage({type:'speed',speed:speeds[index]});}
$('pause').addEventListener('click',()=>testing?leaveCheckpoint():setPaused(!paused));$('checkpoint').addEventListener('click',beginCheckpoint);$('reset').addEventListener('click',()=>startTraining(seed()));$('speed').addEventListener('input',event=>setSpeed(Number(event.target.value)));document.querySelectorAll('[data-speed]').forEach(button=>button.addEventListener('click',()=>setSpeed(Number(button.dataset.speed))));
document.addEventListener('keydown',event=>{if(event.code==='Space' && !['INPUT','BUTTON','SUMMARY','A','TEXTAREA','SELECT'].includes(event.target.tagName)){event.preventDefault();testing?leaveCheckpoint():setPaused(!paused);}});
function pointerPosition(event){const rect=world.getBoundingClientRect();return{x:(event.clientX-rect.left)*worldSize.w/rect.width,y:(event.clientY-rect.top)*worldSize.h/rect.height};}
function cartHit(point){const cartX=worldSize.w/2+displayState[0]*(worldSize.w-72)/5.8,trackY=worldSize.h*.73;return Math.abs(point.x-cartX)<=54 && point.y>=trackY-54 && point.y<=trackY+20;}
world.addEventListener('pointerdown',event=>{
  if(event.button!==0 || dragging)return;
  const point=pointerPosition(event);if(!cartHit(point))return;
  event.preventDefault();beginCheckpoint();world.focus({preventScroll:true});world.setPointerCapture(event.pointerId);
  const scale=(worldSize.w-72)/5.8;
  dragging={pointerId:event.pointerId,offset:point.x-(worldSize.w/2+env.state[0]*scale),target:env.state[0]};
  world.style.cursor='grabbing';updatePauseUI();
});
world.addEventListener('pointermove',event=>{
  const point=pointerPosition(event);
  if(dragging && event.pointerId===dragging.pointerId){
    const scale=(worldSize.w-72)/5.8;dragging.target=Math.max(-2.35,Math.min(2.35,(point.x-dragging.offset-worldSize.w/2)/scale));
  }else world.style.cursor=cartHit(point)?'grab':'default';
});
function releaseDrag(event){if(!dragging||event.pointerId!==dragging.pointerId)return;dragging=null;world.style.cursor='grab';updatePauseUI();}
world.addEventListener('pointerup',releaseDrag);world.addEventListener('pointercancel',releaseDrag);world.addEventListener('lostpointercapture',releaseDrag);
world.addEventListener('keydown',event=>{
  if(!['ArrowLeft','ArrowRight'].includes(event.key))return;
  event.preventDefault();beginCheckpoint();env.impulse(event.key==='ArrowRight'?0.25:-0.25);displayState=env.state.slice();stateBefore=env.state.slice();
});
function fit(canvas,context){const rect=canvas.getBoundingClientRect(),dpr=Math.min(devicePixelRatio||1,2);canvas.width=Math.round(rect.width*dpr);canvas.height=Math.round(rect.height*dpr);context.setTransform(dpr,0,0,dpr,0,0);return{w:rect.width,h:rect.height};}
new ResizeObserver(()=>{worldSize=fit(world,ctx);chartSize=fit(chart,chartCtx);drawChart();drawWorld();}).observe(document.querySelector('.visual-column'));
function roundRect(context,x,y,w,h,r){context.beginPath();context.roundRect(x,y,w,h,r);}
function drawWorld(){
  const {w,h}=worldSize;if(!w||!h)return;ctx.clearRect(0,0,w,h);
  ctx.strokeStyle='#e5ebf5';ctx.lineWidth=.7;
  for(let x=w/2%32;x<w;x+=32){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke();}
  for(let y=16;y<h;y+=32){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke();}
  const trackY=h*.73,scale=(w-72)/5.8,cartX=w/2+displayState[0]*scale,pivotY=trackY-26,length=Math.min(132,h*.4),theta=displayState[2];
  // The limits and vertical reference are tied to the physics, not decorative motion.
  ctx.fillStyle='#dce3ed';roundRect(ctx,24,trackY+5,w-48,3,1.5);ctx.fill();
  for(const offset of[-2.4,-1.2,0,1.2,2.4]){const x=w/2+offset*scale;ctx.strokeStyle=offset===0?'#b3c1d7':'#c9d3e3';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x,trackY+14);ctx.lineTo(x,trackY+19);ctx.stroke();ctx.font='12px -apple-system,BlinkMacSystemFont,sans-serif';ctx.textAlign='center';ctx.fillStyle='#9ba9bc';ctx.fillText(offset===0?'0':`${offset>0?'+':''}${offset.toFixed(1)} m`,x,trackY+36);}
  const limit=.20943951;ctx.setLineDash([4,5]);ctx.strokeStyle='#c0cddd';ctx.lineWidth=1;
  for(const sign of[-1,1]){ctx.beginPath();ctx.moveTo(cartX,pivotY);ctx.lineTo(cartX+Math.sin(limit*sign)*length,pivotY-Math.cos(limit)*length);ctx.stroke();}
  ctx.setLineDash([]);
  ctx.strokeStyle='#d8e2f0';ctx.lineWidth=1;ctx.beginPath();ctx.arc(cartX,pivotY,length,-Math.PI/2-limit,-Math.PI/2+limit);ctx.stroke();
  ctx.save();ctx.translate(cartX,pivotY);ctx.rotate(theta);ctx.shadowColor='#305cff21';ctx.shadowBlur=10;ctx.shadowOffsetY=4;ctx.strokeStyle='#305cff';ctx.lineWidth=8;ctx.lineCap='round';ctx.beginPath();ctx.moveTo(0,0);ctx.lineTo(0,-length);ctx.stroke();ctx.shadowBlur=0;ctx.shadowOffsetY=0;ctx.fillStyle='#99b2ff';ctx.beginPath();ctx.arc(0,-length,4,0,Math.PI*2);ctx.fill();ctx.restore();
  ctx.save();ctx.shadowColor='#305cff25';ctx.shadowBlur=15;ctx.shadowOffsetY=6;const gradient=ctx.createLinearGradient(cartX-38,pivotY,cartX+38,pivotY+22);gradient.addColorStop(0,'#426dff');gradient.addColorStop(1,'#2852ed');ctx.fillStyle=gradient;roundRect(ctx,cartX-39,pivotY-2,78,26,7);ctx.fill();ctx.restore();
  ctx.fillStyle='#193bb4';for(const offset of[-25,25]){ctx.beginPath();ctx.arc(cartX+offset,trackY-1,7,0,Math.PI*2);ctx.fill();ctx.fillStyle='#9aaff2';ctx.beginPath();ctx.arc(cartX+offset,trackY-1,2.4,0,Math.PI*2);ctx.fill();ctx.fillStyle='#193bb4';}
  ctx.fillStyle='white';ctx.beginPath();ctx.arc(cartX,pivotY+2,5,0,Math.PI*2);ctx.fill();ctx.fillStyle='#a2b8ff';ctx.beginPath();ctx.arc(cartX,pivotY+2,2,0,Math.PI*2);ctx.fill();
  if(!paused&&!hold&&!dragging){const dir=action?1:-1,x=cartX+dir*58,y=pivotY+11;ctx.strokeStyle='#829bda';ctx.lineWidth=1.8;ctx.lineCap='round';ctx.beginPath();ctx.moveTo(x-dir*7,y);ctx.lineTo(x+dir*7,y);ctx.moveTo(x+dir*3,y-4);ctx.lineTo(x+dir*7,y);ctx.lineTo(x+dir*3,y+4);ctx.stroke();}
}
function drawChart(){
  const{w,h}=chartSize;if(!w||!h)return;chartCtx.clearRect(0,0,w,h);const l=34,r=10,t=12,b=24,cw=w-l-r,ch=h-t-b;
  chartCtx.font='12px -apple-system,BlinkMacSystemFont,sans-serif';chartCtx.fillStyle='#9ca8b8';chartCtx.lineWidth=.8;
  for(const value of[0,250,500]){const y=t+ch*(1-value/500);chartCtx.strokeStyle='#e9edf3';chartCtx.beginPath();chartCtx.moveTo(l,y);chartCtx.lineTo(w-r,y);chartCtx.stroke();chartCtx.textAlign='right';chartCtx.fillText(String(value),l-9,y+3);}
  const maxEpisode=Math.max(20,stats.episodes);const x=episode=>l+(episode-1)/Math.max(1,maxEpisode-1)*cw,y=reward=>t+ch*(1-reward/500);
  chartCtx.textAlign='center';for(const ep of[1,Math.ceil(maxEpisode/2),maxEpisode])chartCtx.fillText(number.format(ep),x(ep),h-5);
  if(history.length<1)return;
  chartCtx.beginPath();chartCtx.moveTo(x(history[0].episode),t+ch);for(const point of history)chartCtx.lineTo(x(point.episode),y(point.average));chartCtx.lineTo(x(history.at(-1).episode),t+ch);chartCtx.closePath();const fill=chartCtx.createLinearGradient(0,t,0,t+ch);fill.addColorStop(0,'#305cff1f');fill.addColorStop(1,'#305cff02');chartCtx.fillStyle=fill;chartCtx.fill();
  chartCtx.beginPath();history.forEach((p,i)=>i?chartCtx.lineTo(x(p.episode),y(p.reward)):chartCtx.moveTo(x(p.episode),y(p.reward)));chartCtx.strokeStyle='#c3d0f6';chartCtx.lineWidth=1;chartCtx.stroke();
  chartCtx.beginPath();history.forEach((p,i)=>i?chartCtx.lineTo(x(p.episode),y(p.average)):chartCtx.moveTo(x(p.episode),y(p.average)));chartCtx.strokeStyle='#305cff';chartCtx.lineWidth=2;chartCtx.lineJoin='round';chartCtx.stroke();const last=history.at(-1);chartCtx.fillStyle='#305cff';chartCtx.beginPath();chartCtx.arc(x(last.episode),y(last.average),3,0,Math.PI*2);chartCtx.fill();
}
function showEvent(text,time){$('arena-event').textContent=text;$('arena-event').classList.add('visible');eventUntil=time+650;}
function frame(time){
  const delta=previous?Math.min(time-previous,100):0;previous=time;
  if(!paused){
    if(hold>0){if(!dragging)hold-=delta;if(hold<=0){hold=0;env.reset();stateBefore=env.state.slice();displayState=env.state.slice();accumulator=0;$('arena-event').classList.remove('visible');eventUntil=0;}}
    else{
      accumulator+=delta;
      while(accumulator>=20){
        accumulator-=20;stateBefore=env.state.slice();const p=1/(1+Math.exp(-policy.forward(features(env.state)).y));action=+(viewerRng()<p);const result=dragging?env.dragStep(dragging.target):env.step(action);
        if(result.done){hold=650;accumulator=0;displayState=env.state.slice();showEvent(result.truncated?'500 steps. Perfect balance.':dragging?'Pole down. Release the cart.':'Pole down. Trying again.',time);if(dragging)updatePauseUI();break;}
      }
      if(!hold){const alpha=accumulator/20;displayState=env.state.map((n,i)=>stateBefore[i]+(n-stateBefore[i])*alpha);}
    }
  }
  $('run-score').textContent=env.steps;$('angle').innerHTML=`${(displayState[2]*180/Math.PI).toFixed(1)}<span>°</span>`;$('position').innerHTML=`${displayState[0].toFixed(2)}<span> m</span>`;$('action').textContent=paused?'Paused':hold?'Resetting':dragging?'Held by you':action?'Push right':'Push left';
  if(time>eventUntil)$('arena-event').classList.remove('visible');drawWorld();requestAnimationFrame(frame);
}
startTraining();setSpeed(speedIndex);requestAnimationFrame(frame);
// Expose the same actions only on browsers with WebMCP support.
const modelContext=document.modelContext;
if(modelContext?.registerTool){
  const lifecycle=new AbortController();window.addEventListener('pagehide',()=>lifecycle.abort(),{once:true});
  const tools=[
    {name:'get_cartpole_training',title:'Read CartPole training',description:'Read the live learner statistics and current training speed.',inputSchema:{type:'object',properties:{},additionalProperties:false},annotations:{readOnlyHint:true,untrustedContentHint:false},execute:()=>({...stats,paused,testing,checkpoint,speed:speeds[speedIndex]})},
    {name:'configure_cartpole_training',title:'Configure CartPole training',description:'Set the real training speed and pause or resume training and its live view.',inputSchema:{type:'object',properties:{speed:{type:'number',enum:speeds},paused:{type:'boolean'}},additionalProperties:false},annotations:{readOnlyHint:false,untrustedContentHint:false},execute:input=>{if(!input||typeof input!=='object'||Array.isArray(input)||Object.keys(input).some(key=>!['speed','paused'].includes(key))||(input.speed!==undefined&&!speeds.includes(input.speed))||(input.paused!==undefined&&typeof input.paused!=='boolean'))throw new Error('Invalid training configuration.');if(input.speed!==undefined)setSpeed(speeds.indexOf(input.speed));if(input.paused!==undefined){if(!input.paused&&testing)leaveCheckpoint();else setPaused(input.paused);}return{paused,testing,speed:speeds[speedIndex]};}}
  ];
  for(const tool of tools){try{Promise.resolve(modelContext.registerTool(tool,{signal:lifecycle.signal})).catch(()=>{});}catch{}}
}
