export function random(seed) {
  return () => { let t = seed += 0x6d2b79f5; t = Math.imul(t ^ t >>> 15, t | 1); t ^= t + Math.imul(t ^ t >>> 7, t | 61); return ((t ^ t >>> 14) >>> 0) / 4294967296; };
}

export class CartPole {
  constructor(rng = Math.random) { this.rng = rng; this.reset(); }
  reset() { this.state = Array.from({length:4}, () => (this.rng() - .5) * .1); this.steps = 0; return this.state; }
  // A horizontal cart impulse does not apply an angular impulse to the rod.
  // For Gymnasium's uniform pole, delta omega = -3 cos(theta) delta v / (4l).
  impulse(deltaVelocity) {
    this.state[1] += deltaVelocity;
    this.state[3] -= 1.5 * Math.cos(this.state[2]) * deltaVelocity;
  }
  // Mouse dragging prescribes a bounded cart velocity. Couple velocity changes
  // to the pole, then integrate its gravity-driven motion; never set its angle.
  dragStep(targetX) {
    const [x, velocity, theta, omega] = this.state;
    const desiredVelocity = Math.max(-1.2, Math.min(1.2, (targetX-x)/0.02));
    const nextVelocity = velocity + Math.max(-0.15, Math.min(0.15, desiredVelocity-velocity));
    const nextOmega = omega - 1.5*Math.cos(theta)*(nextVelocity-velocity);
    this.state = [x+0.02*nextVelocity, nextVelocity, theta+0.02*nextOmega, nextOmega+0.02*14.7*Math.sin(theta)];
    this.steps++;
    const terminal = Math.abs(this.state[0]) > 2.4 || Math.abs(this.state[2]) > 12*Math.PI/180;
    return {state:this.state, reward:1, terminal, truncated:this.steps>=500, done:terminal||this.steps>=500};
  }
  step(action) {
    const [x, xd, th, thd] = this.state;
    const c = Math.cos(th), s = Math.sin(th), temp = ((action ? 10 : -10) + .05 * thd * thd * s) / 1.1;
    const thacc = (9.8*s-c*temp)/(.5*(4/3-.1*c*c/1.1)), xacc = temp-.05*thacc*c/1.1;
    this.state = [x+.02*xd, xd+.02*xacc, th+.02*thd, thd+.02*thacc];
    this.steps++;
    const terminal = Math.abs(this.state[0]) > 2.4 || Math.abs(this.state[2]) > 12*Math.PI/180;
    return {state:this.state, reward:1, terminal, truncated:this.steps>=500, done:terminal||this.steps>=500};
  }
}

export function features(s) { return [s[0]/2.4, s[1]/2, s[2]*4.7746483, s[3]/2]; }
export class Network {
  constructor(rng, hidden=32, outputScale=.1) {
    this.hidden=hidden;
    this.w=new Float64Array(hidden*6+1); this.g=new Float64Array(this.w.length); this.m=new Float64Array(this.w.length); this.v=new Float64Array(this.w.length); this.t=0;
    const bound=Math.sqrt(6/(hidden+4));
    for(let j=0;j<hidden;j++) { for(let i=0;i<4;i++) this.w[j*5+i]=(rng()*2-1)*bound; this.w[hidden*5+j]=(rng()*2-1)*outputScale; }
  }
  forward(x) {
    const h=new Float64Array(this.hidden); let y=this.w[this.hidden*6];
    for(let j=0;j<this.hidden;j++) { let z=this.w[j*5+4]; for(let i=0;i<4;i++) z+=this.w[j*5+i]*x[i]; h[j]=Math.tanh(z); y+=h[j]*this.w[this.hidden*5+j]; }
    return {x,h,y};
  }
  backward(cache,dy) {
    this.g[this.hidden*6]+=dy;
    for(let j=0;j<this.hidden;j++) { this.g[this.hidden*5+j]+=dy*cache.h[j]; const dz=dy*this.w[this.hidden*5+j]*(1-cache.h[j]*cache.h[j]); for(let i=0;i<4;i++) this.g[j*5+i]+=dz*cache.x[i]; this.g[j*5+4]+=dz; }
  }
  update(lr, batch) {
    let norm=0; for(let i=0;i<this.w.length;i++){this.g[i]/=batch; norm+=this.g[i]*this.g[i];}
    const scale=Math.min(1,.5/(Math.sqrt(norm)+1e-8)); this.t++; const b1=1-Math.pow(.9,this.t),b2=1-Math.pow(.999,this.t);
    for(let i=0;i<this.w.length;i++){const g=this.g[i]*scale;this.m[i]=.9*this.m[i]+.1*g; this.v[i]=.999*this.v[i]+.001*g*g;this.w[i]-=lr*(this.m[i]/b1)/(Math.sqrt(this.v[i]/b2)+1e-8);this.g[i]=0;}
  }
}
export class PPO {
  constructor(seed=42,options={}) {
    this.rng=random(seed); this.actor=new Network(this.rng,options.hidden??32,.01); this.critic=new Network(this.rng,options.hidden??32,.1); this.env=new CartPole(this.rng);
    this.options={rollout:1024,epochs:8,batch:64,actorLR:.001,criticLR:.001,gamma:.99,lambda:.95,clip:.2,entropy:.005,...options};
    this.totalSteps=0;this.episodes=[];this.updates=0;this.buffer=[];
  }
  probability(s) {return 1/(1+Math.exp(-this.actor.forward(features(s)).y));}
  action(s,deterministic=false) { const p=this.probability(s);return deterministic ? +(p>=.5) : +(this.rng()<p); }
  stepExperience() {
      const x=features(this.env.state),p=1/(1+Math.exp(-this.actor.forward(x).y)),a=+(this.rng()<p),v=this.critic.forward(x).y;
      const result=this.env.step(a),nv=result.terminal ? 0 : this.critic.forward(features(result.state)).y;
      this.buffer.push({x,a,logp:Math.log(a?p:1-p),v,nv,done:result.done,r:.01});
      const episodeScore=result.done?this.env.steps:null;
      if(result.done){this.episodes.push(this.env.steps);this.env.reset();}
      this.totalSteps++;
      return {...result,action:a,probability:p,episodeScore,rolloutReady:this.buffer.length>=this.options.rollout};
  }
  finishRollout() {
    const data=this.buffer;this.buffer=[];const o=this.options;
    let gae=0;
    for(let t=data.length-1;t>=0;t--){const d=data[t];gae=d.r+o.gamma*d.nv-d.v+o.gamma*o.lambda*(d.done?0:gae);d.adv=gae;d.ret=gae+d.v;}
    const mean=data.reduce((sum,d)=>sum+d.adv,0)/data.length,std=Math.sqrt(data.reduce((sum,d)=>sum+(d.adv-mean)**2,0)/data.length)+1e-8;
    for(const d of data)d.adv=(d.adv-mean)/std;
    return data;
  }
  collect() {
    while(this.buffer.length<this.options.rollout)this.stepExperience();
    return this.finishRollout();
  }
  learn(data=this.collect()) {
    const o=this.options, indices=Array.from({length:data.length},(_,i)=>i);
    for(let epoch=0;epoch<o.epochs;epoch++) {
      for(let i=indices.length-1;i>0;i--){let j=Math.floor(this.rng()*(i+1));[indices[i],indices[j]]=[indices[j],indices[i]];}
      for(let start=0;start<data.length;start+=o.batch) {
        const end=Math.min(start+o.batch,data.length);
        for(let k=start;k<end;k++) {
          const d=data[indices[k]],ac=this.actor.forward(d.x),p=Math.max(1e-6,Math.min(1-1e-6,1/(1+Math.exp(-ac.y)))),logp=Math.log(d.a?p:1-p),ratio=Math.exp(logp-d.logp);
          const clipped=(d.adv>=0&&ratio>1+o.clip)||(d.adv<0&&ratio<1-o.clip);
          let dy=clipped ? 0 : -d.adv*ratio*(d.a-p);
          dy+=o.entropy*p*(1-p)*Math.log(p/(1-p));
          this.actor.backward(ac,dy);
          const vc=this.critic.forward(d.x); this.critic.backward(vc,vc.y-d.ret);
        }
        this.actor.update(o.actorLR,end-start); this.critic.update(o.criticLR,end-start);
      }
    }
    this.updates++;
  }
  evaluate(seed=123456,count=20,deterministic=true) {
    const rng=random(seed),env=new CartPole(rng),scores=[];
    for(let ep=0;ep<count;ep++){env.reset();for(let t=0;t<500;t++){const p=this.probability(env.state),a=deterministic ? +(p>=.5) : +(rng()<p);if(env.step(a).done)break;}scores.push(env.steps);}
    return {mean:scores.reduce((a,b)=>a+b,0)/count,min:Math.min(...scores),scores};
  }
}
