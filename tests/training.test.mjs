import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { loadModel, createTestClock, APP_URL } from './model-loader.mjs';
const plain = (x) => JSON.parse(JSON.stringify(x));
const params = (n) => [...n.weights.flat(2), ...n.biases.flat()];
const weights = (n) => JSON.stringify([n.weights, n.biases]);
const setup = () => { const clock = createTestClock(), api = loadModel(undefined, { clock }); api.randomizeArchWeights(); api.forward(); return { api, clock }; };

test('fruit samples are exact normalized synthetic train/test points', () => {
  const a = loadModel();
  assert.ok(a.FRUIT_SAMPLES);
  const train = a.FRUIT_SAMPLES.filter(s => s.split === 'train'), held = a.FRUIT_SAMPLES.filter(s => s.split === 'test');
  assert.deepEqual(plain(train.map(s => s.inputs)), [[.7,.2],[.8,.2],[.9,.3],[.7,.4],[.8,.4],[.9,.1],[.2,.8],[.3,.7],[.1,.9],[.4,.8],[.2,.6],[.3,.9]]);
  assert.deepEqual(plain(held.map(s => s.inputs)), [[.65,.3],[.85,.35],[.25,.75],[.35,.85]]);
  assert.deepEqual(plain(train.map(s=>s.target)), [1,1,1,1,1,1,0,0,0,0,0,0]);
});

test('mean batch gradients match finite differences away from ReLU crossings and are pure', () => {
  const a = loadModel(); assert.equal(typeof a.computeBatchGradients, 'function');
  const n = { hidden:[2], inputs:[.2,.7], weights:[[[.3,.4],[.2,.1]],[[.7],[-.3]]], biases:[[.5,.4],[.1]] };
  const samples = [{ inputs:[.2,.7], target:1, split:'train' },{ inputs:[.9,.1], target:0, split:'train' }];
  const before = JSON.stringify([n,samples]), g = a.computeBatchGradients(n,samples), twice = a.computeBatchGradients(n,[...samples,...samples]);
  assert.equal(JSON.stringify([n,samples]),before); assert.deepEqual(plain(g),plain(twice));
  const eps=1e-5;
  for (const [kind,grad] of [['weights',g.weightGradients],['biases',g.biasGradients]]) {
    n[kind].forEach((layer,l) => layer.forEach((row,i) => {
      const entries = Array.isArray(row) ? row.map((_,o)=>o) : [null];
      for(const o of entries) {
        const values=o===null ? n[kind][l] : n[kind][l][i], key=o===null ? i : o, old=values[key];
        values[key]=old+eps; const hi=a.computeBatchGradients(n,samples).loss;
        values[key]=old-eps; const lo=a.computeBatchGradients(n,samples).loss; values[key]=old;
        const actual=o===null ? grad[l][i] : grad[l][i][o]; assert.ok(Math.abs(actual-(hi-lo)/(2*eps))<1e-4);
      }
    }));
  }
});

test('seed 42 converges, finite at all offered rates, updates float32 and recomputes loss', () => {
  const a=loadModel(); assert.equal(typeof a.createFruitNetwork,'function');
  assert.deepEqual(plain(a.createFruitNetwork()),plain(a.createFruitNetwork()));
  for(const rate of [.01,.1,.5]) {
    let n=a.createFruitNetwork(), initial=a.fruitMetrics(n).loss; const before=weights(n);
    let result=a.trainEpoch(n,a.FRUIT_SAMPLES,rate); assert.equal(weights(n),before);
    assert.ok(params(result.network).every(v=>Number.isFinite(v)&&Math.fround(v)===v));
    assert.equal(result.loss,a.fruitMetrics(result.network).loss);
    n=result.network;
    for(let i=1;i<1500;i++) n=a.trainEpoch(n,a.FRUIT_SAMPLES,rate).network;
    assert.ok(a.fruitMetrics(n).loss<initial);
    if(rate===.1) assert.ok(a.fruitMetrics(n).accuracy>=.95);
  }
});

test('BCE is stable for extreme logits; held-out examples never affect gradients', () => {
  const a=loadModel(); assert.equal(typeof a.trainEpoch,'function');
  const n=a.createFruitNetwork(), train=a.FRUIT_SAMPLES.filter(s=>s.split==='train');
  assert.deepEqual(plain(a.computeBatchGradients(n,a.FRUIT_SAMPLES)),plain(a.computeBatchGradients(n,train)));
  n.biases.at(-1)[0]=1000; assert.ok(Number.isFinite(a.computeBatchGradients(n,train).loss));
  assert.throws(()=>a.trainEpoch(n,train,Infinity));
  assert.throws(()=>a.trainEpoch(n,train,Number.MAX_VALUE));
});

test('pause, reset, repeated start and stale callbacks preserve deterministic training', () => {
  const {api:a,clock}=setup(); assert.equal(typeof a.enterFruitLesson,'function'); a.enterFruitLesson();
  const initial=weights(a.state.arch); a.startTraining(); const stale=clock.nextCallback();
  a.startTraining(); assert.equal(clock.pendingCount,1); clock.tick(); assert.ok(a.state.lesson.epoch>0);
  a.pauseTraining(); const trained=weights(a.state.arch), epoch=a.state.lesson.epoch; stale();
  assert.equal(weights(a.state.arch),trained); assert.equal(a.state.lesson.epoch,epoch); assert.equal(clock.pendingCount,0);
  a.stepTraining(); assert.equal(a.state.lesson.epoch,epoch+1);
  a.setLearningRate(.5); a.startTraining(); const resetStale=clock.nextCallback(); a.resetTraining(); resetStale();
  assert.equal(a.state.lesson.epoch,0); assert.equal(a.state.lesson.history.length,0); assert.equal(a.state.lesson.learningRate,.5);
  assert.equal(weights(a.state.arch),initial); assert.equal(clock.pendingCount,0);
  a.stepTraining(); const first=weights(a.state.arch); a.resetTraining(); a.stepTraining(); assert.equal(weights(a.state.arch),first);
});

test('explorer snapshot and RNG stream restore exactly; inference does not change weight bytes', () => {
  const {api:a}=setup(), {api:control}=setup(); assert.equal(typeof a.enterFruitLesson,'function');
  a.selectParameter({layer:1,kind:'bias',input:null,output:1});
  const saved=JSON.stringify([a.state.arch,a.state.selection]); a.enterFruitLesson(); a.stepTraining();
  const trained=weights(a.state.arch); a.selectFruitSample('test-strawberry-1'); a.setInputs([.6,.6]);
  assert.equal(a.state.lesson.selectedSampleId,null); assert.equal(weights(a.state.arch),trained);
  const p=a.evaluateNetwork(a.state.arch,a.state.arch.inputs).output; assert.equal(p+(1-p),1);
  const file=a.buildSafetensors(), data=new DataView(file.bytes.buffer), base=8+Number(data.getBigUint64(0,true));
  for(const tensor of file.tensors) tensor.values.forEach((v,i)=>assert.equal(data.getFloat32(base+tensor.offsets[0]+i*4,true),v));
  a.exitFruitLesson(); assert.equal(JSON.stringify([a.state.arch,a.state.selection]),saved);
  a.randomizeArchWeights(); control.randomizeArchWeights(); assert.equal(weights(a.state.arch),weights(control.state.arch));
});

test('training yields for at least 100 ms and renders once per bounded slice, not per epoch', () => {
  const clock=createTestClock(); let renders=0;
  const a=loadModel(undefined,{clock,onRender:()=>renders++});
  a.randomizeArchWeights(); a.forward(); a.enterFruitLesson(); a.startTraining();
  const before=renders; clock.tick();
  assert.equal(a.state.lesson.epoch,20); assert.equal(renders,before+1);
  clock.tick(); assert.equal(a.state.lesson.epoch,40); assert.equal(renders,before+2);
  assert.ok(clock.delays.every(delay=>delay>=100)); a.pauseTraining();
});

test('fruit PNG data URLs preserve original bytes and every fruit translation has both languages', () => {
  const html=readFileSync(APP_URL,'utf8'), a=loadModel();
  for(const name of ['strawberry','blueberry']) {
    const match=html.match(new RegExp(`${name}: 'data:image/png;base64,([^']+)'`)); assert.ok(match);
    assert.deepEqual(Buffer.from(match[1],'base64'),readFileSync(new URL(`../assets/training/${name}.png`,import.meta.url)));
  }
  const pl=Object.keys(a.T.pl).filter(k=>k.startsWith('fruit')).sort(), en=Object.keys(a.T.en).filter(k=>k.startsWith('fruit')).sort();
  assert.deepEqual(pl,en); assert.ok(pl.length>=40);
});

test('deep topology gradients use pre-update weights and updates stay finite', () => {
  const a=loadModel(), n={hidden:[2,2],inputs:[.3,.7],weights:[[[.3,.4],[.2,.1]],[[.4,.3],[.2,.1]],[[.7],[-.3]]],biases:[[.5,.4],[.2,.3],[.1]]};
  const samples=[{inputs:[.3,.7],target:1,split:'train'},{inputs:[.8,.2],target:0,split:'train'}];
  const g=a.computeBatchGradients(n,samples), eps=1e-5, before=weights(n);
  n.weights.forEach((layer,l)=>layer.forEach((row,i)=>row.forEach((v,o)=>{
    row[o]=v+eps; const hi=a.computeBatchGradients(n,samples).loss;
    row[o]=v-eps; const lo=a.computeBatchGradients(n,samples).loss; row[o]=v;
    assert.ok(Math.abs(g.weightGradients[l][i][o]-(hi-lo)/(2*eps))<1e-4);
  })));
  const result=a.trainEpoch(n,samples,.1); assert.equal(weights(n),before);
  assert.equal(result.network.weights[0][0][0],Math.fround(n.weights[0][0][0]-.1*g.weightGradients[0][0][0]));
  assert.ok(params(result.network).every(Number.isFinite));
});

test('input, language and topology respect training handoff and cap', () => {
  const {api:a,clock}=setup(); assert.equal(typeof a.enterFruitLesson,'function'); a.enterFruitLesson(); a.startTraining();
  assert.throws(()=>a.setArchitecture([4])); assert.throws(()=>a.randomizeArchWeights());
  a.setInputs([.3,.2]); assert.equal(a.state.lesson.phase,'paused'); assert.equal(clock.pendingCount,0);
  a.startTraining(); a.setLanguage('en'); assert.equal(a.state.lesson.phase,'paused');
  a.setArchitecture([4]); assert.equal(a.state.lesson.epoch,0); const initial=weights(a.state.arch);
  a.stepTraining(); a.resetTraining(); assert.equal(weights(a.state.arch),initial);
  a.state.lesson.epoch=4999; a.startTraining(); clock.tick(); assert.equal(a.state.lesson.epoch,5000); assert.equal(clock.pendingCount,0);
});

test('bad candidate keeps last finite network and pauses with error', () => {
  const {api:a}=setup(); assert.equal(typeof a.enterFruitLesson,'function'); a.enterFruitLesson();
  const before=weights(a.state.arch); a.state.lesson.learningRate=Number.MAX_VALUE; a.stepTraining();
  assert.equal(weights(a.state.arch),before); assert.equal(a.state.lesson.phase,'error'); assert.ok(a.state.lesson.error);
});

test('reset or pause at the end of an active slice cannot schedule another callback', () => {
  for(const action of ['resetTraining','pauseTraining']) {
    const clock=createTestClock(); let api, armed=false;
    api=loadModel(undefined,{clock,onRender:()=>{if(armed&&api.state.lesson.epoch===20){armed=false;api[action]();}}});
    api.randomizeArchWeights(); api.forward(); api.enterFruitLesson(); const initial=weights(api.state.arch);
    api.startTraining(); armed=true; clock.tick(); assert.equal(clock.pendingCount,0);
    if(action==='resetTraining') { assert.equal(api.state.lesson.epoch,0); assert.equal(weights(api.state.arch),initial); }
    else { assert.equal(api.state.lesson.epoch,20); assert.equal(api.state.lesson.phase,'paused'); }
  }
});
