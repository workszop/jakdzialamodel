import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { APP_URL, createTestClock, extractNumericalScript, loadModel } from './model-loader.mjs';

const SEED_TENSORS = [
  { name: 'fc1.weight', shape: [4, 2], values: [0.2022075057029724, -0.6503722071647644, -0.10341887921094894, 0.05318508297204971, 0.7049316167831421, -0.453544020652771, 0.3394680917263031, 0.24948930740356445] },
  { name: 'fc1.bias', shape: [4], values: [-0.18738481402397156, -0.09321760386228561, -0.17528744041919708, -0.12572439014911652] },
  { name: 'fc2.weight', shape: [3, 4], values: [0.7309492826461792, 0.7641176581382751, -0.605492353439331, 0.2212418019771576, -0.05536589026451111, 0.4914751350879669, 0.0014589754864573479, -0.9923141002655029, -0.5001525282859802, -0.3859969675540924, 0.37322404980659485, -0.05843615159392357] },
  { name: 'fc2.bias', shape: [3], values: [0.11341891437768936, 0.012134242802858353, -0.18915055692195892] },
  { name: 'fc3.weight', shape: [1, 3], values: [0.6746748685836792, -0.8975814580917358, 0.18464797735214233] },
  { name: 'fc3.bias', shape: [1], values: [-0.13079790771007538] },
];

const plain = (value) => JSON.parse(JSON.stringify(value));
test('generic parameter announcements cannot outlive a language change', () => {
  const callbacks = [], api = loadModel(undefined, { requestFrame: cb => callbacks.push(cb) });
  api.randomizeArchWeights(); api.forward();
  api.selectParameter({ layer:0, kind:'weight', input:0, output:0 });
  api.setLanguage('en'); callbacks.forEach(cb => cb());
  assert.equal(api.liveAnnouncement, '');
});
const closeTo = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-12, `${actual} differs from ${expected}`);
function deepFreeze(value) {
  if (value && typeof value === 'object') {
    Object.values(value).forEach(deepFreeze);
    Object.freeze(value);
  }
  return value;
}

test('loader rejects a missing or moved Render boundary rather than skipping helpers', () => {
  const html = readFileSync(APP_URL, 'utf8');
  assert.throws(() => extractNumericalScript(html.replace('// ─── Render ───', '// rendering')), /boundary.*Render/);
  assert.throws(() => extractNumericalScript(html.replace('// ─── Helpers ───', '// ─── Render ───')), /boundary/);
  const api = loadModel();
  assert.deepEqual(plain(api.state.arch.weights), [], 'Loader must not execute app initialization');
});

test('evaluateNetwork is independent, leaves frozen network and inputs unchanged, and uses input-first weight storage', () => {
  const api = loadModel();
  assert.equal(typeof api.evaluateNetwork, 'function', 'Pure evaluateNetwork must be exported to the test boundary');
  const network = deepFreeze({ hidden: [2], weights: [[[1, -1], [2, 3]], [[2], [-1]]], biases: [[0.5, -0.5], [0.25]], inputs: [0, 0] });
  const inputs = Object.freeze([0.8, 0.3]);
  const before = JSON.stringify(network);
  const result = api.evaluateNetwork(network, inputs);
  assert.equal(JSON.stringify(network), before);
  assert.deepEqual(inputs, [0.8, 0.3]);
  assert.notEqual(result.acts[0], inputs, 'Returned activations must not alias the input array');
  assert.ok(Number.isFinite(result.output));
  closeTo(result.sums[1][0], 1.9);
  closeTo(result.sums[1][1], -0.4);
  closeTo(result.acts[1][0], 1.9);
  assert.equal(result.acts[1][1], 0);
  closeTo(result.sums[2][0], 4.05);
  closeTo(result.output, 0.9828759666842724);
});

test('seed-42 default forward wrapper retains 2-4-3-1 topology, 31 parameters, and output', () => {
  const api = loadModel();
  api.randomizeArchWeights();
  api.forward();
  assert.deepEqual(plain(api.layers()), [2, 4, 3, 1]);
  assert.equal(api.countParams(api.layers()), 31);
  closeTo(api.state.arch.acts.at(-1)[0], 0.4689096695309864);
  assert.equal(typeof api.evaluateNetwork, 'function');
  const result = api.evaluateNetwork(api.state.arch, [0.8, 0.3]);
  closeTo(result.output, 0.4689096695309864);
  assert.deepEqual(plain(api.state.arch.acts), plain(result.acts));
  assert.deepEqual(plain(api.state.arch.sums), plain(result.sums));
  api.state.arch.inputs = [1, 0];
  api.forward();
  assert.deepEqual(plain(api.state.arch.acts), plain(api.evaluateNetwork(api.state.arch, [1, 0]).acts));
});

for (const hidden of [[1], [6], [4, 3], [6, 6, 6]]) {
  test(`evaluateNetwork and all serialized values remain valid at input corners with hidden sizes ${hidden}`, () => {
    const api = loadModel();
    assert.equal(typeof api.evaluateNetwork, 'function');
    api.state.arch.hidden = hidden.slice();
    api.randomizeArchWeights();
    const network = api.state.arch;
    const before = JSON.stringify(network);
    for (const inputs of [[0, 0], [0, 1], [1, 0], [1, 1]]) {
      const result = api.evaluateNetwork(network, inputs);
      assert.equal(JSON.stringify(network), before);
      assert.ok(Number.isFinite(result.output));
      assert.ok(result.output > 0 && result.output < 1);
      assert.deepEqual(plain(result.acts.map((row) => row.length)), [2, ...hidden, 1]);
      assert.equal(result.sums[0], null);
      assert.ok(result.acts.flat().every(Number.isFinite));
      assert.ok(result.sums.slice(1).flat().every(Number.isFinite));
    }
    const file = api.buildSafetensors();
    const view = new DataView(file.bytes.buffer);
    const headerLen = Number(view.getBigUint64(0, true));
    const header = JSON.parse(new TextDecoder().decode(file.bytes.slice(8, 8 + headerLen)));
    let count = 0;
    for (let l = 0; l < hidden.length + 1; l++) {
      const nIn = l === 0 ? 2 : hidden[l - 1];
      const nOut = l === hidden.length ? 1 : hidden[l];
      const weight = header[`fc${l + 1}.weight`];
      const bias = header[`fc${l + 1}.bias`];
      assert.deepEqual(weight.shape, [nOut, nIn]);
      assert.deepEqual(bias.shape, [nOut]);
      for (let o = 0; o < nOut; o++) {
        for (let i = 0; i < nIn; i++) {
          const expected = network.weights[l][i][o];
          assert.equal(Math.fround(expected), expected);
          assert.equal(view.getFloat32(8 + headerLen + weight.data_offsets[0] + (o * nIn + i) * 4, true), expected);
          count++;
        }
        assert.equal(view.getFloat32(8 + headerLen + bias.data_offsets[0] + o * 4, true), network.biases[l][o]);
        count++;
      }
    }
    assert.equal(count, api.countParams([2, ...hidden, 1]));
    assert.equal(file.bytes.length, 8 + headerLen + count * 4);
  });
}

test('seed-42 Safetensors is 556 bytes and contains all 31 known float32 values in output-first tensor order', () => {
  const api = loadModel();
  api.randomizeArchWeights();
  const before = JSON.stringify(api.state.arch);
  const file = api.buildSafetensors();
  assert.equal(file.bytes.length, 556);
  assert.equal(file.headerLen, 424);
  assert.equal(file.dataStart, 432);
  assert.equal(file.dataLen, 124);
  const view = new DataView(file.bytes.buffer);
  const headerLen = Number(view.getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(file.bytes.slice(8, 8 + headerLen)));
  assert.deepEqual(Object.keys(header).filter((name) => name !== '__metadata__'), SEED_TENSORS.map((tensor) => tensor.name));
  let nextOffset = 0;
  for (const expected of SEED_TENSORS) {
    const metadata = header[expected.name];
    assert.equal(metadata.dtype, 'F32');
    assert.deepEqual(metadata.shape, expected.shape);
    assert.deepEqual(metadata.data_offsets, [nextOffset, nextOffset + expected.values.length * 4]);
    const actual = expected.values.map((_, i) => view.getFloat32(8 + headerLen + nextOffset + i * 4, true));
    assert.deepEqual(actual, expected.values, expected.name);
    nextOffset += expected.values.length * 4;
  }
  assert.equal(JSON.stringify(api.state.arch), before);
});

test('numerical checks report malformed shapes and nonfinite inputs without throwing or repairing state', () => {
  const api = loadModel();
  assert.equal(typeof api.numericalChecks, 'function');
  api.randomizeArchWeights();
  assert.ok(api.numericalChecks(api.state.arch).every((record) => record.pass));
  for (const corrupt of [
    (network) => { network.hidden = null; },
    (network) => { network.weights[0] = []; },
    (network) => { network.biases[0] = null; },
    (network) => { network.inputs = new Array(2); },
    (network) => { network.inputs[0] = NaN; },
    (network) => { network.weights[0][0][0] = Infinity; },
  ]) {
    const network = plain(api.state.arch);
    corrupt(network);
    const before = JSON.stringify(network);
    let records;
    assert.doesNotThrow(() => { records = api.numericalChecks(network); });
    assert.ok(records.some((record) => !record.pass));
    assert.ok(records.every((record) => typeof record.name === 'string' && typeof record.pass === 'boolean'));
    assert.equal(JSON.stringify(network), before);
  }
});

test('educational notices scope random weights to the explorer and distinguish estimation from actual float32 export', () => {
  const api = loadModel();
  api.randomizeArchWeights();
  for (const lang of ['pl', 'en']) {
    api.state.lang = lang;
    assert.match(api.t('untrainedNotice'), lang === 'pl' ? /losow.*nie.*nauczy/i : /random.*not learned/i);
    assert.match(api.t('precisionNotice'), lang === 'pl' ? /szacuje.*sieć.*pobierany plik.*float32/i : /estimates.*network.*downloaded file.*float32/i);
    assert.match(api.t('exportDtype'), /float32/);
    assert.match(api.t('archResetNotice'), lang === 'pl' ? /ponown.*losuj/i : /new random/i);
    assert.match(api.t('loadHf'), lang === 'pl' ? /obsługiwanych.*zarejestrowanych/i : /supported.*registered/i);
  }
  const originalWeights = JSON.stringify(api.state.arch.weights);
  const originalFile = Array.from(api.buildSafetensors().bytes);
  for (const precision of ['f32', 'f16', 'i8', 'q4']) {
    api.state.precision = precision;
    const file = api.buildSafetensors();
    assert.equal(JSON.stringify(api.state.arch.weights), originalWeights);
    assert.deepEqual(Array.from(file.bytes), originalFile);
    const header = JSON.parse(new TextDecoder().decode(file.bytes.slice(8, file.dataStart)));
    assert.ok(Object.entries(header).filter(([name]) => name !== '__metadata__').every(([, tensor]) => tensor.dtype === 'F32'));
  }
});

test('folder routes retain model and filename identity, with Llama files never routed to the mini binary inspector', () => {
  const api = loadModel();
  assert.equal(typeof api.modelFileDestination, 'function');
  const cases = [
    ['mini', 'config.json', 'configCard'],
    ['mini', 'model.safetensors', 'fileCard'],
    ['llama', 'config.json', 'configCard'],
    ['llama', 'generation_config.json', 'modelFileDetail'],
    ['llama', 'model-00001-of-00002.safetensors', 'modelFileDetail'],
    ['llama', 'model-00002-of-00002.safetensors', 'modelFileDetail'],
    ['llama', 'model.safetensors.index.json', 'modelFileDetail'],
    ['llama', 'tokenizer.json', 'modelFileDetail'],
    ['llama', 'tokenizer.model', 'modelFileDetail'],
  ];
  for (const [model, name, target] of cases) {
    assert.deepEqual(plain(api.modelFileDestination({ model, name })), { model, name, target });
  }
  for (const input of [null, {}, { model: 'unknown', name: 'config.json' }, { model: 'mini', name: 'tokenizer.json' }, { model: 'llama', name: '../config.json' }]) {
    assert.equal(api.modelFileDestination(input), null, 'Invalid file identity must not navigate');
  }
});

test('canonical parameter identity uses output-first tensor cells while network weights remain input-first', () => {
  const api = loadModel();
  assert.equal(typeof api.describeParameter, 'function', 'Canonical parameter lookup must be available');
  api.randomizeArchWeights();
  const ref = { layer: 0, kind: 'weight', output: 2, input: 1 };
  assert.equal(api.parameterId(ref), 'fc1.weight[2,1]');
  assert.deepEqual(plain(api.describeParameter(ref)), {
    id: 'fc1.weight[2,1]', value: -0.453544020652771, tensorName: 'fc1.weight',
    tensorShape: [4, 2], flatIndex: 5, fileOffset: 452, byteLength: 4,
  });
  assert.equal(api.describeParameter(ref).value, api.state.arch.weights[0][1][2]);
  assert.notEqual(api.describeParameter(ref).value, api.state.arch.weights[0][0][1]);
});

test('every rectangular-layer weight and bias resolves to its exact little-endian float32 file bytes', () => {
  const api = loadModel();
  assert.equal(typeof api.parameterId, 'function', 'Canonical parameter identity must be available');
  for (const hidden of [[3, 2], [6], [1], [2, 5, 3]]) {
    api.state.arch.hidden = hidden.slice();
    api.randomizeArchWeights();
    const file = api.buildSafetensors(), view = new DataView(file.bytes.buffer);
    const headerLen = Number(view.getBigUint64(0, true));
    const header = JSON.parse(new TextDecoder().decode(file.bytes.slice(8, 8 + headerLen)));
    const ls = [2, ...hidden, 1];
    const ids = new Set();
    for (let l = 0; l < ls.length - 1; l++) {
      for (let o = 0; o < ls[l + 1]; o++) {
        for (let i = 0; i < ls[l]; i++) {
          const ref = { layer: l, kind: 'weight', output: o, input: i }, info = api.describeParameter(ref);
          assert.equal(info.id, `fc${l + 1}.weight[${o},${i}]`);
          assert.deepEqual(plain(info.tensorShape), [ls[l + 1], ls[l]]);
          assert.equal(info.flatIndex, o * ls[l] + i);
          assert.equal(info.fileOffset, 8 + headerLen + header[info.tensorName].data_offsets[0] + (o * ls[l] + i) * 4);
          assert.equal(info.byteLength, 4);
          assert.equal(info.value, api.state.arch.weights[l][i][o]);
          assert.equal(view.getFloat32(info.fileOffset, true), info.value);
          ids.add(info.id);
        }
        const ref = { layer: l, kind: 'bias', output: o, input: null }, info = api.describeParameter(ref);
        assert.equal(info.id, `fc${l + 1}.bias[${o}]`);
        assert.deepEqual(plain(info.tensorShape), [ls[l + 1]]);
        assert.equal(info.flatIndex, o);
        assert.equal(info.fileOffset, 8 + headerLen + header[info.tensorName].data_offsets[0] + o * 4);
        assert.equal(info.value, api.state.arch.biases[l][o]);
        assert.equal(view.getFloat32(info.fileOffset, true), info.value);
        ids.add(info.id);
      }
    }
    assert.equal(ids.size, api.countParams(ls));
  }
});

test('invalid parameter references are rejected before selection or file lookup can mutate state', () => {
  const api = loadModel();
  assert.equal(typeof api.selectParameter, 'function', 'Canonical selection must be available');
  api.randomizeArchWeights();
  api.forward();
  for (const ref of [null, {}, { layer: 0, kind: 'other', output: 0, input: 0 },
    { layer: -1, kind: 'weight', output: 0, input: 0 }, { layer: 3, kind: 'weight', output: 0, input: 0 },
    { layer: 0.5, kind: 'weight', output: 0, input: 0 }, { layer: 0, kind: 'weight', output: 4, input: 0 },
    { layer: 0, kind: 'weight', output: 0, input: 2 }, { layer: 0, kind: 'weight', output: 0, input: null },
    { layer: 0, kind: 'bias', output: 0, input: 0 }, { layer: 0, kind: 'bias', output: NaN, input: null }]) {
    const before = JSON.stringify(api.state);
    for (const lookup of [api.parameterId, api.describeParameter, api.selectParameter]) assert.throws(() => lookup(ref), /parameter/i);
    assert.equal(JSON.stringify(api.state), before);
  }
});

test('selection follows the parameter output neuron and preserves identity across inputs and randomization', () => {
  const api = loadModel();
  assert.equal(typeof api.selectParameter, 'function', 'Canonical selection must be available');
  api.randomizeArchWeights();
  api.forward();
  const ref = { layer: 1, kind: 'weight', output: 2, input: 3 };
  api.selectParameter(ref);
  ref.output = 0;
  assert.deepEqual(plain(api.state.selection.parameter), { layer: 1, kind: 'weight', output: 2, input: 3 });
  const selected = plain(api.state.selection.parameter), original = api.describeParameter(selected);
  const bytes = Array.from(api.buildSafetensors().bytes);
  api.state.arch.inputs = [0, 1];
  api.forward();
  assert.deepEqual(plain(api.state.selection.parameter), selected);
  assert.deepEqual(plain(api.describeParameter(selected)), plain(original));
  assert.deepEqual(Array.from(api.buildSafetensors().bytes), bytes);
  api.randomizeArchWeights();
  api.forward();
  assert.deepEqual(plain(api.state.selection.parameter), selected);
  const refreshed = api.describeParameter(selected);
  assert.equal(refreshed.id, original.id);
  assert.notEqual(refreshed.value, original.value);
  assert.equal(new DataView(api.buildSafetensors().bytes.buffer).getFloat32(refreshed.fileOffset, true), refreshed.value);
  api.selectParameter({ layer: 2, kind: 'bias', output: 0, input: null });
  assert.deepEqual(plain(api.state.selection.parameter), { layer: 2, kind: 'bias', output: 0, input: null });
});

test('topology reinitialization replaces invalid selection with a valid parameter without aliasing the old reference', () => {
  const api = loadModel();
  assert.equal(typeof api.selectParameter, 'function', 'Canonical selection must be available');
  api.randomizeArchWeights();
  api.forward();
  api.selectParameter({ layer: 1, kind: 'weight', output: 2, input: 3 });
  const previous = api.state.selection.parameter;
  api.state.arch.hidden = [1];
  api.randomizeArchWeights();
  api.forward();
  api.clampSelection();
  assert.deepEqual(plain(api.state.selection.parameter), { layer: 1, kind: 'weight', output: 0, input: 0 });
  assert.deepEqual(plain(previous), { layer: 1, kind: 'weight', output: 2, input: 3 });
  assert.equal(api.parameterId(api.state.selection.parameter), 'fc2.weight[0,0]');
  assert.equal(new DataView(api.buildSafetensors().bytes.buffer).getFloat32(api.describeParameter(api.state.selection.parameter).fileOffset, true), api.describeParameter(api.state.selection.parameter).value);
});

test('every translation key exists in both languages', () => {
  const api = loadModel();
  assert.deepEqual(Object.keys(api.T.pl).sort(), Object.keys(api.T.en).sort(), 'Every translation key must exist in both languages');
});
