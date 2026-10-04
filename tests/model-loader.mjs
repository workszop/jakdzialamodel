import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

export const APP_URL = new URL('../index.html', import.meta.url);
const SECTION_MARKERS = ['Constants', 'Translations', 'State', 'DOM refs', 'Helpers', 'File builders', 'Precision helpers', 'Render', 'Listeners', 'Init'];

export function extractNumericalScript(html) {
  const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)];
  assert.equal(scripts.length, 1, 'Expected exactly one inline application script');
  const script = scripts[0][1];
  let lastIndex = -1;
  for (const section of SECTION_MARKERS) {
    const marker = `// ─── ${section} ───`;
    const index = script.indexOf(marker);
    assert.ok(index > lastIndex, `Missing or reordered script boundary: ${marker}`);
    assert.equal(script.lastIndexOf(marker), index, `Duplicated script boundary: ${marker}`);
    lastIndex = index;
  }
  assert.equal(script.trimStart().startsWith('// ─── Constants ───'), true, 'Constants must be the first script section');
  return script.slice(0, script.indexOf('// ─── Render ───'));
}

export function createTestClock() {
  let nextId = 0;
  const pending = new Map();
  const delays = [];
  return {
    setTimeout(callback, delay) { const id = ++nextId; pending.set(id, callback); delays.push(delay); return id; },
    delays,
    clearTimeout(id) { pending.delete(id); },
    get pendingCount() { return pending.size; },
    nextCallback() { return pending.values().next().value; },
    tick() {
      const [id, callback] = pending.entries().next().value ?? [];
      if (!callback) throw new Error('No scheduled forward step');
      pending.delete(id);
      callback();
    },
  };
}

export function loadModel(html = readFileSync(APP_URL, 'utf8'), { clock = createTestClock(), reducedMotion = false, requestFrame = (callback) => callback(), onRender = () => {} } = {}) {
  const context = vm.createContext({
    document: { getElementById: () => ({ dataset: {} }) },
    localStorage: { getItem: () => null, setItem: () => {} },
    renderDynamic: onRender,
    renderAll: () => {},
    setTimeout: (callback, delay) => clock.setTimeout(callback, delay),
    clearTimeout: (id) => clock.clearTimeout(id),
    matchMedia: () => ({ matches: reducedMotion }),
    requestAnimationFrame: requestFrame,
    TextEncoder,
    TextDecoder,
  });
  vm.runInContext(`${extractNumericalScript(html)}
    globalThis.testApi = {
      state, forward, randomizeArchWeights, layers, countParams,
      buildConfig, configText, buildSafetensors, sigmoid, relu,
      T, t,
      modelFileDestination: typeof modelFileDestination === 'function' ? modelFileDestination : undefined,
      llmComparisonData: typeof llmComparisonData === 'function' ? llmComparisonData : undefined,
      evaluateNetwork: typeof evaluateNetwork === 'function' ? evaluateNetwork : undefined,
      numericalChecks: typeof numericalChecks === 'function' ? numericalChecks : undefined,
      parameterId: typeof parameterId === 'function' ? parameterId : undefined,
      describeParameter: typeof describeParameter === 'function' ? describeParameter : undefined,
      parameterArithmetic: typeof parameterArithmetic === 'function' ? parameterArithmetic : undefined,
      selectParameter: typeof selectParameter === 'function' ? selectParameter : undefined,
      startFlow: typeof startFlow === 'function' ? startFlow : undefined,
      pauseFlow: typeof pauseFlow === 'function' ? pauseFlow : undefined,
      stepFlow: typeof stepFlow === 'function' ? stepFlow : undefined,
      resetFlow: typeof resetFlow === 'function' ? resetFlow : undefined,
      cancelFlow: typeof cancelFlow === 'function' ? cancelFlow : undefined,
      runFlow: typeof runFlow === 'function' ? runFlow : undefined,
      setInputs: typeof setInputs === 'function' ? setInputs : undefined,
      setArchitecture: typeof setArchitecture === 'function' ? setArchitecture : undefined,
      setLanguage: typeof setLanguage === 'function' ? setLanguage : undefined,
      signedValue,
      createFruitNetwork: typeof createFruitNetwork === 'function' ? createFruitNetwork : undefined,
      computeBatchGradients: typeof computeBatchGradients === 'function' ? computeBatchGradients : undefined,
      trainEpoch: typeof trainEpoch === 'function' ? trainEpoch : undefined,
      fruitMetrics: typeof fruitMetrics === 'function' ? fruitMetrics : undefined,
      FRUIT_SAMPLES: typeof FRUIT_SAMPLES !== 'undefined' ? FRUIT_SAMPLES : undefined,
      enterFruitLesson: typeof enterFruitLesson === 'function' ? enterFruitLesson : undefined,
      startTraining: typeof startTraining === 'function' ? startTraining : undefined,
      pauseTraining: typeof pauseTraining === 'function' ? pauseTraining : undefined,
      stepTraining: typeof stepTraining === 'function' ? stepTraining : undefined,
      resetTraining: typeof resetTraining === 'function' ? resetTraining : undefined,
      exitFruitLesson: typeof exitFruitLesson === 'function' ? exitFruitLesson : undefined,
      selectFruitSample: typeof selectFruitSample === 'function' ? selectFruitSample : undefined,
      setLearningRate: typeof setLearningRate === 'function' ? setLearningRate : undefined,
      get liveAnnouncement() { return liveRegion.textContent; },
      clampSelection,
    };`, context, { filename: APP_URL.pathname, timeout: 1000 });
  return context.testApi;
}
