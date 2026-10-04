import assert from 'node:assert/strict';
import { mkdir, readFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { join } from 'node:path';
import { APP_URL } from './model-loader.mjs';

const ORIGINAL_CHECKS = [
  'arch: node count = sum of layers', 'arch: edge count = sum a·b', 'arch: params = edges + biases',
  'forward: output node = recomputed σ(...)',
  'config.json hidden_sizes = architecture', 'safetensors: header length, contiguous offsets, size = 4·params',
  'safetensors: first float = W(1)[0][0]', 'size: bytes = params × precision',
  'load code output = model output', 'lang attr matches state',
];

let stage = 'load Playwright';
let server;
let browser;
let disconnected = false;

async function inspectDownloads(page) {
  await page.locator('[data-prec="q4"]').click();
  await page.evaluate(() => MODEL_DEMO.startTraining());
  const [download] = await Promise.all([page.waitForEvent('download'), page.locator('#dlWeightsBtn').click()]);
  const bytes = await readFile(await download.path());
  assert.equal(download.suggestedFilename(), 'model.safetensors');
  const snapshot = await page.evaluate(() => ({ network: MODEL_DEMO.state.arch, phase: MODEL_DEMO.state.lesson.phase, output: MODEL_DEMO.evaluateNetwork(MODEL_DEMO.state.arch, MODEL_DEMO.state.arch.inputs).output }));
  assert.equal(snapshot.phase, 'paused', 'Downloading pauses training');
  const headerLength = Number(bytes.readBigUInt64LE(0));
  const header = JSON.parse(bytes.subarray(8, 8 + headerLength).toString('utf8'));
  const layers = [2, ...snapshot.network.hidden, 1];
  let offset = 0, activations = snapshot.network.inputs;
  for (let layer = 0; layer < layers.length - 1; layer++) {
    for (const kind of ['weight', 'bias']) {
      const tensor = header[`fc${layer + 1}.${kind}`];
      const values = kind === 'weight'
        ? Array.from({ length: layers[layer + 1] }, (_, o) => Array.from({ length: layers[layer] }, (_, i) => snapshot.network.weights[layer][i][o])).flat()
        : snapshot.network.biases[layer];
      assert.equal(tensor.dtype, 'F32');
      assert.deepEqual(tensor.shape, kind === 'weight' ? [layers[layer + 1], layers[layer]] : [layers[layer + 1]]);
      assert.deepEqual(tensor.data_offsets, [offset, offset + values.length * 4]);
      for (const value of values) { assert.equal(bytes.readFloatLE(8 + headerLength + offset), value); offset += 4; }
    }
    const w = header[`fc${layer + 1}.weight`], b = header[`fc${layer + 1}.bias`];
    activations = Array.from({ length: layers[layer + 1] }, (_, o) => {
      let sum = bytes.readFloatLE(8 + headerLength + b.data_offsets[0] + o * 4);
      for (let i = 0; i < layers[layer]; i++) sum += activations[i] * bytes.readFloatLE(8 + headerLength + w.data_offsets[0] + (o * layers[layer] + i) * 4);
      return layer === layers.length - 2 ? 1 / (1 + Math.exp(-sum)) : Math.max(0, sum);
    });
  }
  assert.equal(bytes.length, 8 + headerLength + offset);
  assert.ok(Math.abs(activations[0] - snapshot.output) < 1e-12);
  await page.locator('[data-config="mini"]').click();
  const [configDownload] = await Promise.all([page.waitForEvent('download'), page.locator('#dlConfigBtn').click()]);
  const config = JSON.parse(await readFile(await configDownload.path(), 'utf8'));
  assert.deepEqual(config.hidden_sizes, snapshot.network.hidden);
  assert.equal(config.torch_dtype, 'float32');
}

async function inspectFruitTraining(page, label) {
  const saved=await page.evaluate(()=>JSON.stringify([MODEL_DEMO.state.arch,MODEL_DEMO.state.selection]));
  assert.equal(await page.locator('#fruitLesson').evaluate(node=>node.open),false,`${label}: fruit lesson starts collapsed`);
  assert.equal(await page.locator('#fruitEnter').isVisible(),false);
  await page.locator('#fruitLesson > summary').click();
  await page.locator('#fruitEnter').click();
  await page.mouse.move(0,0); // entering scrolls to the diagram; keep the pointer from hover-selecting a plot point
  assert.equal(await page.locator('#fruitLesson').evaluate(node=>node.open),true);
  assert.equal(await page.locator('#fruitExit').isVisible(),true,`${label}: exit stays reachable`);
  assert.equal(await page.locator('#fruitLesson').getAttribute('data-training-mode'),'fruit');
  assert.equal(await page.locator('#fruitSampleImage').evaluate(img=>img.complete&&img.naturalWidth===194),true);
  assert.equal(await page.locator('#fruitDecisionPlot [data-grid-x]').count(),400);
  await inspect(page,`${label} fruit initial`);
  await page.locator('#fruitDecisionPlot [data-sample-id="train-strawberry-1"]').focus();
  await page.keyboard.press('t');
  assert.equal(await page.evaluate(()=>MODEL_DEMO.state.lesson.phase),'running',`${label}: T from focused point starts training`);
  await page.waitForFunction(()=>MODEL_DEMO.state.lesson.epoch>=20);
  assert.equal(await page.evaluate(()=>MODEL_DEMO.state.lesson.phase),'running',`${label}: programmatic focus restoration does not pause training`);
  await page.keyboard.press('t');
  assert.equal(await page.evaluate(()=>MODEL_DEMO.state.lesson.phase),'paused');
  const shortcutState=await page.evaluate(()=>JSON.stringify([MODEL_DEMO.state.lesson.epoch,MODEL_DEMO.state.arch.weights]));
  await page.locator('#fruitRate').focus(); await page.keyboard.press('t'); await page.keyboard.press('r');
  assert.equal(await page.evaluate(()=>MODEL_DEMO.state.lesson.phase),'paused',`${label}: selector keys are not shortcuts`);
  assert.equal(await page.evaluate(()=>JSON.stringify([MODEL_DEMO.state.lesson.epoch,MODEL_DEMO.state.arch.weights])),shortcutState);
  await page.locator('#fruitReset').click();
  const initial=await page.evaluate(()=>MODEL_DEMO.fruitMetrics(MODEL_DEMO.state.arch).loss);
  await page.locator('#fruitTrain').click();
  await page.waitForFunction(()=>MODEL_DEMO.state.lesson.epoch>=20);
  await page.locator('#fruitPause').click();
  await inspect(page,`${label} fruit paused`);
  const paused=await page.evaluate(()=>({epoch:MODEL_DEMO.state.lesson.epoch,weights:JSON.stringify(MODEL_DEMO.state.arch.weights),loss:MODEL_DEMO.fruitMetrics(MODEL_DEMO.state.arch).loss}));
  assert.ok(paused.loss<initial);
  await page.locator('#fruitStep').click();
  assert.equal(await page.evaluate(()=>MODEL_DEMO.state.lesson.epoch),paused.epoch+1);
  await page.locator('#fruitUnseen').check();
  await page.locator('#fruitSampleSelect').selectOption('test-strawberry-1');
  assert.equal(await page.locator('#fruitDecisionPlot [data-split="test"]').count(),4);
  await inspect(page,`${label} fruit held-out`);
  const hash=await page.evaluate(()=>JSON.stringify([...MODEL_DEMO.buildSafetensors().bytes]));
  await page.locator('#x1').evaluate(input=>{input.value='5.5';input.dispatchEvent(new Event('input',{bubbles:true}));});
  assert.equal(await page.evaluate(()=>MODEL_DEMO.state.lesson.selectedSampleId),null);
  assert.equal(await page.locator('#fruitSample').getAttribute('data-target'),'');
  assert.equal(await page.evaluate(()=>JSON.stringify([...MODEL_DEMO.buildSafetensors().bytes])),hash);
  await page.locator('#fruitTrain').click();
  await page.locator('[data-lang-btn="en"]').click();
  assert.equal(await page.evaluate(()=>MODEL_DEMO.state.lesson.phase),'paused');
  await inspect(page,`${label} fruit language handoff`);
  await inspectDownloads(page);
  await page.locator('#fruitRate').selectOption('0.5');
  await page.locator('#fruitReset').click();
  assert.equal(await page.evaluate(()=>MODEL_DEMO.state.lesson.epoch),0);
  assert.equal(await page.evaluate(()=>MODEL_DEMO.state.lesson.learningRate),.5);
  await inspect(page,`${label} fruit reset`);
  const originalViewport = page.viewportSize();
  await page.setViewportSize({ width: 375, height: 844 });
  const contained = await page.evaluate(() => {
    const panel = document.getElementById('fruitTrainingPanel');
    const right = panel.getBoundingClientRect().right - parseFloat(getComputedStyle(panel).paddingRight);
    return ['fruitSampleSelect', 'fruitDecisionPlot'].every(id => document.getElementById(id).getBoundingClientRect().right <= right + 1);
  });
  assert.equal(contained, true, 'Mobile training controls and plot remain inside card padding');
  await page.setViewportSize(originalViewport);
  if(process.env.SMOKE_SCREENSHOT_DIR && label==='http') {
    await page.evaluate(()=>{for(let i=0;i<30;i++)MODEL_DEMO.stepTraining();});
    await mkdir(process.env.SMOKE_SCREENSHOT_DIR,{recursive:true});
    await page.locator('#diagramCard').scrollIntoViewIfNeeded();
    await page.screenshot({path:join(process.env.SMOKE_SCREENSHOT_DIR,'fruit-desktop.png')});
    await page.setViewportSize({width:390,height:844});
    await inspect(page,'fruit mobile');
    await page.locator('#fruitTrainingPanel').scrollIntoViewIfNeeded();
    await page.screenshot({path:join(process.env.SMOKE_SCREENSHOT_DIR,'fruit-mobile.png')});
    await page.setViewportSize({width:1280,height:900});
  }
  await page.locator('#fruitExit').click();
  assert.equal(await page.evaluate(()=>JSON.stringify([MODEL_DEMO.state.arch,MODEL_DEMO.state.selection])),saved);
  await inspect(page,`${label} restored explorer`);
}

async function inspect(page, label) {
  const snapshot = await page.evaluate(() => {
    const api = window.MODEL_DEMO;
    if (typeof api?.runChecks !== 'function' || typeof api?.evaluateNetwork !== 'function') {
      return { missingApi: true };
    }
    const before = JSON.stringify(api.state);
    const checks = api.runChecks();
    const result = api.evaluateNetwork(api.state.arch, api.state.arch.inputs);
    return { checks, original: api.probe(), unchanged: JSON.stringify(api.state) === before, output: result.output };
  });
  assert.equal(snapshot.missingApi, undefined, `${label}: MODEL_DEMO needs runChecks and evaluateNetwork`);
  assert.equal(snapshot.unchanged, true, `${label}: verification must not mutate state`);
  assert.ok(Number.isFinite(snapshot.output));
  assert.ok(snapshot.checks.length > ORIGINAL_CHECKS.length, `${label}: pure checks must complement the original probe`);
  for (const name of ORIGINAL_CHECKS) {
    assert.ok(snapshot.original.some((record) => record.name === name && record.pass), `${label}: original ${name}`);
    assert.ok(snapshot.checks.some((record) => record.name === name && record.pass), `${label}: combined ${name}`);
  }
  assert.deepEqual(snapshot.checks.filter((record) => !record.pass), [], `${label}: failed checks`);
  if (label.endsWith('baseline')) {
    const corruptions = await page.evaluate(() => {
      const api = window.MODEL_DEMO;
      return [
        ['.file-row', 'data-model', 'wrong'],
        ['#archSvg .edge', 'data-param-id', 'wrong'],
        ['#secNav [data-section]', 'data-section', 'wrong'],
      ].map(([selector, attr, value]) => {
        const node = document.querySelector(selector), previous = node.getAttribute(attr);
        const before = JSON.stringify(api.state);
        node.setAttribute(attr, value);
        try { return { failed: api.runChecks().some(check => !check.pass), unchanged: JSON.stringify(api.state) === before }; }
        finally { node.setAttribute(attr, previous); }
      });
    });
    assert.deepEqual(corruptions, Array.from({ length: 3 }, () => ({ failed: true, unchanged: true })), `${label}: shared navigation and diagram contracts detect corruption`);
  }
  return snapshot;
}

async function inspectFileNavigation(page, label) {
  const weightsBefore = await page.evaluate(() => JSON.stringify(window.MODEL_DEMO.state.arch.weights));
  for (const lang of ['pl', 'en']) {
    await page.locator(`[data-lang-btn="${lang}"]`).click();
    assert.equal(await page.locator('#untrainedNotice').isVisible(), true, `${label}: early explorer warning`);
    assert.match(await page.locator('#precisionNotice').textContent(), /float32/);
    assert.equal(await page.locator('#precisionNotice').getAttribute('data-estimate-only'), 'true');
    await page.locator('[data-config="llama"]').click();
    assert.match(await page.locator('#exportDtype').textContent(), /float16/);
    assert.equal(await page.locator('#fileCard').getAttribute('data-model'), 'llama');
    assert.equal(await page.locator('#fileCard').getAttribute('data-dtype'), 'F16');
    assert.equal(await page.locator('[data-file-tab="hex"]').isVisible(), false, `${label}: Llama has no raw-byte view`);
    assert.equal(await page.locator('#dlWeightsBtn').isVisible(), false);
    assert.match(await page.locator('#fileView').textContent(), /model\.layers\.0\.self_attn\.q_proj\.weight.*F16.*\[4096, 4096\]/);
    await page.locator('[data-config="mini"]').click();
    assert.match(await page.locator('#exportDtype').textContent(), /float32/);
    assert.equal(await page.locator('#fileCard').getAttribute('data-dtype'), 'F32');
    assert.equal(await page.locator('[data-file-tab="hex"]').isVisible(), true);
    for (const model of ['mini', 'llama']) {
      await page.locator(`[data-folder="${model}"]`).click();
      const names = await page.locator('#folderList .file-row').evaluateAll((rows) => rows.map((row) => row.dataset.fileName));
      assert.equal(names.length, model === 'mini' ? 2 : 7);
      for (const name of names) {
        const row = page.locator(`#folderList .file-row[data-file-name="${name}"]`);
        assert.equal(await row.getAttribute('data-model'), model);
        await row.focus();
        await page.keyboard.press('Enter');
        const target = name === 'config.json' ? '#configCard' : model === 'mini' ? '#fileCard' : '#modelFileDetail';
        assert.equal(await page.locator(target).getAttribute('data-model'), model, `${label}: destination model for ${name}`);
        assert.equal(await page.locator(target).getAttribute('data-file-name'), name, `${label}: destination file for ${name}`);
        assert.equal(await page.evaluate(() => document.activeElement?.id), target.slice(1), `${label}: keyboard file navigation focuses its destination`);
        if (model === 'llama' && name !== 'config.json') {
          assert.equal(await page.locator('#modelFileDetail').isVisible(), true);
          assert.match(await page.locator('#modelFileDetail').textContent(), /Llama 2 7B/);
          assert.match(await page.locator('#miniInspectorExample').textContent(), lang === 'pl' ? /mini-model/i : /mini.model/i);
          assert.equal(await page.locator('#fileCard').getAttribute('data-model'), 'llama', `${label}: file card follows the selected model`);
        }
      }
    }
    await page.locator('#miniInspectorExample').focus();
    await page.keyboard.press('Enter');
    assert.equal(await page.locator('#fileCard').getAttribute('data-model'), 'mini');
    assert.equal(await page.locator('#fileCard').getAttribute('data-file-name'), 'model.safetensors');
    assert.equal(await page.locator('#modelFileDetail').isVisible(), false);
    assert.equal(await page.evaluate(() => document.activeElement?.id), 'fileCard', `${label}: hidden Llama detail does not retain focus`);
    await page.locator('[data-prec="q4"]').click();
    assert.equal(await page.locator('#fileCard').getAttribute('data-dtype'), 'F32');
    assert.equal(await page.evaluate(() => JSON.stringify(window.MODEL_DEMO.state.arch.weights)), weightsBefore);
  }
  await page.locator('[data-lang-btn="pl"]').click();
}

async function inspectLinkedParameters(page, label) {
  for (const lang of ['pl', 'en']) {
    await page.locator(`[data-lang-btn="${lang}"]`).click();
    await page.locator('[data-config="mini"]').click();
    await page.locator('[data-file-tab="hex"]').click();
    for (const [ref, id] of [
      [{ layer: 0, kind: 'weight', output: 2, input: 1 }, 'fc1.weight[2,1]'],
      [{ layer: 0, kind: 'bias', output: 2, input: null }, 'fc1.bias[2]'],
      [{ layer: 1, kind: 'weight', output: 2, input: 3 }, 'fc2.weight[2,3]'],
    ]) {
      const selection = await page.evaluate((parameter) => {
        const api = window.MODEL_DEMO, info = api.selectParameter(parameter);
        const bytes = [...document.querySelectorAll('#fileView [data-byte-offset][data-selected="true"]')]
          .map((span) => ({ id: span.dataset.paramId, offset: +span.dataset.byteOffset, hex: span.textContent }));
        return { info, bytes, selected: api.state.selection.parameter };
      }, ref);
      assert.deepEqual(selection.selected, ref);
      if (ref.kind === 'weight') assert.equal(await page.locator(`#archSvg .edge[data-param-id="${id}"][data-selected="true"]`).count(), 1);
      assert.equal(await page.locator('#fileCard').getAttribute('data-param-id'), id);
      assert.equal(selection.bytes.length, 4, `${label}: exactly four selected hex byte spans, not duplicate ASCII highlights`);
      assert.deepEqual(selection.bytes.map(({ id: byteId }) => byteId), [id, id, id, id]);
      assert.deepEqual(selection.bytes.map(({ offset }) => offset), Array.from({ length: 4 }, (_, i) => selection.info.fileOffset + i));
      const buffer = Uint8Array.from(selection.bytes.map(({ hex }) => Number.parseInt(hex, 16)));
      assert.equal(new DataView(buffer.buffer).getFloat32(0, true), selection.info.value);
      await inspect(page, `${label}: linked ${id} in ${lang}`);
    }
  }
  await page.locator('[data-lang-btn="pl"]').click();
  assert.match(await page.locator('#paramBreakdown').textContent(), /23.*8.*31/);
  await page.setViewportSize({ width: 375, height: 812 });
  const mobile = await page.evaluate(() => ({ width: document.documentElement.scrollWidth, viewport: innerWidth }));
  assert.ok(mobile.width <= mobile.viewport, `${label}: no page-level horizontal overflow at 375px`);
  assert.equal(await page.locator('#archSvg').isVisible(), true, `${label}: mobile shows the network diagram`);
  await inspect(page, `${label}: mobile linked detail`);
  if (label === 'http' && process.env.SMOKE_SCREENSHOT_DIR) {
    await mkdir(process.env.SMOKE_SCREENSHOT_DIR, { recursive: true });
    await page.locator('#fileCard').scrollIntoViewIfNeeded();
    await page.screenshot({ path: join(process.env.SMOKE_SCREENSHOT_DIR, 'parameter-mobile.png') });
  }
  await page.setViewportSize({ width: 1400, height: 900 });
  if (label === 'http' && process.env.SMOKE_SCREENSHOT_DIR) {
    await page.locator('#fileCard').scrollIntoViewIfNeeded();
    await page.screenshot({ path: join(process.env.SMOKE_SCREENSHOT_DIR, 'parameter-desktop.png') });
  }
}

async function inspectOutlineAndShortcuts(page, label) {
  assert.equal(await page.locator('main > .section').count(), 5, `${label}: original five-section outline`);
  assert.equal(await page.locator('#secNav [data-section]').count(), 5);
  for (const id of ['flowControls', 'parameterInspector', 'neuronPanel', 'mathCard', 's-llm']) {
    assert.equal(await page.locator(`#${id}`).count(), 0, `${label}: removed ${id} stays removed`);
  }
  assert.equal(await page.locator('#archSvg [tabindex], #archSvg [role="button"]').count(), 0, `${label}: diagram is display-only`);
  const id = 'fc1.weight[2,1]';
  await page.evaluate(() => window.MODEL_DEMO.selectParameter({ layer: 0, kind: 'weight', output: 2, input: 1 }));
  const bytesBefore = await page.evaluate(() => Array.from(window.MODEL_DEMO.buildSafetensors().bytes));
  await page.keyboard.press('f');
  assert.deepEqual(await page.evaluate(() => Array.from(window.MODEL_DEMO.buildSafetensors().bytes)), bytesBefore, `${label}: F is no longer a shortcut`);
  await page.keyboard.press('r');
  assert.deepEqual(await page.evaluate(() => Array.from(window.MODEL_DEMO.buildSafetensors().bytes)), bytesBefore, `${label}: R is no longer a shortcut`);
  assert.equal(await page.locator('#weightsCard').count(), 0, `${label}: weights list stays removed`);
  assert.equal(await page.locator('#fileCard').getAttribute('data-param-id'), id);
  await inspect(page, `${label} shortcuts`);
}

try {
  const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
  stage = 'HTTP server startup';
  const html = await readFile(APP_URL);
  server = createServer((request, response) => {
    if (!['/', '/index.html'].includes(new URL(request.url, 'http://localhost').pathname)) {
      response.writeHead(404).end();
      return;
    }
    response.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8', 'Cache-Control': 'no-store' });
    response.end(html);
  });
  server.requestTimeout = 5000;
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  const httpUrl = `http://127.0.0.1:${server.address().port}/index.html`;
  stage = 'browser launch';
  browser = await chromium.launch({
    executablePath: process.env.CHROME_BIN || undefined,
    headless: true,
    timeout: 10000,
    args: ['--no-sandbox', '--disable-dev-shm-usage'],
  });
  browser.on('disconnected', () => { disconnected = true; });
  const context = await browser.newContext({ viewport: { width: 1400, height: 900 } });
  context.setDefaultTimeout(5000);
  await context.route('https://fonts.googleapis.com/**', (route) => route.abort());

  for (const [label, url] of [['file', APP_URL.href], ['http', httpUrl]]) {
    stage = `${label} page initialization`;
    const page = await context.newPage();
    page.on('crash', () => console.error(`Renderer crashed during ${stage}; browser connected=${browser.isConnected()}`));
    const pageErrors = [];
    page.on('pageerror', (error) => pageErrors.push(String(error)));
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 10000 });
    await page.waitForFunction(() => !!window.MODEL_DEMO);
    const baseline = await inspect(page, `${label} baseline`);
    assert.ok(Math.abs(baseline.output - 0.4689096695309864) < 1e-12);
    assert.equal(await page.locator('#archControls').getAttribute('data-layers'), '2-4-3-1');
    assert.equal(await page.evaluate(() => window.MODEL_DEMO.buildSafetensors().bytes.length), 524);

    stage = `${label} bilingual model-file identity`;
    await inspectFileNavigation(page, label);
    await inspect(page, `${label} file navigation`);

    stage = `${label} outline and shortcuts`;
    await inspectOutlineAndShortcuts(page, label);

    stage = `${label} linked parameters`;
    await inspectLinkedParameters(page, label);

    stage = `${label} live fruit training`;
    await inspectFruitTraining(page,label);

    stage = `${label} malformed state and missing contract node`;
    const corruption = await page.evaluate(() => {
      const api = window.MODEL_DEMO;
      const oldWeights = api.state.arch.weights[0];
      const malformed = () => {
        api.state.arch.weights[0] = [];
        const before = JSON.stringify(api.state);
        try { return { failed: api.runChecks().some((record) => !record.pass), unchanged: JSON.stringify(api.state) === before }; }
        finally { api.state.arch.weights[0] = oldWeights; }
      };
      const malformedResult = malformed();
      const node = document.getElementById('fileCard');
      const parent = node.parentNode, next = node.nextSibling;
      const before = JSON.stringify(api.state);
      node.remove();
      try { return { malformed: malformedResult, missingNode: { failed: api.runChecks().some((record) => !record.pass), unchanged: JSON.stringify(api.state) === before } }; }
      finally { parent.insertBefore(node, next); }
    });
    assert.deepEqual(corruption, { malformed: { failed: true, unchanged: true }, missingNode: { failed: true, unchanged: true } });
    await inspect(page, `${label} recovered`);

    stage = `${label} language persistence and input corners`;
    await page.locator('[data-lang-btn="en"]').click();
    await page.reload({ waitUntil: 'domcontentloaded', timeout: 10000 });
    await page.waitForFunction(() => !!window.MODEL_DEMO);
    assert.equal(await page.locator('html').getAttribute('lang'), 'en');
    for (const inputs of [[0, 0], [0, 1], [1, 0], [1, 1]]) {
      await page.evaluate((values) => values.forEach((value, index) => {
        const input = document.getElementById(`x${index + 1}`);
        input.value = value;
        input.dispatchEvent(new Event('input', { bubbles: true }));
      }), inputs);
      await inspect(page, `${label} inputs ${inputs}`);
    }
    await page.locator('[data-lang-btn="pl"]').click();
    await page.locator('#layerMinus').click();
    await inspect(page, `${label} changed architecture`);
    await page.locator('#layerPlus').click();
    await page.locator('[data-prec="q4"]').click();
    await page.locator('[data-config="mini"]').click();
    await page.locator('[data-file-tab="hex"]').click();
    await inspect(page, `${label} precision and file tabs`);

    assert.deepEqual(pageErrors, [], `${label}: page runtime errors`);
    console.log(`${label}: ${baseline.checks.length} checks passed; corruption, persistence and interaction smoke passed`);
    await page.close();
  }
} catch (error) {
  console.error(`Smoke failed during ${stage}; browser disconnected=${disconnected}: ${error.stack || error}`);
  process.exitCode = 1;
} finally {
  await browser?.close();
  if (server?.listening) await new Promise((resolve) => server.close(resolve));
}
