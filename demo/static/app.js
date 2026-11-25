'use strict';
const $ = id => document.getElementById(id);
const player = $('player');
let mediaCatalog = [];
let builtinMedia = $('demo').dataset.src;
let selectedMedia = builtinMedia;
let audioContext, source, captureNode, fallback, objectURL, sid = null, epoch = 0;
let active = false, capturing = false, seq = 0, pipeline = Promise.resolve(), operations = Promise.resolve();
let ignorePause = 0, ignorePlay = 0, seekTimer, lastState, flushNumber = 0;
let seekingSession = false, seekResume = false;
let playBeganAt = null, firstVisibleS = null;
let inputRate = 0;
let sessionPreviewPolicy = null, failure = null, reconnectTimer, reloadTimer, modelStatus = 'ready', modelRecovered = false;
const pendingRequests = new Set();
function clearReconnect() { clearTimeout(reconnectTimer); reconnectTimer = null; }
async function request(path, options = {}, timeoutMs = 15000) {
  const controller = new AbortController(); pendingRequests.add(controller);
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(path, {...options, signal: controller.signal});
    let data;
    try { data = await response.json(); }
    catch (error) { if (controller.signal.aborted) throw error; throw Error(response.ok ? "Invalid JSON response from server" : `Request failed (${response.status})`); }
    return {response, data};
  }
  finally { clearTimeout(timer); pendingRequests.delete(controller); }
}
const editedTiming = new Set();
for (const id of ['initial', 'refresh']) $(id).addEventListener('input', () => editedTiming.add(id));
const hotwordInputs = ['hotword-enabled', 'hotword-words'];
function syncHotwordControls() {
  const busy = $('timing').disabled;
  for (const id of hotwordInputs) $(id).disabled = busy;
  $('hotword-open').textContent = $('hotword-enabled').checked ? 'Hotwords: on' : 'Hotwords: off';
}
new MutationObserver(syncHotwordControls).observe($('timing'), {attributes:true, attributeFilter:['disabled']});
$('hotword-open').onclick = () => { syncHotwordControls(); $('hotword-dialog').showModal(); };
$('hotword-close').onclick = () => $('hotword-dialog').close();
$('hotword-enabled').onchange = () => { hotwordWordsEdited = true; syncHotwordControls(); };
let hotwordWordsEdited = false;
$('hotword-words').addEventListener('input', () => { hotwordWordsEdited = true; });
function applyHotwordPreset(sample) {
  const words = sample?.hotwords || [];
  $('hotword-words').value = words.join('\n');
  $('hotword-enabled').checked = false;
  hotwordWordsEdited = false;
  sessionPreviewPolicy = null;
  syncHotwordControls();
}
function readHotwords() {
  const enabled = $('hotword-enabled').checked;
  if (!enabled) return {enabled:false};
  const words = [...new Set($('hotword-words').value.split(/[,，\n]/u).map(x => x.trim()).filter(Boolean))];
  if (!words.length || words.length > 128 || words.some(w => new TextEncoder().encode(w).length > 256)) {
    throw Error('Enter 1–128 hotwords, each at most 256 UTF-8 bytes.');
  }
  return {enabled, words};
}
function readPreviewPolicy() {
  const policy = {};
  for (const [id, key] of [['initial', 'initial_s'], ['refresh', 'refresh_s']]) {
    const value = Number($(id).value);
    if (!Number.isFinite(value) || value < .32 || value > 24) {
      $(id).reportValidity();
      throw Error('Transcription timing must be between 0.32 and 24 seconds.');
    }
    policy[key] = value;
  }
  policy.hotwords = readHotwords();
  return policy;
}
const flushes = new Map();
const timeLabel = s => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
function status(text) { $('status').textContent = text; }
function quietPause() { if (!player.paused) { ignorePause++; player.pause(); } }
async function quietPlay() { ignorePlay++; try { await player.play(); } catch (error) { ignorePlay--; throw error; } }
async function api(path, body) {
  const {response, data} = await request(path, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
  if (!response.ok) throw Error(data.error || `Request failed (${response.status})`);
  return data;
}
function fail(error) {
  clearTimeout(seekTimer); seekingSession = false;
  if (!failure) failure = {sid, message:error.message || String(error)};
  clearReconnect();
  for (const controller of pendingRequests) controller.abort();
  status(`Incomplete · ${failure.message}`);
  capturing = active = false;
  quietPause();
  if (captureNode?.port) captureNode.port.postMessage({epoch: ++epoch, active: false});
  $('start').disabled = !selectedMedia || modelStatus !== 'ready';
  $('stop').disabled = true;
  $('timing').disabled = false;
}
function updateModel(state) {
  const previous = modelStatus;
  modelStatus = state.model_status || 'ready';
  $('reload').hidden = modelStatus === 'ready';
  $('reload').disabled = modelStatus === 'loading';
  if (modelStatus !== 'loading') { clearTimeout(reloadTimer); reloadTimer = null; }
  else if (!reloadTimer) reloadTimer = setTimeout(() => {
    reloadTimer = null;
    if (modelStatus === 'loading') updateModel({model_status:'unknown'});
  }, 45000);
  if (modelStatus !== 'ready') {
    modelRecovered = false;
    $('start').disabled = true;
    if (modelStatus === 'failed' && (previous !== 'failed' || !failure)) fail(Error(state.model_error || 'Model inference failed. Reload the model to retry.'));
    if (modelStatus === 'loading') status('Reloading model…');
    if (modelStatus === 'unknown') status('Model reload status unavailable. Restore the connection, then retry reload.');
  } else if (previous === 'loading' || previous === 'unknown') {
    modelRecovered = true;
    $('start').disabled = !selectedMedia;
    status('Model reloaded. Start a new transcription to retry.');
  }
}
$('reload').onclick = () => exclusive(async () => {
  $('reload').disabled = true;
  try { updateModel(await api('/api/model/reload', {})); }
  catch (error) { $('reload').disabled = false; throw error; }
});
function exclusive(task) {
  operations = operations.then(task).catch(fail);
  return operations;
}
function enqueue(buffer, generation = epoch) {
  if (!sid || generation !== epoch) return;
  const session = sid, sequence = seq++;
  const trace = new URLSearchParams(location.search).has('trace') ? `&media_t=${player.currentTime.toFixed(6)}&context_rate=${audioContext.sampleRate}&input_rate=${inputRate}&context_t=${audioContext.currentTime.toFixed(6)}&play_elapsed=${((performance.now() - playBeganAt) / 1000).toFixed(6)}` : '';
  pipeline = pipeline.then(async () => {
    const deadline = performance.now() + 30000;
    for (let attempt = 0; ; attempt++) {
      if (performance.now() >= deadline) throw Error("Audio backlog did not recover within 30 seconds");
      if (session !== sid || generation !== epoch) return;
      let response, result;
      try {
        ({response, data:result} = await request(`/api/audio?sid=${encodeURIComponent(session)}&seq=${sequence}${trace}`, {method: 'POST', body: buffer}));
      } catch (error) {
        if (attempt >= 3) throw error;
        await sleep(200); continue;
      }
      if (session !== sid || generation !== epoch) return;
      if (response.status === 429) { status('Device is catching up; keeping queued audio…'); await sleep(250); continue; }
      if (!response.ok) throw Error(result.error || 'Audio transport failed');
      return;
    }
  });
  // Observe errors immediately while retaining rejection for drain/control callers.
  pipeline.catch(error => { if (session === sid && generation === epoch) fail(error); });
}
async function setupAudio() {
  if (audioContext) return audioContext.resume();
  audioContext = new AudioContext();
  source = audioContext.createMediaElementSource(player);
  source.connect(audioContext.destination);
  const sink = audioContext.createGain(); sink.gain.value = 0; sink.connect(audioContext.destination);
  if (audioContext.audioWorklet && window.isSecureContext) {
    await audioContext.audioWorklet.addModule('/static/pcm-worklet.js');
    captureNode = new AudioWorkletNode(audioContext, 'pcm16-worklet', {processorOptions: {sourceSampleRate: audioContext.sampleRate, chunkSamples: 1600}});
    captureNode.port.onmessage = event => {
      if (event.data.flushed !== undefined) { const resolve = flushes.get(event.data.flushed); if (resolve) resolve(); return; }
      enqueue(event.data.buffer, event.data.epoch);
    };
  } else {
    captureNode = audioContext.createScriptProcessor(1024, 2, 1);
    fallback = new Pcm16Resampler(audioContext.sampleRate, 16000, 1600, buffer => enqueue(buffer));
    captureNode.onaudioprocess = event => {
      if (!capturing || player.paused || player.seeking ||
          player.readyState < HTMLMediaElement.HAVE_FUTURE_DATA) return;
      // Multiple audio blocks can arrive at the same reported media time.
      const input = event.inputBuffer; inputRate = input.sampleRate;
      fallback.push(Array.from({length: input.numberOfChannels}, (_, i) => input.getChannelData(i)));
    };
  }
  source.connect(captureNode); captureNode.connect(sink);
  await audioContext.resume();
}
function beginCapture() {
  if (playBeganAt === null) playBeganAt = performance.now();
  if (fallback) fallback.reset();
  else captureNode.port.postMessage({epoch, active: true});
  capturing = true;
}
async function drainCapture() {
  capturing = false;
  if (fallback) fallback.flush();
  else {
    const id = ++flushNumber;
    await new Promise((resolve, reject) => {
      const timer = setTimeout(() => { flushes.delete(id); reject(Error('Audio capture did not flush')); }, 3000);
      flushes.set(id, () => { clearTimeout(timer); flushes.delete(id); resolve(); });
      captureNode.port.postMessage({command: 'flush', id});
    });
  }
  await pipeline;
}
async function newSession(play = true, policy = sessionPreviewPolicy) {
  quietPause();
  capturing = false;
  ++epoch;
  if (captureNode && captureNode.port) captureNode.port.postMessage({epoch, active: false});
  playBeganAt = null; firstVisibleS = null;
  const result = await api('/api/session', {offset_s: player.currentTime, ...policy});
  sessionPreviewPolicy = {...result.preview_policy, hotwords:result.hotwords};
  failure = null; modelRecovered = false; clearReconnect();
  sid = result.sid; seq = 0; pipeline = Promise.resolve(); active = true;
  $('start').disabled = true; $('stop').disabled = false;
  $('timing').disabled = true;
  render(result);
  if (play) { await quietPlay(); beginCapture(); }
  else { await api('/api/control', {sid, action: 'pause'}); }
  status(play ? 'Playing · listening for speech…' : 'Paused at the new position.');
}
function chooseMedia(file = null) {
  clearTimeout(seekTimer); seekingSession = false;
  return exclusive(async () => {
  quietPause();
  if (active) { await drainCapture(); await api('/api/control', {sid, action: 'stop'}); }
  active = capturing = false; sid = null; failure = null; ++epoch;
  $('timing').disabled = false;
  if (objectURL) URL.revokeObjectURL(objectURL);
  objectURL = file ? URL.createObjectURL(file) : null;
  const sample = mediaCatalog.find(item => item.id === $('sample').value) || mediaCatalog[0];
  builtinMedia = sample ? sample.url : builtinMedia;
  selectedMedia = objectURL || builtinMedia;
  player.src = selectedMedia; $('start').disabled = modelStatus !== 'ready'; $('stop').disabled = true;
  if (!file) $('file').value = '';
  player.poster = file ? '' : (sample?.poster || '');
  applyHotwordPreset(file ? null : sample);
  $('media-title').textContent = file ? file.name : (sample?.title || 'Beyond the README · FOSDEM 2025');
  if (sample) {
    const credit = $('sample-credit');
    credit.replaceChildren(document.createTextNode(sample.credit + ' · '));
    for (const [label, url] of [['Source', sample.source], [sample.license, sample.license_url]]) {
      const link = document.createElement('a'); link.textContent = label; link.href = url;
      link.target = '_blank'; link.rel = 'noopener';
      credit.append(link, document.createTextNode(' · '));
    }
    credit.append(document.createTextNode('Trimmed & transcoded'));
  }
  $('media-title').title = $('media-title').textContent;
  $('sample-credit').hidden = !!file;
  $('transcript').replaceChildren(); lastState = null; status('Ready. Start from any position.');
});
}
$('file').onchange = () => chooseMedia($('file').files[0] || null);
$('demo').onclick = () => chooseMedia();
$('sample').onchange = () => chooseMedia();
fetch('/api/media').then(response => {
  if (!response.ok) throw Error('Could not load built-in media');
  return response.json();
}).then(items => {
  mediaCatalog = items;
  $('sample').replaceChildren(...items.map(item => {
    const option = document.createElement('option'); option.value = item.id;
    option.textContent = item.title; return option;
  }));
  $('sample').disabled = false;
  if (!active && !objectURL && !hotwordWordsEdited) {
    applyHotwordPreset(items.find(item => item.url === selectedMedia));
  }
}).catch(error => status(error.message));
$('help-open').onclick = () => $('help').showModal();
$('help-close').onclick = () => $('help').close();
player.src = builtinMedia;
$('start').disabled = false;
status('Demo ready. Start transcription to play the lecture.');
$('start').onclick = () => exclusive(async () => {
  const policy = readPreviewPolicy();
  $('start').disabled = true;
  $('timing').disabled = true;
  await setupAudio();
  if (player.ended) player.currentTime = 0;
  await newSession(true, policy);
});
async function stopPlayback() {
  if (!active) return;
  quietPause(); await drainCapture(); await api('/api/control', {sid, action: 'stop'});
  active = false; $('stop').disabled = true; $('start').disabled = false; $('timing').disabled = false; status('Finishing received speech…');
}
$('stop').onclick = () => {
  clearTimeout(seekTimer); seekingSession = false;
  return exclusive(stopPlayback);
};
document.querySelector('.mode-switch a[href="/batch"]').addEventListener('click', event => {
  if (event.button !== 0 || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
  event.preventDefault();
  const target = event.currentTarget.href;
  clearTimeout(seekTimer); seekingSession = false;
  exclusive(async () => {
    await stopPlayback();
    location.assign(target);
  });
});
player.onpause = () => {
  if (ignorePause) { ignorePause--; return; }
  if (!active || player.ended || player.seeking) return;
  capturing = false;
  exclusive(async () => { await drainCapture(); await api('/api/control', {sid, action: 'pause'}); status('Paused · confirming received speech…'); });
};
player.onplay = () => {
  if (ignorePlay) { ignorePlay--; return; }
  if (seekingSession) return;
  if (!active) return;
  quietPause();
  exclusive(async () => { await api('/api/control', {sid, action: 'resume'}); await quietPlay(); beginCapture(); status('Playing · transcribing…'); });
};
player.onseeking = () => {
  if (!active) return;
  if (!seekingSession) seekResume = !player.paused;
  seekingSession = true;
  quietPause();
  capturing = false;
  ++epoch;
  if (captureNode && captureNode.port) captureNode.port.postMessage({epoch, active: false});
  clearTimeout(seekTimer);
};
player.onseeked = () => {
  if (!active) return;
  clearTimeout(seekTimer);
  const resume = seekResume;
  quietPause();
  seekTimer = setTimeout(() => exclusive(async () => {
    seekingSession = false;
    await newSession(resume);
  }), 180);
};
player.onended = () => {
  if (!active) return;
  exclusive(async () => { await drainCapture(); await api('/api/control', {sid, action: 'end'}); active = false; status('Finishing the last segment…'); });
};
player.onwaiting = () => { if (captureNode?.port) captureNode.port.postMessage({active: false}); };
player.onplaying = () => { if (capturing && captureNode?.port) captureNode.port.postMessage({active: true}); };
player.onerror = () => status('This media could not be decoded. Try a browser-supported audio file or an H.264/AAC MP4.');
player.ontimeupdate = () => { $('position').textContent = timeLabel(player.currentTime); };
function render(state) {
  lastState = state;
  const panel = $('transcript'), nearBottom = panel.scrollHeight - panel.scrollTop - panel.clientHeight < 90;
  const fragment = document.createDocumentFragment();
  for (const row of state.segments || []) {
    const item = document.createElement('div'); item.className = 'segment';
    const stamp = document.createElement('button'); stamp.className = 'stamp'; stamp.textContent = `${timeLabel(row.start)}–${timeLabel(row.end)}`;
    stamp.onclick = () => { player.currentTime = row.start; };
    const text = document.createElement('p'); text.textContent = row.text; item.append(stamp, text);
    if (row.cap_hit) { const warning = document.createElement('div'); warning.className = 'warning'; warning.textContent = 'Generation limit reached · this segment may be incomplete'; item.append(warning); }
    fragment.append(item);
  }
  if (state.partial) {
    const item = document.createElement('div'); item.className = 'partial';
    const label = document.createElement('small'); label.textContent = 'Recognizing · text may change';
    const text = document.createElement('p'); text.textContent = state.partial.text; item.append(label, text); fragment.append(item);
  }
  if (!fragment.childNodes.length) { const empty = document.createElement('div'); empty.className = 'empty'; empty.textContent = 'Listening for speech…'; fragment.append(empty); }
  panel.replaceChildren(fragment); if (nearBottom) panel.scrollTop = panel.scrollHeight;
  if (firstVisibleS === null && playBeganAt !== null &&
      (state.partial?.text || (state.segments || []).some(row => row.text))) {
    firstVisibleS = (performance.now() - playBeganAt) / 1000;
  }
  $('first').textContent = firstVisibleS === null ? '—' : `${firstVisibleS.toFixed(2)} s`;
  $('lag').textContent = `${Math.max(0, state.pending_s || 0).toFixed(1)} s`;
  $('vad').textContent = `${Math.round((state.speech_probability || 0) * 100)}%`;
  $('calls').textContent = state.inferences;
  $('rtf').textContent = state.processed_s ? (state.compute_s / state.processed_s).toFixed(2) : '—';
  if (modelStatus === 'loading') status('Reloading model…');
  else if (modelStatus === 'unknown') status('Model reload status unavailable. Restore the connection, then retry reload.');
  else if (modelStatus === 'failed') status(`Incomplete · model unavailable: ${state.model_error || failure?.message || 'Reload the model to retry.'}`);
  else if (failure && failure.sid === sid) status(modelRecovered ? 'Model reloaded · previous transcript incomplete. Start a new transcription.' : `Incomplete · ${failure.message}`);
  else if (state.error) fail(Error(state.error));
  else if (['done', 'stopped'].includes(state.state)) { active = false; $('start').disabled = false; $('stop').disabled = true; $('timing').disabled = false; status(state.state === 'done' ? 'Complete · all received audio processed.' : 'Stopped · received speech confirmed.'); }
  else if (state.state === 'running' && active) status(state.partial ? 'Playing · updating temporary text…' : 'Playing · listening for speech…');
  else if (state.state === 'paused') status('Paused · received speech confirmed.');
}
const events = new EventSource('/api/events');
events.onmessage = event => { clearReconnect(); const state = JSON.parse(event.data); updateModel(state); if (state.sid === sid && sid) render(state); };
events.onerror = () => {
  if (((!active || failure) && modelStatus !== 'loading') || reconnectTimer) return;
  status('Connection interrupted · attempting to reconnect…');
  const session = sid;
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    if (modelStatus === 'loading') updateModel({model_status:'unknown'});
    else if (active && sid === session) fail(Error('Server connection did not recover within 15 seconds'));
  }, 15000);
};
const confirmedText = () => {
  const incomplete = failure || (lastState?.segments || []).some(row => row.cap_hit);
  const header = incomplete ? `INCOMPLETE TRANSCRIPT${failure ? ': ' + failure.message : ': a generation limit was reached'}\n\n` : '';
  return header + (lastState?.segments || []).map(row => `[${timeLabel(row.start)}] ${row.text}`).join('\n\n');
};
$('copy').onclick = async () => {
  const text = confirmedText();
  try { await navigator.clipboard.writeText(text); status('Confirmed transcript copied.'); }
  catch (_) { const input = document.createElement('textarea'); input.value = text; document.body.append(input); input.select(); document.execCommand('copy'); input.remove(); status('Confirmed transcript copied.'); }
};
$('export').onclick = () => { const url = URL.createObjectURL(new Blob([confirmedText()], {type: 'text/plain;charset=utf-8'})); const a = document.createElement('a'); a.href = url; a.download = 'transcript.txt'; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); };
fetch('/api/health').then(response => response.json()).then(data => {
  if (!active && !$('timing').disabled && data.preview_policy) {
    for (const [id, key] of [['initial', 'initial_s'], ['refresh', 'refresh_s']]) {
      if (!editedTiming.has(id)) $(id).value = data.preview_policy[key];
    }
  }
}).catch(error => status(error.message));
