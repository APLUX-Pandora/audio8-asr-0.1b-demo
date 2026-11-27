'use strict';
const $ = id => document.getElementById(id);
const player = $('player'), audioPlayer = $('audio-player');
const maxAudioBytes = 64 * 1024 * 1024, maxAudioSeconds = 600;
let mediaMode = 'video', selectedFile = null, awaitingVideoMetadata = false;
let upload = null, microphone = null, pendingMicrophone = null;
let mediaCatalog = [];
let builtinMedia = $('demo').dataset.src;
let selectedMedia = builtinMedia;
let audioContext, videoSource, captureSource, captureNode, fallback, objectURL, sid = null, epoch = 0;
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
function setStartReady() {
  $('start').disabled = modelStatus !== 'ready' || awaitingVideoMetadata;
}
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
  if (upload) upload.cancelled = true;
  quietPause();
  pendingMicrophone = null;
  releaseMicrophone();
  if (captureNode?.port) captureNode.port.postMessage({epoch: ++epoch, active: false});
  setStartReady();
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
    setStartReady();
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
  const trace = mediaMode === 'video' && new URLSearchParams(location.search).has('trace') ? `&media_t=${player.currentTime.toFixed(6)}&context_rate=${audioContext.sampleRate}&input_rate=${inputRate}&context_t=${audioContext.currentTime.toFixed(6)}&play_elapsed=${((performance.now() - playBeganAt) / 1000).toFixed(6)}` : '';
  pipeline = pipeline.then(async () => {
    const deadline = performance.now() + 30000;
    for (let attempt = 0; ; attempt++) {
      if (performance.now() >= deadline) throw Error("Audio backlog did not recover within 30 seconds");
      if (session !== sid || generation !== epoch || upload?.cancelled) return;
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
  pipeline.catch(error => { if (session === sid && generation === epoch && !upload?.cancelled) fail(error); });
  return pipeline;
}
async function setupAudio() {
  if (audioContext) return audioContext.resume();
  const context = new AudioContext();
  try {
    const sink = context.createGain();
    sink.gain.value = 0;
    sink.connect(context.destination);
    if (context.audioWorklet && window.isSecureContext) {
      await context.audioWorklet.addModule('/static/pcm-worklet.js');
      captureNode = new AudioWorkletNode(context, 'pcm16-worklet', {processorOptions: {sourceSampleRate: context.sampleRate, chunkSamples: 1600}});
      captureNode.port.onmessage = event => {
        if (event.data.flushed !== undefined) { const resolve = flushes.get(event.data.flushed); if (resolve) resolve(); return; }
        enqueue(event.data.buffer, event.data.epoch);
      };
    } else {
      captureNode = context.createScriptProcessor(1024, 2, 1);
      fallback = new Pcm16Resampler(context.sampleRate, 16000, 1600, buffer => enqueue(buffer));
      captureNode.onaudioprocess = event => {
        if (!capturing || (mediaMode === 'video' && (player.paused || player.seeking ||
            player.readyState < HTMLMediaElement.HAVE_FUTURE_DATA))) return;
        // Multiple audio blocks can arrive at the same reported media time.
        const input = event.inputBuffer; inputRate = input.sampleRate;
        fallback.push(Array.from({length: input.numberOfChannels}, (_, i) => input.getChannelData(i)));
      };
    }
    captureNode.connect(sink);
    await context.resume();
    audioContext = context;
  } catch (error) {
    captureNode = fallback = null;
    await context.close();
    throw error;
  }
}
function connectCapture(input) {
  if (captureSource === input) return;
  if (captureSource) captureSource.disconnect(captureNode);
  captureSource = input;
  if (input) input.connect(captureNode);
}
function stopTracks(stream) {
  for (const track of stream.getTracks()) track.stop();
}
function releaseMicrophone() {
  if (!microphone) return;
  const stream = microphone;
  microphone = null;
  connectCapture(null);
  stopTracks(stream);
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
async function openSession(offset, policy) {
  capturing = false;
  ++epoch;
  if (captureNode && captureNode.port) captureNode.port.postMessage({epoch, active: false});
  playBeganAt = null; firstVisibleS = null;
  const result = await api('/api/session', {offset_s: offset, ...policy});
  sessionPreviewPolicy = {...result.preview_policy, hotwords:result.hotwords};
  failure = null; modelRecovered = false; clearReconnect();
  sid = result.sid; seq = 0; pipeline = Promise.resolve(); active = true;
  $('start').disabled = true; $('stop').disabled = false;
  $('timing').disabled = true;
  render(result);
}
async function newSession(play = true, policy = sessionPreviewPolicy) {
  quietPause();
  await openSession(player.currentTime, policy);
  if (play) { await quietPlay(); beginCapture(); }
  else { await api('/api/control', {sid, action: 'pause'}); }
  status(play ? 'Playing · listening for speech…' : 'Paused at the new position.');
}
function showMedia(mode) {
  mediaMode = mode;
  player.hidden = mode !== 'video';
  $('audio-media').hidden = mode !== 'audio';
  $('mic-media').hidden = mode !== 'mic';
  $('sample-picker').hidden = mode !== 'video';
  $('microphone').setAttribute('aria-pressed', String(mode === 'mic'));
  $('start').textContent = mode === 'audio' ? 'Upload and transcribe' : 'Start transcription';
}
function cancelUpload() {
  if (upload) upload.cancelled = true;
}
function chooseMedia(mode = 'video', file = null) {
  pendingMicrophone = null;
  cancelUpload();
  clearTimeout(seekTimer); seekingSession = false;
  return exclusive(async () => {
    await stopPlayback();
    quietPause();
    audioPlayer.pause();
    active = capturing = false; sid = null; failure = null; ++epoch;
    $('timing').disabled = false;
    if (objectURL) URL.revokeObjectURL(objectURL);
    selectedFile = file;
    objectURL = file ? URL.createObjectURL(file) : null;
    const sample = mediaCatalog.find(item => item.id === $('sample').value) || mediaCatalog[0];
    builtinMedia = sample ? sample.url : builtinMedia;
    selectedMedia = objectURL || builtinMedia;
    awaitingVideoMetadata = mode === 'video' && !!file;
    showMedia(mode);
    for (const [element, kind] of [[player, 'video'], [audioPlayer, 'audio']]) {
      if (mode === kind) element.src = selectedMedia;
      else if (element.hasAttribute('src')) { element.removeAttribute('src'); element.load(); }
    }
    setStartReady(); $('stop').disabled = true;
    player.poster = file ? '' : (sample?.poster || '');
    applyHotwordPreset(file || mode === 'mic' ? null : sample);
    $('media-title').textContent = file?.name || (mode === 'mic' ? 'Microphone' : (sample?.title || 'Beyond the README · FOSDEM 2025'));
    $('media-title').title = $('media-title').textContent;
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
    $('sample-credit').hidden = !!file || mode === 'mic';
    $('transcript').replaceChildren(); lastState = null;
    $('position').textContent = '00:00';
    status(mode === 'audio' ? 'Listen to the file or upload it for transcription.' : 'Ready to transcribe.');
  });
}
$('file').onchange = () => {
  const input = $('file'), file = input.files[0];
  input.value = '';
  if (!file) return;
  const isAudio = file.type.startsWith('audio/') ||
    (!file.type.startsWith('video/') && /\.(aac|flac|m4a|mp3|oga|ogg|opus|wav|wave)$/i.test(file.name));
  chooseMedia(isAudio ? 'audio' : 'video', file);
};
$('demo').onclick = () => chooseMedia();
$('sample').onchange = () => chooseMedia();
$('microphone').onclick = () => chooseMedia('mic');
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
  if (!active && mediaMode === 'video' && !objectURL && !hotwordWordsEdited) {
    applyHotwordPreset(items.find(item => item.url === selectedMedia));
  }
}).catch(error => status(error.message));
$('help-open').onclick = () => $('help').showModal();
$('help-close').onclick = () => $('help').close();
player.src = builtinMedia;
$('start').disabled = false;
status('Demo ready. Start transcription to play the lecture.');
async function transcribeAudio(policy) {
  const file = selectedFile;
  if (!file.size) throw Error('The selected audio file is empty.');
  const previewDuration = audioPlayer.duration;
  if (file.size > maxAudioBytes || (Number.isFinite(previewDuration) && previewDuration > maxAudioSeconds)) {
    throw Error('Choose an audio file up to 64 MiB and 10 minutes long.');
  }
  const job = {cancelled: false};
  upload = job;
  sid = null; failure = null; active = false;
  $('stop').disabled = false;
  status('Decoding audio…');
  try {
    let audio;
    try {
      audio = await new OfflineAudioContext(1, 1, 16000).decodeAudioData(await file.arrayBuffer());
    } catch {
      throw Error('This audio could not be decoded. Choose a supported audio file and retry.');
    }
    if (job.cancelled) return;
    if (audio.duration > maxAudioSeconds) throw Error('Choose an audio file up to 10 minutes long.');
    if (audio.length < 201) throw Error('The selected audio is too short to transcribe.');
    await openSession(0, policy);
    playBeganAt = performance.now();
    status('Uploading audio for transcription…');
    const channels = Array.from({length: audio.numberOfChannels}, (_, index) => audio.getChannelData(index));
    const resampler = new Pcm16Resampler(16000, 16000, 16000, buffer => enqueue(buffer));
    for (let start = 0; start < audio.length && !job.cancelled; start += 16000) {
      resampler.push(channels.map(channel => channel.subarray(start, start + 16000)));
      await pipeline;
    }
    if (job.cancelled) return;
    resampler.flush();
    await pipeline;
    if (!job.cancelled) await stopPlayback('end');
  } finally {
    upload = null;
    if (job.cancelled && !sid) {
      setStartReady();
      $('stop').disabled = true; $('timing').disabled = false;
      status('Upload stopped. Start transcription to retry.');
    }
  }
}
function requestMicrophone(policy) {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw Error('Microphone access requires HTTPS or localhost.');
  }
  const attempt = {};
  pendingMicrophone = attempt;
  status('Waiting for microphone permission…');
  // Leave the operation queue free while the browser waits for permission.
  navigator.mediaDevices.getUserMedia({audio: true}).then(stream => {
    if (pendingMicrophone !== attempt) {
      stopTracks(stream);
      return;
    }
    exclusive(async () => {
      if (pendingMicrophone !== attempt) {
        stopTracks(stream);
        return;
      }
      pendingMicrophone = null;
      await startMicrophone(stream, policy);
    });
  }, () => {
    if (pendingMicrophone === attempt) {
      fail(Error('Could not access the microphone. Check browser permissions and retry.'));
    }
  });
}
async function startMicrophone(stream, policy) {
  microphone = stream;
  await setupAudio();
  if (microphone !== stream) return;
  for (const track of stream.getAudioTracks()) {
    track.addEventListener('ended', () => exclusive(async () => {
      if (microphone === stream) await stopPlayback();
    }), {once: true});
  }
  connectCapture(audioContext.createMediaStreamSource(stream));
  await openSession(0, policy);
  if (microphone !== stream) {
    await stopPlayback();
    return;
  }
  beginCapture();
  status('Listening to the microphone…');
}
$('start').onclick = () => exclusive(async () => {
  const policy = readPreviewPolicy();
  $('start').disabled = true;
  $('timing').disabled = true;
  if (mediaMode === 'audio') return transcribeAudio(policy);
  if (mediaMode === 'mic') return requestMicrophone(policy);
  await setupAudio();
  if (!videoSource) {
    videoSource = audioContext.createMediaElementSource(player);
    videoSource.connect(audioContext.destination);
  }
  connectCapture(videoSource);
  if (player.ended) player.currentTime = 0;
  await newSession(true, policy);
});
async function stopPlayback(action = 'stop') {
  if (!active) return;
  try {
    if (mediaMode === 'video') quietPause();
    if (mediaMode !== 'audio') await drainCapture();
    await api('/api/control', {sid, action});
  } finally {
    releaseMicrophone();
  }
  active = false;
  $('stop').disabled = true;
  if (action === 'stop') { setStartReady(); $('timing').disabled = false; }
  status('Finishing received speech…');
}
$('stop').onclick = () => {
  cancelUpload();
  clearTimeout(seekTimer); seekingSession = false;
  return exclusive(() => stopPlayback());
};
document.querySelector('.mode-switch a[href="/batch"]').addEventListener('click', event => {
  if (event.button !== 0 || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
  event.preventDefault();
  pendingMicrophone = null;
  cancelUpload();
  const target = event.currentTarget.href;
  clearTimeout(seekTimer); seekingSession = false;
  exclusive(async () => {
    await stopPlayback();
    location.assign(target);
  });
});
player.onpause = () => {
  if (ignorePause) { ignorePause--; return; }
  if (mediaMode !== 'video' || !active || player.ended || player.seeking) return;
  capturing = false;
  exclusive(async () => { await drainCapture(); await api('/api/control', {sid, action: 'pause'}); status('Paused · confirming received speech…'); });
};
player.onplay = () => {
  if (ignorePlay) { ignorePlay--; return; }
  if (mediaMode !== 'video' || seekingSession || !active) return;
  quietPause();
  exclusive(async () => { await api('/api/control', {sid, action: 'resume'}); await quietPlay(); beginCapture(); status('Playing · transcribing…'); });
};
player.onseeking = () => {
  if (mediaMode !== 'video' || !active) return;
  if (!seekingSession) seekResume = !player.paused;
  seekingSession = true;
  quietPause();
  capturing = false;
  ++epoch;
  if (captureNode && captureNode.port) captureNode.port.postMessage({epoch, active: false});
  clearTimeout(seekTimer);
};
player.onseeked = () => {
  if (mediaMode !== 'video' || !active || !seekingSession) return;
  clearTimeout(seekTimer);
  const resume = seekResume;
  quietPause();
  seekTimer = setTimeout(() => exclusive(async () => {
    seekingSession = false;
    await newSession(resume);
  }), 180);
};
player.onended = () => {
  if (mediaMode === 'video' && active) exclusive(() => stopPlayback('end'));
};
player.onwaiting = () => { if (mediaMode === 'video' && captureNode?.port) captureNode.port.postMessage({active: false}); };
player.onplaying = () => { if (mediaMode === 'video' && capturing && captureNode?.port) captureNode.port.postMessage({active: true}); };
player.onloadedmetadata = () => {
  if (mediaMode !== 'video' || !awaitingVideoMetadata) return;
  awaitingVideoMetadata = false;
  if (!player.videoWidth) {
    showMedia('audio');
    audioPlayer.src = selectedMedia;
    player.removeAttribute('src');
    player.load();
    status('Listen to the file or upload it for transcription.');
  }
  setStartReady();
};
player.onerror = () => {
  if (mediaMode !== 'video') return;
  if (active) fail(Error('This video could not be decoded. Try an H.264/AAC MP4.'));
  else status('This file could not be decoded. Choose a supported video or audio file and retry.');
};
audioPlayer.onerror = () => {
  if (mediaMode === 'audio' && !upload && !failure) status('This audio could not be played. Choose a supported audio file and retry.');
};
for (const [element, mode] of [[player, 'video'], [audioPlayer, 'audio']]) {
  element.ontimeupdate = () => {
    if (mediaMode === mode) $('position').textContent = timeLabel(element.currentTime);
  };
}
window.addEventListener('pagehide', () => {
  if (mediaMode !== 'mic') return;
  pendingMicrophone = null;
  if (sid) {
    const body = JSON.stringify({sid, action: 'stop'});
    if (!navigator.sendBeacon?.('/api/control', body)) {
      fetch('/api/control', {method: 'POST', body, keepalive: true}).catch(() => {});
    }
  }
  active = capturing = false;
  ++epoch;
  releaseMicrophone();
  setStartReady();
  $('stop').disabled = true; $('timing').disabled = false;
});
function render(state) {
  lastState = state;
  const panel = $('transcript'), nearBottom = panel.scrollHeight - panel.scrollTop - panel.clientHeight < 90;
  const fragment = document.createDocumentFragment();
  for (const row of state.segments || []) {
    const item = document.createElement('div'); item.className = 'segment';
    const stamp = document.createElement('button'); stamp.className = 'stamp'; stamp.textContent = `${timeLabel(row.start)}–${timeLabel(row.end)}`;
    stamp.disabled = mediaMode === 'mic';
    stamp.onclick = () => { (mediaMode === 'audio' ? audioPlayer : player).currentTime = row.start; };
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
  if (mediaMode === 'mic') $('position').textContent = timeLabel(state.received_s);
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
  else if (['done', 'stopped'].includes(state.state)) {
    active = false;
    if (pendingMicrophone) return;
    setStartReady(); $('stop').disabled = true; $('timing').disabled = false;
    status(state.state === 'done' ? 'Complete · all received audio processed.' : 'Stopped · received speech confirmed.');
  }
  else if (state.state === 'running' && active) {
    if (mediaMode === 'audio') status('Uploading audio for transcription…');
    else if (mediaMode === 'mic') status('Listening to the microphone…');
    else status(state.partial ? 'Playing · updating temporary text…' : 'Playing · listening for speech…');
  }
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
