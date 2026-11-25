'use strict';

const form = document.getElementById('batch-form');
const countInput = document.getElementById('stream-count');
const start = document.getElementById('start-all');
const statusLine = document.getElementById('batch-status');
const metrics = document.getElementById('batch-metrics');
const rtfNote = document.getElementById('rtf-note');
const empty = document.getElementById('batch-empty');
const results = document.getElementById('batch-results');
const sample = document.getElementById('sample');
const sampleError = document.getElementById('sample-error');
let running = false;

function link(text, href) {
  const anchor = document.createElement('a');
  anchor.textContent = text;
  anchor.href = href;
  return anchor;
}

function showSampleError() {
  sampleError.textContent = 'Sample audio is unavailable.';
  sampleError.hidden = false;
}

sample.addEventListener('error', showSampleError);
fetch('/api/media').then(response => {
  if (!response.ok) throw new Error('Media catalog unavailable');
  return response.json();
}).then(catalog => {
  const lecture = catalog.find(item => item.batch_audio);
  const clip = lecture.batch_audio;
  sample.src = clip.url;
  document.getElementById('sample-credit').append(
    link('Lorna Mitchell · FOSDEM 2025', lecture.source), ' · ',
    link(lecture.license, lecture.license_url),
    ' · Audio excerpt, trimmed and converted to mono WAV'
  );
}).catch(showSampleError);

async function readBatch(id) {
  const url = '/api/batch?job=' + encodeURIComponent(id);
  let networkFailures = 0;
  while (true) {
    await new Promise(resolve => setTimeout(resolve, 150));
    let response;
    try {
      response = await fetch(url);
      networkFailures = 0;
    } catch (error) {
      if (++networkFailures >= 3) throw error;
      continue;
    }
    if (response.status === 202) continue;
    const batch = await response.json();
    if (!response.ok) throw new Error(batch.error || 'Batch transcription failed. Try again.');
    return batch;
  }
}

countInput.addEventListener('invalid', () => {
  statusLine.textContent = 'Choose a whole number from 2 to 32.';
});

form.addEventListener('submit', async event => {
  event.preventDefault();
  if (running || !countInput.reportValidity()) return;
  const started = performance.now();
  const count = countInput.valueAsNumber;
  running = true;
  countInput.disabled = true;
  start.disabled = true;
  statusLine.textContent = 'Transcribing ' + count + ' streams…';
  metrics.hidden = true;
  rtfNote.hidden = true;
  empty.hidden = true;
  results.replaceChildren();

  try {
    const response = await fetch('/api/batch', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({count})
    });
    const job = await response.json();
    if (!response.ok) throw new Error(job.error || 'Batch transcription failed. Try again.');
    const batch = await readBatch(job.id);
    const waitSeconds = (performance.now() - started) / 1000;
    if (!(batch.audio_s > 0) || batch.results.length !== count ||
        batch.results.some((row, i) => row.stream !== i + 1)) {
      throw new Error('Incomplete batch response. Try again.');
    }
    const waitRTF = (waitSeconds / batch.audio_s).toFixed(3);
    const rows = document.createDocumentFragment();
    const template = document.getElementById('batch-result-template');
    for (const row of batch.results) {
      const card = template.content.cloneNode(true);
      card.querySelector('h3').textContent = 'Stream ' + row.stream;
      card.querySelector('.pill').textContent = row.finish_reason === 'length' ? 'Output limit reached' : 'Complete';
      card.querySelector('.batch-text').textContent = row.text || 'No speech recognized.';
      card.querySelector('.batch-wait').textContent = 'Waiting RTF ' + waitRTF;
      rows.append(card);
    }
    document.getElementById('sdk-time').textContent = batch.sdk_s.toFixed(3) + ' s';
    document.getElementById('throughput-rtf').textContent = batch.throughput_rtf.toFixed(3);
    document.getElementById('waiting-rtf').textContent = waitRTF;
    results.append(rows);
    metrics.hidden = false;
    rtfNote.hidden = false;
    statusLine.textContent = count + ' streams finished in ' + waitSeconds.toFixed(3) + ' s';
  } catch (error) {
    statusLine.textContent = error.message;
    empty.hidden = false;
  } finally {
    running = false;
    countInput.disabled = false;
    start.disabled = false;
  }
});
