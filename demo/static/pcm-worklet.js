// Audio capture of the demo page: the page's sound (any rate, any channel count) to 16 kHz mono PCM16, posted in fixed
// chunks (default 1600 samples = 100 ms).
//
// Pcm16Resampler: streaming polyphase low-pass FIR. Each output is centered at j * srcRate / dstRate.
// Filter history is retained across input blocks and channels are averaged before filtering.
// Used twice: by the AudioWorklet below and, where AudioWorklet is missing (plain http pages), by the page's
// ScriptProcessorNode fallback (index.html loads this file as a script too).
//
// Pcm16Worklet: the AudioWorklet around it, plus an epoch: port.postMessage({epoch}) restarts the resampler, and every
// chunk is posted as {epoch, buffer} so the page can drop audio captured before a start or a seek.
class Pcm16Resampler {
  constructor(srcRate, dstRate, chunkSamples, emit) {
    this.step = srcRate / dstRate;
    this.chunkSamples = chunkSamples;
    this.emit = emit;
    this.radius = srcRate === dstRate ? 0 : Math.ceil(16 * Math.max(1, this.step));
    if (this.radius) {
      let a = srcRate, b = dstRate;
      while (b) [a, b] = [b, a % b];
      this.phaseCount = Math.min(320, dstRate / a);
      const cutoff = 0.45 / Math.max(1, this.step);
      this.filters = Array.from({ length: this.phaseCount + 1 }, (_, phase) => {
        const taps = new Float64Array(2 * this.radius + 1);
        const fraction = phase / this.phaseCount;
        let sum = 0;
        for (let i = 0; i < taps.length; i++) {
          const x = i - this.radius - fraction;
          if (Math.abs(x) >= this.radius) continue;
          const angle = Math.PI * x / this.radius;
          const window = 0.42 + 0.5 * Math.cos(angle) + 0.08 * Math.cos(2 * angle);
          const sinc = x === 0 ? 2 * cutoff : Math.sin(2 * Math.PI * cutoff * x) / (Math.PI * x);
          taps[i] = window * sinc;
          sum += taps[i];
        }
        return taps.map(value => value / sum);
      });
    }
    this.reset();
  }

  reset() {
    this.j = 0;
    this.base = 0;
    this.first = 0;
    this.history = new Float64Array(0);
    this.chunk = new Int16Array(this.chunkSamples);
    this.fill = 0;
  }

  writeSample(sample) {
    const value = Math.max(-1, Math.min(1, sample));
    this.chunk[this.fill++] = value < 0 ? value * 0x8000 : value * 0x7fff;
    if (this.fill === this.chunkSamples) {
      const buffer = this.chunk.buffer;
      this.chunk = new Int16Array(this.chunkSamples);
      this.fill = 0;
      this.emit(buffer);
    }
  }

  drain(samples, start, end, flushing) {
    for (;;) {
      const pos = this.j * this.step;
      if (pos >= end) break;
      const center = Math.floor(pos);
      if (!flushing && center + this.radius >= end) break;
      const taps = this.filters[Math.round((pos - center) * this.phaseCount)];
      let value = 0;
      for (let i = 0; i < taps.length; i++) {
        const index = center - this.radius + i;
        // Extend the edges of the session while retaining the exact output duration.
        const input = index < 0 ? this.first : samples[Math.min(index, end - 1) - start];
        value += taps[i] * input;
      }
      this.writeSample(value);
      this.j++;
    }
  }

  flush() {
    if (this.radius) this.drain(this.history, this.base - this.history.length, this.base, true);
    if (this.fill) this.emit(this.chunk.slice(0, this.fill).buffer);
    this.chunk = new Int16Array(this.chunkSamples);
    this.fill = 0;
  }

  push(channels) {
    const n = channels[0].length;
    if (!n) return;
    const carry = this.history.length;
    const samples = new Float64Array(carry + n);
    samples.set(this.history);
    for (const channel of channels) {
      for (let i = 0; i < n; i++) samples[carry + i] += channel[i];
    }
    for (let i = carry; i < samples.length; i++) samples[i] /= channels.length;
    if (!this.radius) {
      for (const value of samples) this.writeSample(value);
      return;
    }
    if (!this.base) this.first = samples[0];
    const end = this.base + n;
    this.drain(samples, this.base - carry, end, false);
    // The next output center can be as early as end - radius.
    this.history = samples.slice(Math.max(0, samples.length - 2 * this.radius));
    this.base = end;
  }
}

if (typeof AudioWorkletProcessor !== "undefined") {
  class Pcm16Worklet extends AudioWorkletProcessor {
    constructor(options) {
      super();
      const o = options.processorOptions || {};
      this.epoch = 0;
      this.active = false;
      this.resampler = new Pcm16Resampler(o.sourceSampleRate || sampleRate, o.targetSampleRate || 16000,
        o.chunkSamples || 1600, (buf) => this.port.postMessage({ epoch: this.epoch, buffer: buf }, [buf]));
      this.port.onmessage = (e) => {
        if (e.data && e.data.epoch !== undefined) {
          this.epoch = e.data.epoch;
          this.resampler.reset();
          this.active = !!e.data.active;
        }
        if (e.data && e.data.active !== undefined) this.active = !!e.data.active;
        if (e.data && e.data.command === "flush") {
          this.active = false;
          this.resampler.flush();
          this.port.postMessage({flushed: e.data.id});
        }
      };
    }

    process(inputs) {
      const channels = inputs[0];
      if (this.active && channels && channels.length > 0 && channels[0].length > 0) this.resampler.push(channels);
      return true;
    }
  }
  registerProcessor("pcm16-worklet", Pcm16Worklet);
}
