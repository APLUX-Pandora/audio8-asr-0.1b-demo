#!/usr/bin/env python3
"""Audio/video playback transport and application segmentation over the native SDK."""
import argparse
import json
import math
import re
import secrets
import threading
import time
import wave
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
from inference import recognize
from lifecycle import run_owned
from pandora_audio8_asr_0_1b._package import ModelPackageError, resolve_model_dir
from pandora_audio8_asr_0_1b._types import ASRInferenceError, ASRStateError
from pandora_audio8_asr_0_1b._web import load_model
from streaming import Segmenter, SileroVad, trim_overlap

ROOT = Path(__file__).resolve().parents[1]


def preview_policy(initial_s, refresh_s):
    values = {'initial_s': initial_s, 'refresh_s': refresh_s}
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not .32 <= value <= 24:
            raise ValueError(f'{name} must be a number between 0.32 and 24 seconds')
    return values


def hotword_options(value=None):
    defaults = dict(enabled=False, words=[])
    if value is None:
        return defaults
    if not isinstance(value, dict) or set(value) - set(defaults):
        raise ValueError('Invalid hotword configuration')
    options = dict(defaults, **value)
    if not isinstance(options['enabled'], bool):
        raise ValueError('Hotword enabled must be boolean')
    words = options['words']
    if not isinstance(words, list) or len(words) > 128 or any(
            not isinstance(w, str) or '\0' in w or len(w.encode('utf-8')) > 256 for w in words):
        raise ValueError('Use at most 128 hotwords, each at most 256 UTF-8 bytes')
    if options['enabled'] and not any(w.strip() for w in words):
        raise ValueError('Enter at least one hotword')
    return options


class Playback:
    def __init__(self, engine, vad, initial_s=1.5, refresh_s=.75, *, model_factory=None):
        self.engine, self.vad = engine, vad
        self.model_factory = model_factory
        self.policy_options = preview_policy(initial_s, refresh_s)
        self.cv = threading.Condition()
        self.engine_lock = threading.Lock()
        self.batch_active = False
        self.batches = {}
        self.jobs = deque()
        self.sid = None
        self.request_hotwords = None
        self.version = 0
        self.closed = False
        self.state = {'state': 'ready', 'model_status': 'ready', 'model_error': None, 'segments': [], 'partial': None, 'received_s': 0.,
                      'processed_s': 0., 'offset_s': 0., 'pending_s': 0., 'inferences': 0,
                      'compute_s': 0., 'first_text_s': None, 'speech_probability': 0., 'error': None,
                      'preview_policy': dict(self.policy_options)}
        self.thread = threading.Thread(target=self.worker, daemon=False)
        self.thread.start()

    def changed(self):
        self.version += 1
        self.cv.notify_all()

    def snapshot(self):
        return dict(self.state, sid=self.sid, revision=self.version)

    def start(self, offset, overrides=None, hotwords=None):
        hotwords = hotword_options(hotwords)
        options = dict(self.policy_options, **(overrides or {}))
        options = preview_policy(**options)
        with self.cv:
            if self.closed:
                raise ValueError('Service is closing')
            if self.state['model_status'] != 'ready':
                raise ValueError('Reload the model before starting another session')
            if self.batch_active:
                raise ValueError('Wait for batch transcription to finish')
            self.state.update(model_status='ready', model_error=None)
            self.sid = secrets.token_urlsafe(18)
            self.seq = 0
            self.accepting = True
            self.started = time.monotonic()
            self.jobs.clear()
            self.state.update(state='running', segments=[], partial=None, received_s=0., processed_s=0.,
                              pending_s=0., offset_s=offset, inferences=0, compute_s=0., first_text_s=None, error=None,
                              preview_policy=dict(options), hotwords=dict(hotwords))
            self.jobs.append((self.sid, 'start', (options, hotwords)))
            self.changed()
            return self.snapshot()

    def model_failed(self, error):
        with self.cv:
            self.state.update(state='error', error=str(error), model_error=str(error), model_status='failed')
            self.jobs.clear()
            self.accepting = False
            self.changed()

    def reload_model(self):
        with self.cv:
            if self.closed or self.model_factory is None:
                return 409, {'error': 'Model reload is unavailable'}
            if self.state['model_status'] == 'loading' or self.state['state'] in ('running', 'paused'):
                return 409, {'error': 'Stop the active session before reloading the model'}
            if self.batch_active or not self.engine_lock.acquire(blocking=False):
                return 409, {'error': 'Inference is still running'}
            self.engine_lock.release()
            self.sid = None
            self.accepting = False
            self.jobs.clear()
            self.state.update(model_status='loading', state='loading')
            self.jobs.append((self.sid, 'reload', None))
            self.changed()
            return 202, self.snapshot()

    def audio(self, sid, seq, pcm):
        with self.cv:
            if self.sid is None or sid != self.sid or not self.accepting:
                return 409, {'error': 'This playback session is no longer accepting audio'}
            if seq < 0:
                return 400, {'error': 'Invalid audio sequence'}
            if seq < self.seq:
                return 200, {'accepted': True, 'duplicate': True, 'next_seq': self.seq}
            if seq != self.seq:
                return 409, {'error': 'Audio sequence gap', 'next_seq': self.seq}
            if self.state['pending_s'] >= 30:
                return 429, {'error': 'Audio backlog; retry this same chunk'}
            self.jobs.append((sid, 'audio', pcm.copy()))
            self.seq += 1
            self.state['received_s'] += len(pcm) / 16000
            self.state['pending_s'] += len(pcm) / 16000
            self.cv.notify_all()
            return 200, {'accepted': True, 'next_seq': self.seq}

    def control(self, sid, action):
        with self.cv:
            if self.sid is None or sid != self.sid:
                return 409, {'error': 'Stale playback session'}
            if not self.accepting:
                return 409, {'error': 'Session has ended'}
            if action in ('end', 'stop'):
                self.accepting = False
            self.jobs.append((sid, action, None))
            self.cv.notify_all()
            return 200, {'accepted': True}

    def recognize(self, sid, request):
        with self.cv:
            if sid != self.sid:
                return
        x = request.samples
        if len(x) < 201:
            x = np.pad(x, (0, 201 - len(x)))
        assert len(x) <= 480000
        t = time.monotonic()
        with self.engine_lock:
            result = recognize(self.engine, x, hotwords=self.request_hotwords, max_new_tokens=256)
        elapsed = time.monotonic() - t
        text = result.text.strip()
        ids = result.token_ids
        with self.cv:
            if sid != self.sid or self.closed or self.state['model_status'] != 'ready':
                return
            previous = self.state['segments'][-1]['text'] if self.state['segments'] else ''
            removed = 0
            if request.overlap:
                text, removed = trim_overlap(previous, text)
            item = {'start': self.state['offset_s'] + request.start,
                    'end': self.state['offset_s'] + request.end, 'text': text,
                    'reason': request.reason, 'cap_hit': not result.is_complete,
                    'finish_reason': result.finish_reason,
                    'token_count': len(ids), 'compute_s': elapsed, 'overlap_trimmed_chars': removed}
            self.state['inferences'] += 1
            self.state['compute_s'] += elapsed
            if text and self.state['first_text_s'] is None:
                self.state['first_text_s'] = time.monotonic() - self.started
            if request.final:
                if text or item['cap_hit']:
                    self.state['segments'] = self.state['segments'] + [item]
                self.state['partial'] = None
            else:
                self.state['partial'] = item
            self.changed()

    def start_batch(self, count, pcm):
        with self.cv:
            status = self.state['model_status']
            if status == 'failed':
                return 503, {'error': 'Reload the model in Live transcription, then retry'}
            if self.closed or status != 'ready' or self.state['state'] in ('running', 'paused'):
                return 409, {'error': 'Stop playback or wait for the model to become ready'}
            if self.batch_active or self.engine_lock.locked():
                return 409, {'error': 'Inference is already running. Retry when it finishes'}
            job = secrets.token_urlsafe(18)
            self.batch_active = True
            # Retain recent results so a disconnect while polling can recover.
            if len(self.batches) == 16:
                self.batches.pop(next(iter(self.batches)))
            self.batches[job] = (202, {'status': 'running'})
            self.jobs.append((self.sid, 'batch', (job, count, pcm)))
            self.cv.notify_all()
            return 202, {'id': job}

    def run_batch(self, job, count, pcm):
        code, response = 500, {'error': 'Batch transcription failed. Try again'}
        try:
            with self.engine_lock:
                audio_items = [pcm.copy() for _ in range(count)]
                started = time.monotonic()
                results = self.engine.transcribe_batch(audio_items, sample_rate=16000, max_new_tokens=256)
                sdk_s = time.monotonic() - started
            if len(results) != count:
                raise RuntimeError('Unexpected SDK batch result count')
            audio_s = pcm.size / 16000
            response = {
                'audio_s': audio_s, 'sdk_s': sdk_s, 'throughput_rtf': sdk_s / (count * audio_s),
                'results': [{'stream': i, 'text': result.text, 'finish_reason': result.finish_reason}
                            for i, result in enumerate(results, 1)],
            }
            code = 200
            print(f'SDK batch: inputs={count} results={len(results)} sdk_s={sdk_s:.6f} audio_s={audio_s:.6f}',
                  flush=True)
        except (ASRInferenceError, ASRStateError) as error:
            self.model_failed(error)
            print(f'Batch inference failed: {error}', flush=True)
            code, response = 503, {'error': 'Reload the model in Live transcription, then retry'}
        except (ValueError, RuntimeError) as error:
            print(f'Batch transcription failed: {error}', flush=True)
        finally:
            with self.cv:
                self.batches[job] = (code, response)
                self.batch_active = False

    def worker(self):
        segmenter = None
        while True:
            with self.cv:
                self.cv.wait_for(lambda: self.jobs or self.closed)
                if self.closed:
                    return
                sid, action, data = self.jobs.popleft()
                if sid != self.sid:
                    continue
            try:
                if action == 'batch':
                    self.run_batch(*data)
                elif action == 'reload':
                    with self.engine_lock:
                        if self.engine is not None:
                            self.engine.close()
                            self.engine = None
                        self.engine = self.model_factory()
                    segmenter = None
                    with self.cv:
                        self.state.update(state='ready', error=None, model_error=None, model_status='ready')
                        self.changed()
                elif action == 'start':
                    options, hotwords = data
                    self.request_hotwords = hotwords
                    segmenter = Segmenter(self.vad, **options)
                elif action == 'audio':
                    for request in segmenter.push(data):
                        self.recognize(sid, request)
                    with self.cv:
                        if sid == self.sid:
                            self.state['pending_s'] = max(0., self.state['pending_s'] - len(data) / 16000)
                            self.state['processed_s'] += len(data) / 16000
                            self.state['speech_probability'] = segmenter.probability
                            self.changed()
                elif action in ('pause', 'end', 'stop'):
                    for request in segmenter.flush(action):
                        self.recognize(sid, request)
                    with self.cv:
                        if sid == self.sid and self.state['model_status'] == 'ready':
                            self.state['state'] = {'pause': 'paused', 'end': 'done', 'stop': 'stopped'}[action]
                            self.changed()
                elif action == 'resume':
                    with self.cv:
                        if sid == self.sid and self.state['model_status'] == 'ready':
                            self.state['state'] = 'running'
                            self.changed()
            except Exception as error:
                with self.cv:
                    if sid == self.sid and self.state['model_status'] != 'failed':
                        self.model_failed(error)

    def close(self):
        with self.cv:
            self.closed = True
            self.accepting = False
            self.jobs.clear()
            self.cv.notify_all()
        self.thread.join()
        if self.model_factory is not None:
            with self.engine_lock:
                if self.engine is not None:
                    self.engine.close()
                    self.engine = None


MEDIA_DIR = ROOT / 'demo/media'
MEDIA_CATALOG = json.loads((MEDIA_DIR / 'catalog.json').read_text())
BATCH_SAMPLE = next(sample['batch_audio'] for sample in MEDIA_CATALOG if 'batch_audio' in sample)
BUILTIN_MEDIA = {sample['url']: 'video/mp4' for sample in MEDIA_CATALOG}
BUILTIN_MEDIA[BATCH_SAMPLE['url']] = 'audio/wav'


def load_batch_audio():
    # A missing sample leaves single-video playback available.
    path = MEDIA_DIR / Path(BATCH_SAMPLE['url']).name
    try:
        with wave.open(str(path), 'rb') as audio:
            expected_samples = round((BATCH_SAMPLE['end_s'] - BATCH_SAMPLE['start_s']) * 16000)
            layout = (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getnframes())
            if layout != (1, 2, 16000, expected_samples):
                raise ValueError('Expected the prepared mono 16 kHz PCM16 sample')
            pcm = np.frombuffer(audio.readframes(expected_samples), dtype='<i2').astype(np.float32) / 32768
            if pcm.size != expected_samples:
                raise ValueError('Incomplete batch audio sample')
            return pcm
    except (OSError, EOFError, ValueError, wave.Error) as error:
        print(f'Batch audio unavailable: {error}', flush=True)
        return None


class Handler(BaseHTTPRequestHandler):
    timeout = 10

    def media(self, head_only=False):
        route = urlsplit(self.path).path
        if route not in BUILTIN_MEDIA:
            self.send_error(404)
            return
        path = MEDIA_DIR / route.rsplit('/', 1)[-1]
        if not path.is_file():
            self.send_error(404, 'Built-in media is not installed')
            return
        size = path.stat().st_size
        start, end, code = 0, size - 1, 200
        requested = None if head_only else self.headers.get('Range')
        if requested:
            match = re.fullmatch(r'bytes=(\d*)-(\d*)', requested.strip())
            if match and (match[1] or match[2]):
                if match[1]:
                    start = int(match[1])
                    end = min(int(match[2]), size - 1) if match[2] else size - 1
                else:
                    start = max(0, size - int(match[2]))
                code = 206
            if not match or not (match[1] or match[2]) or start > end or start >= size:
                self.send_response(416)
                self.send_header('Content-Range', f'bytes */{size}')
                self.send_header('Content-Length', '0')
                self.end_headers()
                return
        self.send_response(code)
        self.send_header('Content-Type', BUILTIN_MEDIA[route])
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Length', str(end - start + 1))
        if code == 206:
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.end_headers()
        if head_only:
            return
        try:
            with path.open('rb') as stream:
                stream.seek(start)
                remaining = end - start + 1
                while remaining:
                    block = stream.read(min(65536, remaining))
                    if not block:
                        break
                    self.wfile.write(block)
                    remaining -= len(block)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass

    def do_HEAD(self):
        if urlsplit(self.path).path in BUILTIN_MEDIA:
            return self.media(head_only=True)
        self.send_error(404)

    def reply(self, code, obj):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def body(self, maximum):
        if self.headers.get('Transfer-Encoding') or len(self.headers.get_all('Content-Length', [])) > 1:
            self.close_connection = True
            raise ValueError('Ambiguous request body framing')
        size = int(self.headers.get('Content-Length', '0'))
        if size < 0 or size > maximum:
            self.close_connection = True
            raise ValueError('Request body exceeds its allowed size')
        data = self.rfile.read(size)
        if len(data) != size:
            raise ValueError('Incomplete request body')
        return data

    def do_GET(self):
        route = urlsplit(self.path).path
        app = self.server.app
        if route in BUILTIN_MEDIA:
            return self.media()
        if route == '/api/media':
            return self.reply(200, MEDIA_CATALOG)
        if route == '/api/batch':
            job = parse_qs(urlsplit(self.path).query).get('job', [''])[0]
            with app.cv:
                code, result = app.batches.get(job, (404, {'error': 'Batch result expired. Start a new batch'}))
            return self.reply(code, result)
        if route == '/api/health':
            with app.cv:
                model_status, model_error = app.state['model_status'], app.state['model_error']
            return self.reply(200, {'status': model_status, 'model_error': model_error, 'model_version': self.server.model_version,
                                  'sample_rate': 16000, 'max_samples': 480000, 'mode': 'playback',
                                  'preview_policy': app.policy_options})
        if route == '/api/state':
            with app.cv:
                return self.reply(200, app.snapshot())
        if route == '/api/events':
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            revision = -1
            try:
                while not app.closed:
                    with app.cv:
                        app.cv.wait_for(lambda: app.version != revision or app.closed, timeout=10)
                        state = app.snapshot()
                        revision = app.version
                    self.wfile.write(('data: ' + json.dumps(state, ensure_ascii=False) + '\n\n').encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass
            return
        if route == '/multi':
            self.send_response(302)
            self.send_header('Location', '/batch')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Content-Length', '0')
            self.end_headers()
            return
        files = {'/': ('index.html', 'text/html; charset=utf-8'),
                 '/batch': ('batch.html', 'text/html; charset=utf-8'),
                 '/static/batch.js': ('batch.js', 'text/javascript'),
                 '/static/style.css': ('style.css', 'text/css'),
                 '/static/app.js': ('app.js', 'text/javascript'),
                 '/static/pcm-worklet.js': ('pcm-worklet.js', 'text/javascript'),
                 '/static/logo.png': ('logo.png', 'image/png'),
                 '/static/fosdem-poster.jpg': ('fosdem-poster.jpg', 'image/jpeg'),
                 '/static/pandora-mark.svg': ('pandora-mark.svg', 'image/svg+xml')}
        if route not in files:
            return self.reply(404, {'error': 'Not found'})
        name, kind = files[route]
        data = (ROOT / 'demo/static' / name).read_bytes()
        self.send_response(200)
        self.send_header('Content-Type', kind)
        if route in ('/', '/batch', '/static/app.js', '/static/batch.js', '/static/style.css'):
            self.send_header('Cache-Control', 'no-cache')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        app = self.server.app
        url = urlsplit(self.path)
        try:
            path = url.path
            query = parse_qs(url.query)
            if path == '/api/model/reload':
                code, result = app.reload_model()
                return self.reply(code, result)
            if path == '/api/session':
                obj = json.loads(self.body(4096))
                if not isinstance(obj, dict):
                    raise ValueError('Expected a JSON object')
                offset = float(obj.get('offset_s', 0))
                if not math.isfinite(offset) or offset < 0:
                    raise ValueError('Invalid playback offset')
                options = {name: obj[name] for name in ('initial_s', 'refresh_s') if name in obj}
                result = app.start(offset, options, obj.get('hotwords'))
                return self.reply(200, result)
            if path == '/api/audio':
                data = self.body(32000)
                if not data or len(data) % 2:
                    raise ValueError('Expected 16 kHz mono PCM16, at most one second')
                pcm = np.frombuffer(data, dtype='<i2').astype(np.float32) / 32768
                code, result = app.audio(query.get('sid', [''])[0], int(query.get('seq', ['-1'])[0]), pcm)
                return self.reply(code, result)
            if path == '/api/control':
                obj = json.loads(self.body(4096))
                if not isinstance(obj, dict):
                    raise ValueError('Expected a JSON object')
                action = obj.get('action')
                if action not in ('pause', 'resume', 'end', 'stop'):
                    raise ValueError('Invalid playback action')
                code, result = app.control(obj.get('sid'), action)
                return self.reply(code, result)
            if path == '/api/batch':
                request = json.loads(self.body(128))
                count = request.get('count') if isinstance(request, dict) else None
                if type(count) is not int or not 2 <= count <= 32:
                    raise ValueError('Choose a whole number from 2 to 32')
                if self.server.batch_audio is None:
                    return self.reply(503, {'error': 'The sample audio is unavailable'})
                code, result = app.start_batch(count, self.server.batch_audio)
                return self.reply(code, result)
            if path == '/api/transcribe':
                data = self.body(1920000)
                if len(data) < 804 or len(data) % 4:
                    raise ValueError('Expected 201..480000 float32 samples')
                pcm = np.frombuffer(data, dtype='<f4')
                if not np.isfinite(pcm).all() or np.any(np.abs(pcm) > 1):
                    raise ValueError('Expected finite normalized audio in [-1, 1]')
                with app.cv:
                    busy = (app.closed or app.batch_active or app.state['model_status'] != 'ready'
                            or (app.sid and app.state['state'] not in ('done', 'stopped', 'error')))
                    if busy or not app.engine_lock.acquire(blocking=False):
                        return self.reply(409, {'error': 'Playback engine is busy'})
                try:
                    t = time.monotonic()
                    result = recognize(app.engine, pcm, max_new_tokens=256)
                    text = result.text
                    ids = result.token_ids
                except Exception as error:
                    app.model_failed(error)
                    raise
                finally:
                    app.engine_lock.release()
                return self.reply(200, {'text': text, 'audio_s': len(pcm) / 16000,
                                      'elapsed_s': time.monotonic() - t, 'token_count': len(ids),
                                      'cap_hit': not result.is_complete, 'finish_reason': result.finish_reason})
            self.reply(404, {'error': 'Not found'})
        except (ValueError, KeyError, TypeError) as error:
            self.reply(400, {'error': str(error)})
        except Exception as error:
            self.reply(500, {'error': str(error)})


class Server(ThreadingHTTPServer):
    daemon_threads = False
    block_on_close = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--models', help='Directory directly containing models.json; omitted: discover a unique package in models/')
    parser.add_argument('--library')
    parser.add_argument('--vad-model', help='Silero VAD file; omitted: demo/assets/silero_vad.onnx')
    parser.add_argument('--initial-s', type=float, default=1.5,
                        help='Segment duration before the first provisional decode')
    parser.add_argument('--refresh-s', type=float, default=.75,
                        help='Audio interval between subsequent provisional decodes')
    args = parser.parse_args()
    if any(not math.isfinite(v) or not .32 <= v <= 24 for v in (args.initial_s, args.refresh_s)):
        parser.error('--initial-s and --refresh-s must be finite and between 0.32 and 24 seconds')
    try:
        args.models = resolve_model_dir(args.models, default_container=ROOT / "models")
    except ModelPackageError as error:
        parser.error(str(error))
    print(f'Model package: {args.models}', flush=True)
    def serve(stopping):
        vad_path = args.vad_model or ROOT / 'demo/assets/silero_vad.onnx'
        vad = SileroVad(vad_path)
        batch_audio = load_batch_audio()
        with load_model(args.models, library=args.library) as engine:
            if stopping.is_set():
                return
            app = Playback(engine, vad, args.initial_s, args.refresh_s,
                           model_factory=lambda: load_model(args.models, library=args.library))
            server = None
            http = None
            try:
                server = Server((args.host, args.port), Handler)
                server.app = app
                server.batch_audio = batch_audio
                # A successful SDK load has already validated the Audio8 v3 package.
                server.model_version = 'v3'
                def accept():
                    try:
                        server.serve_forever(poll_interval=.1)
                    finally:
                        stopping.set()
                http = threading.Thread(target=accept, name='audio8-http', daemon=False)
                http.start()
                print(f'Listening on http://{args.host}:{args.port}', flush=True)
                stopping.wait()
            finally:
                if http is not None and http.ident is not None:
                    server.shutdown()
                    http.join()
                # Wake SSE and finish the playback worker before joining HTTP
                # handlers, including a one-shot request still using the SDK.
                app.close()
                if server is not None:
                    server.server_close()
        print(json.dumps({'model_closed': True}), flush=True)
    run_owned(serve)


if __name__ == '__main__':
    main()
