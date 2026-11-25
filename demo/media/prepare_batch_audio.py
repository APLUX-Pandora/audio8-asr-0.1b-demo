"""Extract the batch transcription sample from the installed lecture excerpt."""

import argparse
import json
import subprocess
import wave
from pathlib import Path
from tempfile import TemporaryDirectory


def main():
    media = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', nargs='?', type=Path, default=media / 'fosdem_beyond_readme_10min.mp4')
    parser.add_argument('--output-dir', type=Path, default=media)
    args = parser.parse_args()
    catalog = json.loads((media / 'catalog.json').read_text())
    sample = next(item['batch_audio'] for item in catalog if 'batch_audio' in item)
    duration = sample['end_s'] - sample['start_s']
    sample_rate = 16000
    expected_frames = round(duration * sample_rate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output_dir) as staging:
        output = Path(staging) / Path(sample['url']).name
        subprocess.run([
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin',
            '-ss', str(sample['start_s']), '-i', str(args.source),
            '-t', str(duration), '-map', '0:a:0',
            '-ac', '1', '-ar', str(sample_rate), '-c:a', 'pcm_s16le', str(output),
        ], check=True)
        with wave.open(str(output), 'rb') as audio:
            if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, sample_rate):
                parser.error('Expected mono 16 kHz PCM16 audio from ffmpeg')
            if (audio.getnframes() != expected_frames
                    or len(audio.readframes(expected_frames)) != expected_frames * 2):
                parser.error(f'Expected {expected_frames} complete frames in the extracted batch WAV')
        output.replace(args.output_dir / output.name)


if __name__ == '__main__':
    main()
