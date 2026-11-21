import json
import subprocess
import wave
from pathlib import Path

# Every aligned clip is written in this exact PCM format so merge_audio can
# copy raw samples without decoding. Edge TTS produces 24 kHz mono.
SAMPLE_RATE = 24_000
CHANNELS = 1
SAMPLE_WIDTH = 2  # bytes, s16le
PCM_ARGS = ["-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS), "-c:a", "pcm_s16le"]

# Edge TTS pads every clip with ~0.2s of leading and ~0.8s of trailing silence.
SILENCE_THRESHOLD = "-50dB"

# atempo accepts 0.5..100. Anything above this cap is cut off by atrim instead.
MAX_SPEED = 100.0

# Short fades so trimmed edges don't click.
FADE_IN_S = 0.005
FADE_OUT_S = 0.02


def get_audio_duration_ms(file_path: str | Path) -> int:
  """Get audio duration in milliseconds using ffprobe."""
  result = subprocess.run(
    [
      "ffprobe",
      "-v",
      "error",
      "-show_entries",
      "format=duration",
      "-of",
      "json",
      str(file_path),
    ],
    capture_output=True,
    text=True,
    check=True,
  )

  data = json.loads(result.stdout)
  duration = data.get("format", {}).get("duration")
  return round(float(duration) * 1000) if duration else 0


def get_wav_duration_ms(file_path: str | Path) -> int:
  """Exact WAV duration from the header (no subprocess)."""
  with wave.open(str(file_path), "rb") as wav:
    return wav.getnframes() * 1000 // wav.getframerate()


def trim_silence(input_path: Path, output_path: Path) -> None:
  """Remove leading and trailing silence, writing a WAV file."""
  edge = f"silenceremove=start_periods=1:start_threshold={SILENCE_THRESHOLD}"
  subprocess.run(
    [
      "ffmpeg",
      "-y",
      "-v",
      "error",
      "-i",
      str(input_path),
      "-af",
      f"{edge},areverse,{edge},areverse",
      *PCM_ARGS,
      str(output_path),
    ],
    check=True,
  )


def align_audio_file(
  input_path: str | Path,
  output_path: str | Path,
  target_duration_ms: int,
) -> dict:
  """Fit audio into target_duration_ms: trim silence, speed up, hard-cap length.

  The output never exceeds target_duration_ms. Shorter audio is left as is;
  the remaining time becomes silence when clips are placed on the timeline.
  Output is WAV so no encoder padding changes its length.
  """
  input_path = Path(input_path)
  output_path = Path(output_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)

  if target_duration_ms <= 0:
    raise ValueError("Target duration must be positive.")

  trimmed_path = output_path.with_name(f"{output_path.stem}.trim.wav")
  trim_silence(input_path, trimmed_path)

  actual_ms = get_wav_duration_ms(trimmed_path)
  source = trimmed_path

  if actual_ms <= 0:
    # Everything was below the threshold; fall back to the original clip.
    source = input_path
    actual_ms = get_audio_duration_ms(input_path)

  # Never slow down audio. Only speed it up when it is too long.
  speed = min(max(1.0, actual_ms / target_duration_ms), MAX_SPEED)
  target_s = target_duration_ms / 1000

  filters = []

  if speed > 1.001:
    filters.append(f"atempo={speed:.6f}")

  filters += [
    f"afade=t=in:st=0:d={FADE_IN_S}",
    # Hard limit: the clip can never run past the subtitle's end time.
    f"atrim=duration={target_s:.3f}",
    f"afade=t=out:st={max(0.0, target_s - FADE_OUT_S):.3f}:d={FADE_OUT_S}",
  ]

  subprocess.run(
    [
      "ffmpeg",
      "-y",
      "-v",
      "error",
      "-i",
      str(source),
      "-af",
      ",".join(filters),
      *PCM_ARGS,
      str(output_path),
    ],
    check=True,
  )

  trimmed_path.unlink(missing_ok=True)

  output_ms = get_wav_duration_ms(output_path)
  expected_ms = actual_ms / speed

  return {
    "input_duration_ms": actual_ms,
    "target_duration_ms": target_duration_ms,
    "speed": speed,
    "output_duration_ms": output_ms,
    # True when speed hit MAX_SPEED and atrim had to cut speech.
    "truncated": expected_ms > target_duration_ms + 1,
  }
