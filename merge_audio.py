import subprocess
import wave
from collections.abc import Sequence
from pathlib import Path

from align_audio import CHANNELS, SAMPLE_RATE, SAMPLE_WIDTH


def merge_audio(
  clips: list[dict],
  total_duration_ms: int,
  output_path: str | Path,
  codec_args: Sequence[str] = ("-codec:a", "libmp3lame", "-q:a", "2"),
) -> None:
  """Paste each clip onto a silent timeline at its start time, then encode.

  codec_args are the ffmpeg output options for the chosen format
  (see audio_format.FORMATS).

  clips: [{"path": Path, "start_ms": int, "slot_ms": int}, ...]
  Clips are guaranteed not to overlap (see compute_slots), so no mixing is
  needed: each clip is copied into its own region of one PCM buffer. Work is
  proportional to the timeline length, not clips x timeline like amix.
  """
  output_path = Path(output_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)

  if not clips:
    raise ValueError("No clips to merge.")

  frame_bytes = SAMPLE_WIDTH * CHANNELS
  total_frames = ms_to_frames(total_duration_ms)
  timeline = bytearray(total_frames * frame_bytes)

  for clip in clips:
    with wave.open(str(clip["path"]), "rb") as wav:
      if (
        wav.getframerate() != SAMPLE_RATE
        or wav.getnchannels() != CHANNELS
        or wav.getsampwidth() != SAMPLE_WIDTH
      ):
        raise ValueError(f"Unexpected audio format: {clip['path']}")

      data = wav.readframes(wav.getnframes())

    start = ms_to_frames(clip["start_ms"])
    # Never write past the slot, even by one sample.
    max_frames = min(ms_to_frames(clip["slot_ms"]), total_frames - start)
    data = data[: max_frames * frame_bytes]

    offset = start * frame_bytes
    timeline[offset : offset + len(data)] = data

  # Encode the raw PCM straight from memory.
  subprocess.run(
    [
      "ffmpeg",
      "-y",
      "-v",
      "error",
      "-f",
      "s16le",
      "-ar",
      str(SAMPLE_RATE),
      "-ac",
      str(CHANNELS),
      "-i",
      "pipe:0",
      *codec_args,
      str(output_path),
    ],
    input=bytes(timeline),
    check=True,
  )


def ms_to_frames(ms: int) -> int:
  return ms * SAMPLE_RATE // 1000
