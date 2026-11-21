import asyncio
import hashlib
import os
import shutil
import time
from pathlib import Path

from align_audio import align_audio_file
from audio_format import AudioFormat, select_format
from merge_audio import merge_audio
from parser import parse_srt
from progress import FileProgress, format_elapsed
from timing import analyze_timeline, merge_same_start
from tts import NoAudioReceived, generate_speech
from vobsub import convert_vobsub, find_pair
from voice_picker import select_voice

SUBTITLE_DIR = Path("subtitles")
OUTPUT_DIR = Path("output")
TEMP_DIR = Path("temp")

# Parallel Edge TTS requests (network bound). Too high may get rate limited.
TTS_CONCURRENCY = 8
# Parallel ffmpeg processes (CPU bound).
ALIGN_CONCURRENCY = os.cpu_count() or 4

def compute_slots(results: list[dict]) -> list[int]:
  """Return the allowed audio duration (ms) for each subtitle, in input order.

  A line may speak from its start until its own end time, but never past the
  start of the next line (handles SRTs whose cues overlap).
  """
  order = sorted(range(len(results)), key=lambda i: results[i]["start_ms"])
  slots = [0] * len(results)

  for rank, i in enumerate(order):
    subtitle = results[i]
    slot_end = subtitle["end_ms"]

    if rank + 1 < len(order):
      slot_end = min(slot_end, results[order[rank + 1]]["start_ms"])

    slot_ms = slot_end - subtitle["start_ms"]

    if slot_ms <= 0:
      raise ValueError(
        f"Subtitle #{subtitle['index']} starts at the same time as the next "
        f"line; there is no room to speak it."
      )

    slots[i] = slot_ms

  return slots


def tts_cache_name(position: int, voice: str, text: str) -> str:
  """File name that changes whenever the voice or text changes."""
  digest = hashlib.sha1(f"{voice}\n{text}".encode()).hexdigest()[:12]
  return f"{position:05}_{digest}.mp3"


def is_up_to_date(target: Path, *sources: Path) -> bool:
  return target.exists() and all(
    target.stat().st_mtime >= source.stat().st_mtime for source in sources
  )


async def vobsub_to_srt(input_path: Path, progress: FileProgress) -> tuple[Path, str]:
  """OCR a VobSub pair into output/<name>.srt. Returns (srt path, note).

  An existing .srt newer than both inputs is reused, since OCR is slow (and
  the user may have corrected OCR mistakes in it by hand).
  """
  idx_path, sub_path = find_pair(input_path)
  srt_path = OUTPUT_DIR / f"{input_path.stem}.srt"

  if is_up_to_date(srt_path, idx_path, sub_path):
    return srt_path, f"reused existing {srt_path} (delete it to OCR again)"

  loop = asyncio.get_running_loop()

  def on_progress(done: int, total: int) -> None:
    # Called from OCR worker threads; update the display on the event loop.
    loop.call_soon_threadsafe(progress.ocr, done, total)

  stats = await asyncio.to_thread(
    convert_vobsub, input_path, srt_path, on_progress=on_progress
  )
  progress.ocr_finished()

  note = (
    f"OCR ({stats['ocr_language']}) -> {srt_path}: {stats['cues']} subtitles"
    + (f", {stats['empty']} images had no text" if stats["empty"] else "")
  )
  return srt_path, note


async def process_subtitle_file(
  input_path: Path,
  voice: str,
  audio_format: AudioFormat,
  label: str | None = None,
) -> None:
  """Generate one audio file from one .srt or VobSub file, showing progress."""
  progress = FileProgress(label or input_path.name, total=0)
  ticker = asyncio.create_task(progress.tick())

  try:
    notes = []
    srt_path = input_path

    if input_path.suffix.lower() == ".idx":
      srt_path, note = await vobsub_to_srt(input_path, progress)
      notes.append(note)

    await _process_subtitle_file(srt_path, voice, audio_format, progress, notes)
  except Exception as exc:
    progress.fail(exc)
    raise
  finally:
    ticker.cancel()


async def _process_subtitle_file(
  srt_path: Path,
  voice: str,
  audio_format: AudioFormat,
  progress: FileProgress,
  notes: list[str],
) -> None:
  subtitles = parse_srt(srt_path)
  results, merge_notes = merge_same_start(analyze_timeline(subtitles))

  if not results:
    raise ValueError("No valid subtitles found.")

  slots = compute_slots(results)

  # Each SRT gets its own temporary directories. TTS files are kept across
  # failed/interrupted runs, so a rerun only generates what is missing.
  temp_root = TEMP_DIR / srt_path.stem
  audio_dir = temp_root / "audio"
  aligned_dir = temp_root / "aligned"

  audio_dir.mkdir(parents=True, exist_ok=True)
  aligned_dir.mkdir(parents=True, exist_ok=True)

  progress.total = len(results)
  progress.start()

  # (position, message); position 0 sorts file-level notes first.
  warnings = [(0, note) for note in merge_notes]
  clips = []

  tts_limit = asyncio.Semaphore(TTS_CONCURRENCY)
  align_limit = asyncio.Semaphore(ALIGN_CONCURRENCY)

  def skip(position: int, subtitle: dict, reason: str) -> None:
    """Leave this line silent instead of failing the whole file."""
    text = subtitle["text"].replace("\n", " ")
    warnings.append((position, f"#{subtitle['index']}: skipped, {reason}: {text!r}"))
    progress.line_done()

  async def process_line(position: int, subtitle: dict, slot_ms: int) -> None:
    if not any(char.isalnum() for char in subtitle["text"]):
      skip(position, subtitle, "no speakable text")
      return

    source = audio_dir / tts_cache_name(position, voice, subtitle["text"])
    aligned = aligned_dir / f"{position:05}.wav"
    cached = source.exists()

    if not cached:
      try:
        async with tts_limit:
          await generate_speech(
            subtitle["text"],
            source,
            voice,
            on_retry=progress.retry,
          )
      except NoAudioReceived:
        skip(position, subtitle, "Edge TTS returned no audio")
        return

    # Fit the audio inside [start, slot end] so it never reaches the next line.
    async with align_limit:
      result = await asyncio.to_thread(align_audio_file, source, aligned, slot_ms)

    if result["truncated"]:
      warnings.append(
        (
          position,
          (
            f"#{subtitle['index']}: speech cut off "
            f"(needs {result['input_duration_ms']}ms, slot {slot_ms}ms)"
          ),
        )
      )

    clips.append(
      {"path": aligned, "start_ms": subtitle["start_ms"], "slot_ms": slot_ms}
    )

    progress.line_done(cached=cached)

  try:
    async with asyncio.TaskGroup() as group:
      for position, (subtitle, slot_ms) in enumerate(zip(results, slots), 1):
        group.create_task(process_line(position, subtitle, slot_ms))

  except ExceptionGroup as group_error:
    # Surface the first real error instead of "unhandled errors in a TaskGroup".
    raise group_error.exceptions[0] from None

  if not clips:
    raise ValueError("No speakable subtitles found.")

  # Place all aligned clips onto the SRT timeline.
  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
  final_path = OUTPUT_DIR / f"{srt_path.stem}{audio_format.extension}"

  progress.merging()

  await asyncio.to_thread(
    merge_audio,
    clips,
    max(subtitle["end_ms"] for subtitle in results),
    final_path,
    audio_format.args,
  )

  # Delete temporary audio only after merge_audio succeeds.
  shutil.rmtree(temp_root)

  progress.finish(
    str(final_path),
    [message for _, message in sorted(warnings)],
    notes,
  )


def find_subtitle_files(folder: Path) -> list[Path]:
  """.srt files plus VobSub pairs (listed by their .idx) directly in folder.

  If a name exists as both .srt and .idx, the .srt is used: it is text
  already and needs no OCR.
  """
  files = [path for path in folder.iterdir() if path.is_file()]
  srt_stems = {path.stem for path in files if path.suffix.lower() == ".srt"}

  return sorted(
    path
    for path in files
    if path.suffix.lower() == ".srt"
    or (path.suffix.lower() == ".idx" and path.stem not in srt_stems)
  )


def main() -> None:
  SUBTITLE_DIR.mkdir(parents=True, exist_ok=True)
  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

  srt_files = find_subtitle_files(SUBTITLE_DIR)

  print(f"Found {len(srt_files)} subtitle files")

  if not srt_files:
    print(f"Place your .srt or .idx/.sub files in: {SUBTITLE_DIR.resolve()}")
    return

  voice = select_voice()
  audio_format = select_format()

  completed = 0
  failed = 0
  failures = []
  batch_started = time.monotonic()
  print()

  for number, srt_path in enumerate(srt_files, start=1):
    label = f"[{number}/{len(srt_files)}] {srt_path.name}"

    try:
      asyncio.run(process_subtitle_file(srt_path, voice, audio_format, label))
      completed += 1

    except Exception as exc:
      failed += 1
      failures.append((srt_path.name, str(exc)))

  print("\n========== Batch summary ==========")
  print(f"Time:      {format_elapsed(time.monotonic() - batch_started)}")
  print(f"Total:     {len(srt_files)}")
  print(f"Completed: {completed}")
  print(f"Failed:    {failed}")

  if failures:
    print("\nFailed files:")
    for filename, error in failures:
      print(f"- {filename}: {error}")

  print(f"\nOutput directory: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
  main()
