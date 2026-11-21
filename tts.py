import asyncio
from collections.abc import Callable
from pathlib import Path

import edge_tts
from edge_tts.exceptions import NoAudioReceived

VOICE = "en-US-AriaNeural"

RETRIES = 4
RETRY_BASE_DELAY_S = 1.0


async def generate_speech(
  text: str,
  output_path: str | Path,
  voice: str = VOICE,
  on_retry: Callable[[], None] | None = None,
) -> None:
  """Generate an MP3 file from text using Edge TTS.

  Retries transient network errors with exponential backoff, calling
  on_retry before each new attempt. Writes to a
  temporary file and renames it, so an existing output is always complete
  (safe to reuse as a cache after an interrupted run).
  """
  text = text.strip()

  if not text:
    raise ValueError("Cannot generate speech from empty text.")

  output = Path(output_path)
  output.parent.mkdir(parents=True, exist_ok=True)
  partial = output.with_name(f"{output.name}.part")

  for attempt in range(RETRIES + 1):
    try:
      communicate = edge_tts.Communicate(text, voice)
      await communicate.save(str(partial))
      partial.replace(output)
      return

    except Exception as exc:
      partial.unlink(missing_ok=True)

      # Edge returns no audio for text it can't speak ("♪", "...").
      # Retrying won't change that, so fail immediately.
      if isinstance(exc, NoAudioReceived) or attempt == RETRIES:
        raise

      if on_retry:
        on_retry()

      await asyncio.sleep(RETRY_BASE_DELAY_S * 2**attempt)
