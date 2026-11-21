import re

# Accepts 1+ digit hours and 1-3 digit fractions: 0:00:01,5 / 00:00:01.500
TIMESTAMP_PATTERN = re.compile(r"(\d+):([0-5]\d):([0-5]\d)[,.](\d{1,3})")


def timestamp_to_ms(timestamp: str) -> int:
  """Convert an SRT timestamp to milliseconds."""
  match = TIMESTAMP_PATTERN.fullmatch(timestamp.strip())

  if not match:
    raise ValueError(f"invalid timestamp {timestamp!r}")

  hours, minutes, seconds, fraction = match.groups()

  return (
    int(hours) * 3_600_000
    + int(minutes) * 60_000
    + int(seconds) * 1_000
    # ",5" means half a second, not 5ms.
    + int(fraction.ljust(3, "0"))
  )


def location(subtitle: dict) -> str:
  """'#12 (line 45)' for error messages."""
  line = subtitle.get("line")
  return f"#{subtitle['index']}" + (f" (line {line})" if line else "")


def analyze_timeline(
  subtitles: list[dict],
) -> list[dict]:
  """Validate subtitle timing and add start_ms / end_ms."""
  results = []

  for subtitle in subtitles:
    try:
      start_ms = timestamp_to_ms(subtitle["start"])
      end_ms = timestamp_to_ms(subtitle["end"])
    except ValueError as exc:
      raise ValueError(f"Subtitle {location(subtitle)}: {exc}") from None

    if end_ms <= start_ms:
      raise ValueError(
        f"Subtitle {location(subtitle)}: end time must be later than start time."
      )

    results.append({**subtitle, "start_ms": start_ms, "end_ms": end_ms})

  return results


def merge_same_start(results: list[dict]) -> tuple[list[dict], list[str]]:
  """Sort by start time and combine cues that start at the same moment.

  Two cues at the same start (e.g. two speakers) would leave one of them no
  time to speak. They become one cue: texts joined, ending at the later end.
  Returns (merged results, notes describing each merge).
  """
  ordered = sorted(results, key=lambda item: item["start_ms"])
  merged = []
  notes = []

  for subtitle in ordered:
    previous = merged[-1] if merged else None

    if previous and previous["start_ms"] == subtitle["start_ms"]:
      previous["text"] = f"{previous['text']}\n{subtitle['text']}"
      previous["end_ms"] = max(previous["end_ms"], subtitle["end_ms"])
      notes.append(
        f"{location(subtitle)}: same start time as #{previous['index']}, "
        f"merged into one line"
      )
      continue

    merged.append(dict(subtitle))

  return merged, notes
