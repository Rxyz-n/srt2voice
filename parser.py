import re
from pathlib import Path

# A cue starts at a timing line. Anything after the end timestamp (position
# coordinates like "X1:100 X2:200", styling) is ignored.
TIMING_LINE = re.compile(r"^(\S+)\s*-->\s*(\S+)")


def read_text(path: Path) -> str:
  """Decode an SRT file, detecting the encoding.

  Order: UTF-16 (BOM or many NUL bytes) -> UTF-8 -> Windows-1252 -> Latin-1.
  Latin-1 accepts any byte, so decoding never fails; legacy Vietnamese
  encodings (VNI, TCVN3) can't be detected and will come out garbled.
  """
  data = path.read_bytes()

  if data.startswith((b"\xff\xfe", b"\xfe\xff")):
    return data.decode("utf-16")

  # UTF-16 without BOM: ASCII text has a NUL in every other byte.
  if data and data.count(b"\x00") > len(data) // 4:
    for encoding in ("utf-16-le", "utf-16-be"):
      try:
        return data.decode(encoding)
      except UnicodeDecodeError:
        pass

  for encoding in ("utf-8-sig", "cp1252"):
    try:
      return data.decode(encoding)
    except UnicodeDecodeError:
      pass

  return data.decode("latin-1")


def is_index_line(lines: list[str], i: int) -> bool:
  """True if lines[i] is a cue number: digits only, at the start of the file
  or right after a blank line (so a numeric text line isn't mistaken)."""
  return i >= 0 and lines[i].isdigit() and (i == 0 or not lines[i - 1])


def parse_srt(file_path: str | Path) -> list[dict]:
  """Parse an SRT file into a list of subtitle dictionaries.

  Line-based rather than splitting on blank lines, so it tolerates:
  - whitespace-only or repeated blank lines between cues
  - blank lines inside a cue's text
  - missing index numbers (numbered by position instead)
  - extra data after the end timestamp
  Each subtitle records its timing line number ("line") for error messages.
  """
  text = read_text(Path(file_path))
  lines = [
    line.strip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
  ]

  timing_rows = [i for i, line in enumerate(lines) if TIMING_LINE.match(line)]
  subtitles = []

  for position, row in enumerate(timing_rows):
    start, end = TIMING_LINE.match(lines[row]).groups()

    # Text runs until the next cue's index line (or timing line if unnumbered).
    stop = timing_rows[position + 1] if position + 1 < len(timing_rows) else len(lines)
    if is_index_line(lines, stop - 1) and stop < len(lines):
      stop -= 1

    text_lines = [line for line in lines[row + 1 : stop] if line]

    if not text_lines:
      continue

    has_index = is_index_line(lines, row - 1)

    subtitles.append(
      {
        "index": int(lines[row - 1]) if has_index else position + 1,
        "line": row + 1,
        "start": start,
        "end": end,
        "text": "\n".join(text_lines),
      }
    )

  return subtitles
