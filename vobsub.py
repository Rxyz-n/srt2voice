"""Convert VobSub (.idx + .sub) image subtitles to .srt using Tesseract OCR.

VobSub stores each subtitle as a small bitmap (DVD subpicture). This module
reads the .idx index, extracts every subpicture from the .sub file, renders
it as black text on a white background and runs Tesseract on it.

Standalone usage:
  python vobsub.py movie.idx                 # -> movie.srt next to it
  python vobsub.py movie.idx -o out.srt
  python vobsub.py movie.idx --list-tracks   # show language tracks
  python vobsub.py movie.idx --track vi      # pick a language track
  python vobsub.py movie.idx --ocr-lang vie  # override Tesseract language

Requires the `tesseract` command and its language data, e.g.
  sudo apt install tesseract-ocr tesseract-ocr-eng tesseract-ocr-vie
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

# .idx language id (ISO 639-1) -> Tesseract language code.
TESSERACT_LANGS = {
  "ar": "ara", "cs": "ces", "da": "dan", "de": "deu", "el": "ell",
  "en": "eng", "es": "spa", "fi": "fin", "fr": "fra", "he": "heb",
  "hu": "hun", "id": "ind", "it": "ita", "ja": "jpn", "ko": "kor",
  "nl": "nld", "no": "nor", "pl": "pol", "pt": "por", "ro": "ron",
  "ru": "rus", "sv": "swe", "th": "tha", "tr": "tur", "uk": "ukr",
  "vi": "vie", "zh": "chi_sim",
}  # fmt: skip

# Used when a subpicture has no stop command: show until the next one,
# but never longer than this.
DEFAULT_DURATION_MS = 5000

# Each text line is enlarged so its lowercase letters are about this tall;
# Tesseract is most accurate around 20-30 px x-height. Small DVD text is
# often only 10-15 px.
TARGET_X_HEIGHT = 30
MAX_SCALE = 8
PADDING = 20

# A color counts as letter fill if it is at least this bright relative to the
# brightest color in the subpicture.
TEXT_BRIGHTNESS = 0.6

# Rows with at least this fraction of the densest row's ink are the "core"
# of a text line (the x-height zone). Each line has exactly one core.
CORE_DENSITY = 0.2


# ---------------------------------------------------------------------------
# .idx parsing
# ---------------------------------------------------------------------------


@dataclass
class Track:
  language: str  # e.g. "en"
  index: int  # substream number in the .sub file
  entries: list[tuple[int, int]] = field(default_factory=list)  # (ms, filepos)


@dataclass
class IdxFile:
  palette: list[tuple[int, int, int]]
  tracks: list[Track]
  default_index: int = 0


def parse_timestamp(value: str) -> int:
  """'HH:MM:SS:mmm' (optionally signed) -> milliseconds."""
  sign = -1 if value.startswith("-") else 1
  hours, minutes, seconds, millis = (int(p) for p in value.lstrip("+-").split(":"))
  return sign * (((hours * 60 + minutes) * 60 + seconds) * 1000 + millis)


def parse_idx(path: Path) -> IdxFile:
  palette = [(0, 0, 0)] * 16
  tracks: list[Track] = []
  default_index = 0
  delay = 0

  for raw in path.read_text(encoding="latin-1").splitlines():
    line = raw.strip()

    if not line or line.startswith("#"):
      continue

    key, _, value = line.partition(":")
    key, value = key.strip().lower(), value.strip()

    if key == "palette":
      colors = [c.strip() for c in value.split(",")][:16]
      palette = [
        (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)) for c in colors
      ] + palette[len(colors) :]

    elif key == "langidx":
      default_index = int(value)

    elif key == "id":
      # "id: en, index: 0"
      match = re.match(r"(\w*)\s*,\s*index:\s*(\d+)", value)
      if match:
        tracks.append(Track(match.group(1) or "en", int(match.group(2))))
        delay = 0

    elif key == "delay" and tracks:
      delay += parse_timestamp(value)

    elif key == "timestamp" and tracks:
      # "timestamp: 00:00:01:234, filepos: 000000000"
      match = re.match(r"([+-]?[\d:]+)\s*,\s*filepos:\s*([0-9a-fA-F]+)", value)
      if match:
        ms = parse_timestamp(match.group(1)) + delay
        tracks[-1].entries.append((ms, int(match.group(2), 16)))

  return IdxFile(palette, tracks, default_index)


# ---------------------------------------------------------------------------
# .sub (MPEG program stream) reading
# ---------------------------------------------------------------------------


def read_spu(data: bytes, pos: int, stream_index: int) -> bytes:
  """Collect one subpicture unit for a substream, starting at pos.

  A subpicture is split across PES packets (private stream 1, substream
  0x20 + index) inside MPEG-PS packs; its first two bytes give its size.
  """
  spu = bytearray()
  size = None

  while pos + 6 <= len(data):
    if data[pos : pos + 3] != b"\x00\x00\x01":
      raise ValueError(f"Bad MPEG start code at byte {pos}")

    code = data[pos + 3]

    if code == 0xBA:  # pack header
      if data[pos + 4] & 0xC0 == 0x40:  # MPEG-2
        pos += 14 + (data[pos + 13] & 0x07)
      else:  # MPEG-1
        pos += 12
      continue

    if code == 0xB9:  # program end
      break

    length = int.from_bytes(data[pos + 4 : pos + 6], "big")
    body = pos + 6
    next_pos = body + length

    if code == 0xBD:  # private stream 1
      p = body

      if data[p] & 0xC0 == 0x80:  # MPEG-2 PES header
        p += 3 + data[p + 2]
      else:  # MPEG-1 PES header
        while data[p] == 0xFF:
          p += 1
        if data[p] & 0xC0 == 0x40:
          p += 2
        p += {0x20: 5, 0x30: 10}.get(data[p] & 0xF0, 1)

      if data[p] == 0x20 + stream_index:
        chunk = data[p + 1 : next_pos]

        if size is None:
          size = int.from_bytes(chunk[:2], "big")

        spu += chunk

        if len(spu) >= size:
          return bytes(spu[:size])

    pos = next_pos

  raise ValueError("Subpicture ends before its declared size")


# ---------------------------------------------------------------------------
# Subpicture decoding
# ---------------------------------------------------------------------------


@dataclass
class SubPicture:
  start_ms: float
  stop_ms: float | None
  width: int
  height: int
  pixels: bytearray  # one 2-bit color index (0-3) per pixel
  colors: list[int]  # palette index for each of the 4 pixel values
  alpha: list[int]  # 0 (transparent) - 15 (opaque) for each pixel value


def decode_spu(spu: bytes) -> SubPicture | None:
  """Decode control commands and RLE bitmap. None if nothing is drawn."""
  control_offset = int.from_bytes(spu[2:4], "big")
  start_ms = 0.0
  stop_ms = None
  colors = [0, 1, 2, 3]
  alpha = [0, 15, 15, 15]
  coords = None
  offsets = None

  offset = control_offset
  seen = set()

  while offset + 4 <= len(spu) and offset not in seen:
    seen.add(offset)
    # Delay unit is 1024/90000 s.
    delay_ms = int.from_bytes(spu[offset : offset + 2], "big") * 1024 / 90
    next_offset = int.from_bytes(spu[offset + 2 : offset + 4], "big")
    p = offset + 4

    while p < len(spu):
      command = spu[p]
      p += 1

      if command in (0x00, 0x01):  # (forced) start display
        start_ms = delay_ms
      elif command == 0x02:  # stop display
        stop_ms = delay_ms
      elif command in (0x03, 0x04):  # palette / alpha, nibbles for values 3,2,1,0
        b0, b1 = spu[p], spu[p + 1]
        values = [b1 & 0x0F, b1 >> 4, b0 & 0x0F, b0 >> 4]
        if command == 0x03:
          colors = values
        else:
          alpha = values
        p += 2
      elif command == 0x05:  # display area
        c = spu[p : p + 6]
        x1, x2 = (c[0] << 4) | (c[1] >> 4), ((c[1] & 0x0F) << 8) | c[2]
        y1, y2 = (c[3] << 4) | (c[4] >> 4), ((c[4] & 0x0F) << 8) | c[5]
        coords = (x1, x2, y1, y2)
        p += 6
      elif command == 0x06:  # RLE offsets of top and bottom fields
        offsets = (
          int.from_bytes(spu[p : p + 2], "big"),
          int.from_bytes(spu[p + 2 : p + 4], "big"),
        )
        p += 4
      else:  # 0xFF end of sequence, or unknown
        break

    if next_offset == offset:
      break
    offset = next_offset

  if not coords or not offsets:
    return None

  x1, x2, y1, y2 = coords
  width, height = x2 - x1 + 1, y2 - y1 + 1

  if width <= 0 or height <= 0:
    return None

  pixels = bytearray(width * height)
  decode_field(spu, offsets[0], pixels, width, range(0, height, 2))
  decode_field(spu, offsets[1], pixels, width, range(1, height, 2))

  return SubPicture(start_ms, stop_ms, width, height, pixels, colors, alpha)


def decode_field(
  spu: bytes, offset: int, pixels: bytearray, width: int, rows: range
) -> None:
  """Decode one interlaced field of 2-bit run-length codes into pixels."""
  nibble = offset * 2
  limit = len(spu) * 2

  def read() -> int:
    nonlocal nibble
    if nibble >= limit:
      return 0
    byte = spu[nibble // 2]
    value = byte >> 4 if nibble % 2 == 0 else byte & 0x0F
    nibble += 1
    return value

  for row in rows:
    x = 0
    base = row * width

    while x < width:
      code = read()
      if code < 0x4:
        code = (code << 4) | read()
        if code < 0x10:
          code = (code << 4) | read()
          if code < 0x40:
            code = (code << 4) | read()

      run, color = code >> 2, code & 0x3
      run = width - x if run == 0 else min(run, width - x)  # 0 = to end of line
      pixels[base + x : base + x + run] = bytes([color]) * run
      x += run

      if nibble >= limit:
        return

    nibble += nibble % 2  # each line starts on a byte boundary


def text_mask(
  picture: SubPicture, palette: list[tuple[int, int, int]]
) -> tuple[int, int, bytes]:
  """Black text on white at native size. Returns (width, height, gray bytes).

  DVD subtitles are bright text with a darker outline. Only the brightest
  visible colors (the letter fill) become black; outline, anti-aliasing and
  transparent background become white, so letters don't merge together.
  """
  brightness = []
  for value in range(4):
    r, g, b = palette[picture.colors[value]]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    brightness.append(luminance * picture.alpha[value] / 15)

  threshold = max(brightness) * TEXT_BRIGHTNESS
  shade = bytearray(range(256))
  for value in range(4):
    shade[value] = 0 if brightness[value] and brightness[value] >= threshold else 255

  return picture.width, picture.height, bytes(picture.pixels.translate(shade))


def has_ink(image: bytes) -> bool:
  return min(image, default=255) < 128


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------


def check_tesseract(language: str) -> None:
  """Fail early with install instructions if Tesseract or its data is missing."""
  if not shutil.which("tesseract"):
    raise RuntimeError(
      "Tesseract is not installed. Install it with:\n"
      f"  sudo apt install tesseract-ocr tesseract-ocr-{language}"
    )

  result = subprocess.run(
    ["tesseract", "--list-langs"], capture_output=True, text=True, check=False
  )
  available = set(result.stdout.split()[1:])  # first line is a header

  missing = [lang for lang in language.split("+") if lang not in available]
  if missing:
    packages = " ".join(f"tesseract-ocr-{lang.replace('_', '-')}" for lang in missing)
    raise RuntimeError(
      f"Tesseract language data missing: {', '.join(missing)}. Install with:\n"
      f"  sudo apt install {packages}"
    )


def find_lines(width: int, height: int, gray: bytes) -> list[tuple[int, int, int]]:
  """Split an image into text lines. Returns (top, bottom, x_height) per line.

  Each line has one dense "core" of rows (its lowercase letters). Lines are
  cut at the emptiest row between consecutive cores, which works even when
  descenders of one line touch ascenders or accents of the next. Accents and
  dots stay with the line they belong to instead of becoming lines.
  """
  ink = [gray[y * width : (y + 1) * width].count(0) for y in range(height)]
  rows = [y for y in range(height) if ink[y]]

  if not rows:
    return []

  dense = max(ink) * CORE_DENSITY
  cores = []
  for y in range(height):
    if ink[y] >= dense:
      if cores and cores[-1][1] == y:
        cores[-1][1] = y + 1
      else:
        cores.append([y, y + 1])

  # Join core fragments split by a thin sparse row (e.g. inside an "e").
  merged = [cores[0]]
  for top, bottom in cores[1:]:
    previous = merged[-1]
    if top - previous[1] < (previous[1] - previous[0]) * 0.5:
      previous[1] = bottom
    else:
      merged.append([top, bottom])

  # Ignore slivers (a lone underline or dash row) when counting lines.
  tallest = max(bottom - top for top, bottom in merged)
  cores = [c for c in merged if c[1] - c[0] >= tallest * 0.4]

  cuts = [rows[0]]
  for (_, upper_bottom), (lower_top, _) in zip(cores, cores[1:]):
    cuts.append(min(range(upper_bottom, lower_top), key=lambda y: ink[y]))
  cuts.append(rows[-1] + 1)

  return [
    (top, bottom, core_bottom - core_top)
    for (top, bottom), (core_top, core_bottom) in zip(zip(cuts, cuts[1:]), cores)
  ]


def line_image(
  width: int, gray: bytes, top: int, bottom: int, x_height: int
) -> tuple[int, int, bytes]:
  """Crop one line, enlarge it to TARGET_X_HEIGHT and pad it with white."""
  scale = max(1, min(MAX_SCALE, round(TARGET_X_HEIGHT / max(1, x_height))))
  out_width = width * scale + PADDING * 2
  out_height = (bottom - top) * scale + PADDING * 2
  image = bytearray(b"\xff" * out_width * out_height)

  for y in range(top, bottom):
    source = gray[y * width : (y + 1) * width]
    row = bytearray(width * scale)
    for dx in range(scale):
      row[dx::scale] = source
    for dy in range(scale):
      start = (PADDING + (y - top) * scale + dy) * out_width + PADDING
      image[start : start + len(row)] = row

  return out_width, out_height, bytes(image)


def ocr_image(width: int, height: int, gray: bytes, language: str, workdir: Path) -> str:
  """OCR a subtitle image line by line.

  Each line is read with Tesseract's single-line mode (--psm 7). Block mode
  (--psm 6) occasionally misreads the layout of a clean subtitle and returns
  garbage; single lines are read reliably.
  """
  lines = []

  for top, bottom, x_height in find_lines(width, height, gray):
    line_width, line_height, pixels = line_image(width, gray, top, bottom, x_height)

    fd, name = tempfile.mkstemp(suffix=".pgm", dir=workdir)
    with os.fdopen(fd, "wb") as file:
      file.write(f"P5\n{line_width} {line_height}\n255\n".encode() + pixels)

    try:
      result = subprocess.run(
        ["tesseract", name, "stdout", "-l", language, "--psm", "7"],
        capture_output=True,
        text=True,
        check=True,
        # One thread per process; parallelism comes from running many at once.
        env={**os.environ, "OMP_THREAD_LIMIT": "1"},
      )
    finally:
      os.unlink(name)

    lines.append(result.stdout)

  return clean_text("\n".join(lines))


def clean_text(text: str) -> str:
  lines = [line.strip() for line in text.splitlines()]
  text = "\n".join(line for line in lines if line)
  # Common OCR confusion on subtitle fonts.
  return text.replace("|", "I")


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def format_srt_time(ms: int) -> str:
  hours, rest = divmod(max(0, int(ms)), 3_600_000)
  minutes, rest = divmod(rest, 60_000)
  seconds, millis = divmod(rest, 1000)
  return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


def find_pair(path: str | Path) -> tuple[Path, Path]:
  """Accept either file of the pair; return (idx, sub)."""
  path = Path(path)

  def locate(suffix: str) -> Path:
    # Accept MOVIE.IDX / MOVIE.SUB as well as lower case.
    for candidate in (path.with_suffix(suffix), path.with_suffix(suffix.upper())):
      if candidate.exists():
        return candidate
    raise FileNotFoundError(
      f"Missing {path.with_suffix(suffix).name} (VobSub needs both .idx and .sub)"
    )

  return locate(".idx"), locate(".sub")


def select_track(idx: IdxFile, language: str | None) -> Track:
  usable = [t for t in idx.tracks if t.entries]

  if not usable:
    raise ValueError("No subtitle entries found in .idx file.")

  if language:
    for track in usable:
      if track.language.lower() == language.lower() or str(track.index) == language:
        return track
    names = ", ".join(f"{t.language} (index {t.index})" for t in usable)
    raise ValueError(f"Track {language!r} not found. Available: {names}")

  for track in usable:
    if track.index == idx.default_index:
      return track

  return usable[0]


def convert_vobsub(
  input_path: str | Path,
  output_path: str | Path | None = None,
  track: str | None = None,
  ocr_lang: str | None = None,
  workers: int | None = None,
  on_progress: Callable[[int, int], None] | None = None,
) -> dict:
  """Convert a VobSub pair to SRT. Returns stats about the conversion.

  on_progress(done, total) is called from worker threads as images finish.
  """
  idx_path, sub_path = find_pair(input_path)
  output_path = Path(output_path) if output_path else idx_path.with_suffix(".srt")

  idx = parse_idx(idx_path)
  chosen = select_track(idx, track)
  language = ocr_lang or TESSERACT_LANGS.get(chosen.language.lower(), "eng")
  check_tesseract(language)

  data = sub_path.read_bytes()
  entries = sorted(chosen.entries)

  # Decode every subpicture and work out its display time.
  jobs = []
  for i, (timestamp, filepos) in enumerate(entries):
    picture = decode_spu(read_spu(data, filepos, chosen.index))
    if picture is None:
      continue

    start = timestamp + picture.start_ms
    next_start = entries[i + 1][0] if i + 1 < len(entries) else None

    if picture.stop_ms is not None:
      end = timestamp + picture.stop_ms
    else:
      end = start + DEFAULT_DURATION_MS
    if next_start is not None:
      end = min(end, next_start)

    image = text_mask(picture, idx.palette)
    if end > start and has_ink(image[2]):
      jobs.append((round(start), round(end), image))

  total = len(jobs)
  done = 0
  lock = threading.Lock()

  def run(job, workdir):
    nonlocal done
    start, end, (width, height, gray) = job
    text = ocr_image(width, height, gray, language, workdir)
    with lock:
      done += 1
      if on_progress:
        on_progress(done, total)
    return start, end, text

  if on_progress:
    on_progress(0, total)

  with tempfile.TemporaryDirectory(prefix="vobsub-") as tmp:
    with ThreadPoolExecutor(max_workers=workers or os.cpu_count() or 4) as pool:
      results = list(pool.map(lambda job: run(job, Path(tmp)), jobs))

  cues = [(start, end, text) for start, end, text in results if text]

  output_path.parent.mkdir(parents=True, exist_ok=True)
  with open(output_path, "w", encoding="utf-8") as srt:
    for number, (start, end, text) in enumerate(cues, start=1):
      srt.write(f"{number}\n{format_srt_time(start)} --> {format_srt_time(end)}\n{text}\n\n")

  return {
    "output": output_path,
    "track": chosen.language,
    "ocr_language": language,
    "images": total,
    "cues": len(cues),
    "empty": total - len(cues),
  }


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Convert VobSub (.idx + .sub) subtitles to .srt with Tesseract OCR."
  )
  parser.add_argument("input", help=".idx or .sub file (both must exist)")
  parser.add_argument("-o", "--output", help="output .srt (default: next to input)")
  parser.add_argument("--track", help="language id (e.g. en) or index of the track")
  parser.add_argument("--ocr-lang", help="Tesseract language, e.g. eng, vie, eng+vie")
  parser.add_argument("--list-tracks", action="store_true", help="show tracks and exit")
  args = parser.parse_args()

  try:
    if args.list_tracks:
      idx = parse_idx(find_pair(args.input)[0])
      for t in idx.tracks:
        default = " (default)" if t.index == idx.default_index else ""
        print(f"{t.language}  index {t.index}  {len(t.entries)} subtitles{default}")
      return

    def show(done: int, total: int) -> None:
      print(f"\rOCR: {done}/{total}", end="", flush=True)

    stats = convert_vobsub(args.input, args.output, args.track, args.ocr_lang, on_progress=show)

  except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
    print(f"\nError: {exc}", file=sys.stderr)
    raise SystemExit(1) from None

  print(
    f"\rWrote {stats['cues']} subtitles to {stats['output']} "
    f"(track {stats['track']}, OCR {stats['ocr_language']}"
    + (f", {stats['empty']} images had no text" if stats["empty"] else "")
    + ")"
  )


if __name__ == "__main__":
  main()
