"""Output audio formats and the menu to choose one."""

import subprocess
import sys
from dataclasses import dataclass

import menu

DEFAULT_FORMAT = "mp3"


@dataclass(frozen=True)
class AudioFormat:
  name: str
  extension: str  # including the dot
  encoder: str  # ffmpeg encoder that must be available
  args: tuple[str, ...]  # ffmpeg output options
  description: str


# Speech from Edge TTS is 24 kHz mono, so modest bitrates are transparent.
FORMATS = [
  AudioFormat(
    "mp3", ".mp3", "libmp3lame", ("-codec:a", "libmp3lame", "-q:a", "2"),
    "MP3 (VBR ~190 kbps) - plays everywhere",
  ),
  AudioFormat(
    "m4a", ".m4a", "aac", ("-codec:a", "aac", "-b:a", "128k", "-movflags", "+faststart"),
    "AAC in M4A (128 kbps) - Apple devices, video editors",
  ),
  AudioFormat(
    "opus", ".opus", "libopus", ("-codec:a", "libopus", "-b:a", "64k"),
    "Opus (64 kbps) - smallest file at the same quality",
  ),
  AudioFormat(
    "ogg", ".ogg", "libvorbis", ("-codec:a", "libvorbis", "-q:a", "5"),
    "Ogg Vorbis (~160 kbps) - open format",
  ),
  AudioFormat(
    "flac", ".flac", "flac", ("-codec:a", "flac"),
    "FLAC - lossless, compressed",
  ),
  AudioFormat(
    "wav", ".wav", "pcm_s16le", ("-codec:a", "pcm_s16le"),
    "WAV (16-bit PCM) - lossless, uncompressed, largest",
  ),
]  # fmt: skip


def available_formats() -> list[AudioFormat]:
  """Formats whose encoder this ffmpeg build provides."""
  result = subprocess.run(
    ["ffmpeg", "-hide_banner", "-encoders"],
    capture_output=True,
    text=True,
    check=False,
  )
  encoders = {line.split()[1] for line in result.stdout.splitlines() if len(line.split()) > 1}
  return [fmt for fmt in FORMATS if fmt.encoder in encoders]


def describe(fmt: AudioFormat) -> str:
  return f"{fmt.name:<6} {fmt.extension:<7} {fmt.description}"


def get_format(name: str) -> AudioFormat:
  for fmt in FORMATS:
    if fmt.name == name:
      return fmt
  raise ValueError(f"Unknown audio format {name!r}")


def select_format(default: str = DEFAULT_FORMAT) -> AudioFormat:
  """Let the user choose the output format. Exits the program on Esc."""
  formats = available_formats()

  if not formats:
    raise SystemExit("No supported audio encoder found in ffmpeg.")

  if default not in {fmt.name for fmt in formats}:
    default = formats[0].name

  if not (sys.stdin.isatty() and sys.stdout.isatty()):
    choice = get_format(default)
    print(f"Non-interactive input; using format: {choice.name}")
    return choice

  try:
    choice = menu.pick(
      "Select audio format",
      formats,
      key=lambda fmt: fmt.name,
      describe=describe,
      default=default,
      noun="formats",
    )
  except ImportError:
    choice = pick_by_name(formats, default)

  if choice is None:
    print("Cancelled.")
    raise SystemExit(0)

  print(f"Selected format: {choice.name} ({choice.description})")
  return choice


def pick_by_name(formats: list[AudioFormat], default: str) -> AudioFormat:
  """Plain prompt for terminals without curses."""
  names = {fmt.name: fmt for fmt in formats}

  for fmt in formats:
    print(f"  {describe(fmt)}")

  while True:
    choice = input(f"Audio format [{default}]: ").strip().lower() or default

    if choice in names:
      return names[choice]

    print(f"Unknown format. Choose one of: {', '.join(names)}")
