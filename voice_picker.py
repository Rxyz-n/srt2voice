import asyncio
import re
import sys

import edge_tts

import menu

DEFAULT_VOICE = "en-US-AriaNeural"

# Used when the voice list can't be downloaded.
FALLBACK_VOICES = [
  {
    "ShortName": "en-US-AriaNeural",
    "Gender": "Female",
    "Locale": "en-US",
    "LocaleName": "English (United States)",
  },
  {
    "ShortName": "en-US-GuyNeural",
    "Gender": "Male",
    "Locale": "en-US",
    "LocaleName": "English (United States)",
  },
]

def load_voices() -> list[dict]:
  """Fetch every Edge TTS voice, sorted by locale then name."""
  print("Loading voice list...")

  try:
    voices = asyncio.run(edge_tts.list_voices())
  except Exception as exc:
    print(f"Could not download voice list ({exc}); using built-in voices.")
    voices = FALLBACK_VOICES

  return sorted(voices, key=lambda v: (v["Locale"], v["ShortName"]))


def describe(voice: dict) -> str:
  """One display row: name, gender, language, personality."""
  personalities = ", ".join(
    voice.get("VoiceTag", {}).get("VoicePersonalities", [])
  )
  language = voice.get("LocaleName") or voice["Locale"]
  return (
    f"{voice['ShortName']:<34} {voice['Gender']:<7} {language:<36} {personalities}"
  )


def search_words(voice: dict) -> list[str]:
  """Lowercase words a filter can match, e.g. for en-US-AvaMultilingualNeural:
  en-us, en, us, ava, multilingual, neural, female, english, united, states...
  """
  locale, _, name = voice["ShortName"].partition("-")[2].partition("-")
  text = " ".join(
    [
      voice["Locale"],
      voice["Locale"].replace("-", " "),
      re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name),
      voice["Gender"],
      voice.get("LocaleName", ""),
      " ".join(voice.get("VoiceTag", {}).get("VoicePersonalities", [])),
    ]
  )
  return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


def match_score(words: list[str], terms: list[str]) -> int | None:
  """None if any term doesn't start a word; else how many terms match a
  whole word (used to rank exact hits first).

  Word-prefix matching keeps results precise: "vi" finds vi-VN / Vietnamese
  but not "Herzegovina", and "male" doesn't match "female".
  """
  score = 0

  for term in terms:
    if term in words:
      score += 1
    elif not any(word.startswith(term) for word in words):
      return None

  return score


def filter_voices(voices: list[dict], query: str) -> list[dict]:
  terms = query.lower().split()

  if not terms:
    return voices

  scored = []

  for voice in voices:
    words = voice.setdefault("_words", search_words(voice))
    score = match_score(words, terms)

    if score is not None:
      scored.append((-score, len(scored), voice))

  return [voice for *_, voice in sorted(scored)]


def pick_with_curses(voices: list[dict], default: str) -> str | None:
  """Arrow-key picker. Returns the ShortName, or None on Esc."""
  voice = menu.pick(
    "Select voice",
    voices,
    key=lambda v: v["ShortName"],
    describe=describe,
    default=default,
    filter_items=filter_voices,
    noun="voices",
  )
  return voice["ShortName"] if voice else None


def pick_by_name(voices: list[dict], default: str) -> str:
  """Plain prompt for terminals without curses (e.g. Windows without windows-curses)."""
  names = {v["ShortName"].lower(): v["ShortName"] for v in voices}
  print(f"{len(voices)} voices available, e.g. en-US-GuyNeural, vi-VN-HoaiMyNeural")

  while True:
    choice = input(f"Voice name [{default}]: ").strip() or default

    if choice.lower() in names:
      return names[choice.lower()]

    print("Unknown voice. Try again.")


def select_voice(default: str = DEFAULT_VOICE) -> str:
  """Let the user choose any Edge TTS voice. Exits the program on Esc."""
  voices = load_voices()

  if not (sys.stdin.isatty() and sys.stdout.isatty()):
    print(f"Non-interactive input; using voice: {default}")
    return default

  try:
    choice = pick_with_curses(voices, default)
  except ImportError:
    choice = pick_by_name(voices, default)

  if choice is None:
    print("Cancelled.")
    raise SystemExit(0)

  voice = next(v for v in voices if v["ShortName"] == choice)
  print(f"Selected voice: {choice} ({voice['Gender']}, {voice.get('LocaleName', voice['Locale'])})")
  return choice
