import asyncio
import time

CLEAR_LINE = "\r\033[K"


def format_elapsed(seconds: float) -> str:
  minutes, seconds = divmod(int(seconds), 60)
  return f"{minutes:02}:{seconds:02}"


class FileProgress:
  """Single-line progress display for one SRT file, redrawn in place.

  0-99%  : lines generated + aligned
  99%    : merging
  100%   : done
  """

  def __init__(self, label: str, total: int) -> None:
    self.label = label
    self.total = total
    self.done = 0
    self.cached = 0
    self.retries = 0
    self.merging_started = False
    self.finished = False
    self.started = time.monotonic()

  def elapsed(self) -> str:
    return format_elapsed(time.monotonic() - self.started)

  def _show(self, text: str, final: bool = False) -> None:
    if final:
      # Stop the ticker from redrawing over the final line.
      self.finished = True

    print(f"{CLEAR_LINE}{self.label}: {text}", end="\n" if final else "", flush=True)

  async def tick(self, interval: float = 1.0) -> None:
    """Redraw periodically so the timer keeps moving between events
    (slow TTS lines, retries, merging). Run as a task; cancel when done."""
    while True:
      await asyncio.sleep(interval)
      self._render()

  def _render(self) -> None:
    if self.finished:
      return

    if self.merging_started:
      self._show(f" 99% merging... {self.elapsed()}")
    else:
      self._render_lines()

  def _render_lines(self) -> None:
    percent = self.done * 99 // self.total if self.total else 0
    details = [f"{self.done}/{self.total} lines"]

    if self.cached:
      details.append(f"{self.cached} cached")

    if self.retries:
      details.append(f"{self.retries} retries")

    self._show(f"{percent:3}% ({', '.join(details)}) {self.elapsed()}")

  def start(self) -> None:
    self._render_lines()

  def line_done(self, cached: bool = False) -> None:
    self.done += 1
    self.cached += cached
    self._render_lines()

  def retry(self) -> None:
    self.retries += 1
    self._render_lines()

  def merging(self) -> None:
    self.merging_started = True
    self._render()

  def finish(self, output: str, warnings: list[str]) -> None:
    summary = f"100% done in {self.elapsed()} -> {output}"

    if self.retries:
      summary += f" ({self.retries} retries)"

    if warnings:
      summary += f" ({len(warnings)} warnings)"

    self._show(summary, final=True)

    for warning in warnings:
      print(f"    WARNING {warning}")

  def fail(self, error: Exception) -> None:
    self._show(f"FAILED after {self.elapsed()}", final=True)
    print(f"    Error: {error}")
