"""Full-screen arrow-key menu with type-to-filter, built on curses (stdlib)."""

import os
from collections.abc import Callable, Sequence
from typing import TypeVar

Item = TypeVar("Item")

HELP = "Up/Down move | type to filter | Backspace erase | Enter select | Esc quit"


def contains_all_terms(items: Sequence[Item], query: str, describe) -> list[Item]:
  """Default filter: every typed word must appear in the item's text."""
  terms = query.lower().split()
  return [item for item in items if all(t in describe(item).lower() for t in terms)]


def pick(
  title: str,
  items: Sequence[Item],
  *,
  key: Callable[[Item], str],
  describe: Callable[[Item], str],
  default: str | None = None,
  filter_items: Callable[[Sequence[Item], str], list[Item]] | None = None,
  noun: str = "items",
) -> Item | None:
  """Show items in a curses menu and return the chosen one, or None on Esc.

  key identifies an item (used for the default and to keep the cursor in
  place while filtering); describe gives its display line. Raises ImportError
  when curses is unavailable so callers can fall back to a plain prompt.
  """
  import curses

  if filter_items is None:
    filter_items = lambda items, query: contains_all_terms(items, query, describe)  # noqa: E731

  # Make Esc respond immediately instead of after a 1s delay.
  os.environ.setdefault("ESCDELAY", "25")

  def run(screen) -> Item | None:
    curses.curs_set(0)
    curses.use_default_colors()
    screen.keypad(True)

    query = ""
    filtered = list(items)
    # The item the user last moved to; the cursor returns to it whenever it
    # is visible again (e.g. after clearing the filter).
    preferred = default
    cursor = next((i for i, item in enumerate(filtered) if key(item) == default), 0)
    top = 0

    while True:
      height, width = screen.getmaxyx()
      list_height = max(1, height - 4)

      # Keep the cursor visible.
      cursor = max(0, min(cursor, len(filtered) - 1))
      if cursor < top:
        top = cursor
      elif cursor >= top + list_height:
        top = cursor - list_height + 1

      screen.erase()
      put(screen, 0, f"{title}  ({HELP})", width, curses.A_BOLD)
      put(screen, 1, f"Filter: {query}", width)

      for row, item in enumerate(filtered[top : top + list_height]):
        index = top + row
        attr = curses.A_REVERSE if index == cursor else curses.A_NORMAL
        marker = ">" if index == cursor else " "
        put(screen, 2 + row, f"{marker} {describe(item)}", width, attr)

      if not filtered:
        put(screen, 2, f"  (no {noun} match)", width, curses.A_DIM)

      put(screen, height - 1, f"{len(filtered)}/{len(items)} {noun}", width, curses.A_DIM)
      screen.refresh()

      pressed = screen.get_wch()

      moves = {
        curses.KEY_UP: cursor - 1,
        curses.KEY_DOWN: cursor + 1,
        curses.KEY_PPAGE: cursor - list_height,
        curses.KEY_NPAGE: cursor + list_height,
        curses.KEY_HOME: 0,
        curses.KEY_END: len(filtered) - 1,
      }

      if pressed in moves:
        if filtered:
          cursor = max(0, min(moves[pressed], len(filtered) - 1))
          preferred = key(filtered[cursor])
        continue
      elif pressed in ("\n", "\r", curses.KEY_ENTER):
        if filtered:
          return filtered[cursor]
        continue
      elif pressed == "\x1b":
        return None
      elif pressed in (curses.KEY_BACKSPACE, "\x7f", "\b"):
        query = query[:-1]
      elif isinstance(pressed, str) and pressed.isprintable():
        query += pressed
      else:
        continue

      # Filter changed: stay on the preferred item if it still matches.
      filtered = filter_items(items, query)
      cursor = next((i for i, item in enumerate(filtered) if key(item) == preferred), 0)
      top = 0

  return curses.wrapper(run)


def put(screen, row: int, text: str, width: int, attr: int = 0) -> None:
  """Write one line, clipped so curses never errors at the screen edge."""
  try:
    screen.addnstr(row, 0, text, max(0, width - 1), attr)
  except Exception:
    pass
