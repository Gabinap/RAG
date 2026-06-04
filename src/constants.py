"""Shared ANSI colour palette and display utilities."""

from dataclasses import dataclass

# ── Raw ANSI codes ───────────────────────────────────────────────────────────
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"

# ── Semantic aliases ─────────────────────────────────────────────────────────
COLOR_SUCCESS = GREEN
COLOR_WARNING = YELLOW
COLOR_ERROR = RED
COLOR_INFO = BLUE
COLOR_ACCENT = CYAN
COLOR_TITLE = MAGENTA

# ── Layout ───────────────────────────────────────────────────────────────────
DIVIDER_WIDTH = 56
BAR_WIDTH = 24


def colorize(text: str, *codes: str) -> str:
    """Wrap text with one or more ANSI codes and reset at the end.

    Args:
        text: The text to colorize.
        *codes: One or more ANSI escape codes to apply (left to right).

    Returns:
        The text wrapped in the given codes with a reset suffix.
    """
    return "".join(codes) + text + RESET


def divider(thin: bool = False, color: str = COLOR_ACCENT) -> str:
    """Return a full-width coloured divider string.

    Args:
        thin: If True, returns a dimmed thin line using the given color.
        color: ANSI code for the divider color.

    Returns:
        A coloured divider string ready to print.
    """
    if thin:
        return colorize("─" * DIVIDER_WIDTH, DIM + color)
    return colorize("━" * DIVIDER_WIDTH, color)


# ── Theme palettes ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Palette:
    """A cohesive colour scheme for a dataset type."""

    name: str
    icon: str
    border: str
    title: str
    section: str
    value: str
    dim_section: str


DOCS_PALETTE = Palette(
    name="DOCS",
    icon="📄",
    border=CYAN,
    title=BLUE,
    section=CYAN,
    value=BLUE,
    dim_section=CYAN,
)

CODE_PALETTE = Palette(
    name="CODE",
    icon="💻",
    border=MAGENTA,
    title=YELLOW,
    section=MAGENTA,
    value=YELLOW,
    dim_section=MAGENTA,
)
