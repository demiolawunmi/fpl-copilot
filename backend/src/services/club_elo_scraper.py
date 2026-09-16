"""
HTML fallback scraper for ClubElo.

Used when ``api.clubelo.com`` (the CSV API) is unreachable — its outage modes
(hangs / 5xx) have historically left ``clubelo.com/ranking`` serving pages
normally. The page embeds a server-rendered standings table with one row per
club::

    <a href="/Arsenal"><span class="NonAst">ARS</span><span class="Ast">Arsenal</span></a>
    </td><td class="r">2005</td>

The href slug matches the CSV API's club naming (e.g. ``ManCity``, ``Forest``,
``Tottenham``), so results drop straight into the same name-resolution logic.
"""

from __future__ import annotations

import logging
import re

import requests

logger = logging.getLogger(__name__)

CLUBELO_RANKING_URL = "http://clubelo.com/ranking"

# Club row: link slug + optional display markup, followed by the Elo cell.
_ROW_RE = re.compile(
    r'<a href="/([A-Za-z]+)">'
    r'(?:<span class="NonAst">[A-Za-z0-9]+</span>)?'
    r"(?:<span class=\"Ast\">[^<]*</span>)?"
    r"</a></td><td class=\"r\">([0-9]+(?:\.[0-9]+)?)</td>"
)


def scrape_elo_ratings(timeout: float = 15.0) -> dict[str, float]:
    """Scrape ``{club_slug: elo}`` from the ClubElo rankings page.

    Raises on network failure or if no recognizable rows are found — callers
    should treat any exception as "fallback unavailable".
    """
    resp = requests.get(CLUBELO_RANKING_URL, timeout=timeout, headers={"User-Agent": "FPLCopilot/1.0"})
    resp.raise_for_status()

    # Skip rows inside the sortable-table bootstrap data (top-N only); the
    # server-rendered table below it contains every club.
    ratings: dict[str, float] = {}
    for slug, elo_str in _ROW_RE.findall(resp.text):
        try:
            ratings[slug] = float(elo_str)
        except ValueError:
            continue

    if len(ratings) < 20:
        raise ValueError(f"ClubElo scraper found only {len(ratings)} clubs")

    logger.info("ClubElo scraper: extracted %d clubs from %s", len(ratings), CLUBELO_RANKING_URL)
    return ratings
