"""Live injury scraper with multi-source fallback.

Pulls team-level injury reports from ESPN (primary) and Rotowire (fallback).
Results are cached 30 min to keep scrapers polite.

Return shape
------------
    { team_abbr: [ { player, status, injury, return }, ... ] }

    status ∈ {'Out', 'Doubtful', 'Questionable', 'Probable', 'Day-to-Day', 'Unknown'}

``status_severity`` returns a numeric 0-1 score for feature injection.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from typing import Iterable

import requests
from bs4 import BeautifulSoup


def _normalise_name(raw: str) -> str:
    """Strip diacritics + non-alpha chars, lowercase — for fuzzy comparison."""
    decomposed = unicodedata.normalize("NFKD", raw or "")
    ascii_only = decomposed.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z]", "", ascii_only.lower())

from .api_cache import ttl_cache

log = logging.getLogger(__name__)

_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_0) AppleWebKit/537.36"

_SEVERITY = {
    "out":          1.00,
    "doubtful":     0.85,
    "questionable": 0.50,
    "probable":     0.15,
    "day-to-day":   0.40,
    "gtd":          0.40,
    "unknown":      0.25,
}


def status_severity(raw: str) -> float:
    """Map a status string to a 0-1 severity score."""
    s = (raw or "").strip().lower()
    for key, score in _SEVERITY.items():
        if key in s:
            return score
    return 0.0


# ---------------------------------------------------------------------------
# Source: ESPN
# ---------------------------------------------------------------------------

def _fetch_espn() -> dict[str, list[dict]]:
    """Scrape ESPN's NBA injuries page, grouped by team.

    ESPN lays out injuries by team section with a header like "Boston Celtics"
    followed by a table. If the layout changes this silently returns {} and
    the caller falls back.
    """
    url = "https://www.espn.com/nba/injuries"
    try:
        resp = requests.get(url, headers={"User-Agent": _UA}, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        log.warning("[injuries] ESPN fetch failed: %s", e)
        return {}

    soup = BeautifulSoup(resp.text, "html.parser")
    out: dict[str, list[dict]] = {}

    # Each team's injury list lives inside a <div class="Table__Title"> header
    # and a sibling table. Structure has varied over time, so we fall back to
    # a flatter parse: iterate team-titled sections if present, else dump all
    # table rows under 'unknown'.
    sections = soup.find_all("div", class_="ResponsiveTable")
    if not sections:
        # Flat fallback
        rows = soup.find_all("tr", class_="Table__TR")
        entries = _parse_espn_rows(rows)
        if entries:
            out["UNKNOWN"] = entries
        return out

    for section in sections:
        title_el = section.find("div", class_="Table__Title")
        team = title_el.get_text(strip=True) if title_el else "UNKNOWN"
        rows = section.find_all("tr", class_="Table__TR")
        entries = _parse_espn_rows(rows)
        if entries:
            out[team] = entries
    return out


def _parse_espn_rows(rows: Iterable) -> list[dict]:
    entries: list[dict] = []
    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 4:
            continue
        player = cells[0].get_text(strip=True)
        if not player or player.lower() == "name":
            continue
        status = cells[1].get_text(strip=True)
        injury = cells[2].get_text(strip=True) if len(cells) > 2 else ""
        ret = cells[3].get_text(strip=True) if len(cells) > 3 else ""
        entries.append({
            "player": player,
            "status": status,
            "injury": injury,
            "return": ret,
            "severity": status_severity(status),
        })
    return entries


# ---------------------------------------------------------------------------
# Source: Rotowire (fallback)
# ---------------------------------------------------------------------------

def _fetch_rotowire() -> dict[str, list[dict]]:
    url = "https://www.rotowire.com/basketball/tables/injury-report.php?team=ALL&pos=ALL"
    try:
        resp = requests.get(url, headers={"User-Agent": _UA}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        log.warning("[injuries] Rotowire fetch failed: %s", e)
        return {}
    except ValueError:
        log.warning("[injuries] Rotowire response not JSON")
        return {}

    out: dict[str, list[dict]] = {}
    for row in data if isinstance(data, list) else data.get("data", []):
        team = str(row.get("team") or "UNKNOWN").upper()
        status = str(row.get("status") or "")
        out.setdefault(team, []).append({
            "player": str(row.get("player") or ""),
            "status": status,
            "injury": str(row.get("injury") or ""),
            "return": str(row.get("returns") or ""),
            "severity": status_severity(status),
        })
    return out


# ---------------------------------------------------------------------------
# Public: cached, multi-source
# ---------------------------------------------------------------------------

@ttl_cache(ttl_seconds=1800, name="live_injuries_all_teams")
def fetch_live_injuries() -> dict[str, list[dict]]:
    """Return merged injury report, ESPN first, Rotowire fallback."""
    data = _fetch_espn()
    if not data or sum(len(v) for v in data.values()) == 0:
        log.info("[injuries] ESPN empty — falling back to Rotowire")
        data = _fetch_rotowire()
    return data or {}


def find_player_injury(player_name: str) -> dict | None:
    """Look up a single player by name (diacritic-insensitive substring match)."""
    target = _normalise_name(player_name)
    for team_entries in fetch_live_injuries().values():
        for entry in team_entries:
            norm = _normalise_name(entry.get("player", ""))
            if norm and (norm == target or target in norm or norm in target):
                return entry
    return None


def summarise_team(team_abbr_or_name: str) -> dict:
    """Aggregate: count by severity bucket, max severity."""
    target = team_abbr_or_name.strip().lower()
    hits: list[dict] = []
    for team, entries in fetch_live_injuries().items():
        if target in team.lower():
            hits.extend(entries)
    if not hits:
        return {"count": 0, "max_severity": 0.0, "out_count": 0,
                "questionable_count": 0, "players": []}
    out_count = sum(1 for e in hits if e["severity"] >= 0.85)
    q_count = sum(1 for e in hits if 0.35 <= e["severity"] < 0.85)
    return {
        "count": len(hits),
        "max_severity": max(e["severity"] for e in hits),
        "out_count": out_count,
        "questionable_count": q_count,
        "players": hits,
    }


# ---------------------------------------------------------------------------
# Feature-vector helpers — for inference-time risk-context injection and for
# the next training run to consume as model features.
# ---------------------------------------------------------------------------

INJURY_FEATURE_KEYS = [
    "team_severity_max",
    "team_out_count",
    "team_questionable_count",
    "team_total_injuries",
]


def team_severity_features(team_id: int | None,
                           prefix: str = "team_") -> dict:
    """Numeric injury features for a team, keyed for ML feature injection.

    Resolves ``team_id`` -> NBA full name -> injury summary, with safe
    fallbacks (returns zeroed features when nba_api or scrape fails).

    ``prefix`` is prepended to every key (default ``team_``); pass
    ``opp_`` to namespace opponent features alongside.
    """
    zero = {
        f"{prefix}severity_max": 0.0,
        f"{prefix}out_count": 0,
        f"{prefix}questionable_count": 0,
        f"{prefix}total_injuries": 0,
    }
    if team_id is None:
        return zero
    try:
        from nba_api.stats.static import teams as _teams_static
        info = _teams_static.find_team_name_by_id(int(team_id))
        if not info:
            return zero
        full = info.get("full_name") or info.get("nickname")
        if not full:
            return zero
        s = summarise_team(full)
        return {
            f"{prefix}severity_max":       float(s.get("max_severity", 0.0)),
            f"{prefix}out_count":          int(s.get("out_count", 0)),
            f"{prefix}questionable_count": int(s.get("questionable_count", 0)),
            f"{prefix}total_injuries":     int(s.get("count", 0)),
        }
    except Exception as e:  # noqa: BLE001 — best-effort fallback
        log.debug("[injuries] team_severity_features failed for %s: %s",
                  team_id, e)
        return zero


def matchup_injury_context(team_id: int | None,
                           opp_team_id: int | None) -> dict:
    """Combined team + opponent injury features for a matchup."""
    a = team_severity_features(team_id, prefix="team_")
    b = team_severity_features(opp_team_id, prefix="opp_")
    return {**a, **b}
