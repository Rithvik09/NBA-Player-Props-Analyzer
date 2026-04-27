"""In-memory TTL cache for nba_api / odds calls.

Keyed by (args, sorted kwargs). Thread-safe. Used as a decorator:

    from src.api_cache import ttl_cache

    @ttl_cache(ttl_seconds=3600)
    def expensive(player_id, season):
        return nba_api_call(...)

Set env ``NBA_CACHE_DISABLE=1`` to bypass (useful in tests).
"""
from __future__ import annotations

import hashlib
import os
import pickle
import threading
import time
from functools import wraps
from typing import Any, Callable, Dict, Tuple

_DISABLED = os.environ.get("NBA_CACHE_DISABLE", "").strip() in ("1", "true", "yes")

# Registry of caches by name for /cache/stats introspection
_CACHES: Dict[str, "_TTLCache"] = {}


def _hash_key(args: tuple, kwargs: dict) -> str:
    """Stable hash of call args."""
    try:
        payload = pickle.dumps((args, sorted(kwargs.items())), protocol=4)
    except Exception:
        payload = repr((args, sorted(kwargs.items()))).encode()
    return hashlib.blake2s(payload, digest_size=16).hexdigest()


class _TTLCache:
    __slots__ = ("ttl", "data", "lock", "hits", "misses", "name",
                 "key_locks", "key_lock_guard", "disk_path",
                 "stampede_blocks")

    def __init__(self, ttl_seconds: float, name: str,
                 disk_path: str | None = None):
        self.ttl = float(ttl_seconds)
        self.data: Dict[str, Tuple[Any, float]] = {}
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.stampede_blocks = 0  # # of waits saved by per-key serialisation
        self.name = name
        # Per-key locks: a second thread asking for the same key while the
        # first is still computing it should wait, then return the cached
        # value rather than firing a duplicate upstream request.
        self.key_locks: Dict[str, threading.Lock] = {}
        self.key_lock_guard = threading.Lock()
        self.disk_path = disk_path
        if disk_path:
            self._load_disk()

    # ----- in-memory ------------------------------------------------------

    def get(self, key: str):
        now = time.time()
        with self.lock:
            entry = self.data.get(key)
            if entry is None:
                self.misses += 1
                return None
            val, expiry = entry
            if now >= expiry:
                del self.data[key]
                self.misses += 1
                return None
            self.hits += 1
            return val

    def set(self, key: str, value) -> None:
        expiry = time.time() + self.ttl
        with self.lock:
            self.data[key] = (value, expiry)
        if self.disk_path:
            self._persist_one(key, value, expiry)

    def clear(self) -> None:
        with self.lock:
            self.data.clear()
        if self.disk_path:
            try:
                if os.path.exists(self.disk_path):
                    os.remove(self.disk_path)
            except OSError:
                pass

    def stats(self) -> dict:
        with self.lock:
            return {
                "name": self.name,
                "size": len(self.data),
                "hits": self.hits,
                "misses": self.misses,
                "stampede_blocks": self.stampede_blocks,
                "ttl_s": self.ttl,
                "disk_backed": self.disk_path is not None,
            }

    # ----- per-key locking (stampede protection) -------------------------

    def acquire_key_lock(self, key: str) -> threading.Lock:
        """Return (creating if needed) a lock for ``key``.

        Caller is responsible for ``with`` semantics. Two threads racing for
        the same uncached key serialise here, so only one fires the upstream
        call; the second sees the cached result on its next ``get``.
        """
        with self.key_lock_guard:
            lk = self.key_locks.get(key)
            if lk is None:
                lk = threading.Lock()
                self.key_locks[key] = lk
        return lk

    def note_stampede_block(self) -> None:
        with self.lock:
            self.stampede_blocks += 1

    # ----- disk persistence ----------------------------------------------

    def _load_disk(self) -> None:
        """Load on-disk snapshot, dropping any expired entries."""
        if not self.disk_path or not os.path.exists(self.disk_path):
            return
        try:
            with open(self.disk_path, "rb") as f:
                blob = pickle.load(f)
            if not isinstance(blob, dict):
                return
            now = time.time()
            with self.lock:
                for k, (v, exp) in blob.items():
                    if exp > now:
                        self.data[k] = (v, exp)
        except (OSError, pickle.UnpicklingError, EOFError):
            # Corrupted snapshot — start fresh
            try:
                os.remove(self.disk_path)
            except OSError:
                pass

    def _persist_one(self, key: str, value: Any, expiry: float) -> None:
        """Re-dump full cache to disk. Atomic via tmp + rename."""
        try:
            os.makedirs(os.path.dirname(self.disk_path) or ".", exist_ok=True)
            with self.lock:
                snapshot = dict(self.data)
            tmp = self.disk_path + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(snapshot, f, protocol=4)
            os.replace(tmp, self.disk_path)
        except (OSError, pickle.PicklingError):
            pass  # best-effort


_DISK_CACHE_DIR = os.environ.get(
    "NBA_CACHE_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 ".cache", "api"),
)


def ttl_cache(ttl_seconds: float = 3600, name: str | None = None,
              disk: bool = False):
    """Decorator — cache function return values for ``ttl_seconds``.

    ``None`` returns are NOT cached (so transient errors retry).

    Concurrent callers asking for the same uncached key serialise on a
    per-key lock — only one upstream call fires, the rest reuse its result.

    Set ``disk=True`` for long-TTL items that should survive process
    restart. Snapshot persists at ``$NBA_CACHE_DIR/<name>.pkl``. Defaults
    to ``<repo>/.cache/api/``.
    """
    def decorator(fn: Callable):
        cache_name = name or f"{fn.__module__}.{fn.__qualname__}"
        disk_path = None
        if disk:
            safe = cache_name.replace("/", "_").replace(":", "_")
            disk_path = os.path.join(_DISK_CACHE_DIR, f"{safe}.pkl")
        cache = _TTLCache(ttl_seconds, cache_name, disk_path=disk_path)
        _CACHES[cache_name] = cache

        @wraps(fn)
        def wrapper(*args, **kwargs):
            if _DISABLED:
                return fn(*args, **kwargs)
            key = _hash_key(args, kwargs)
            hit = cache.get(key)
            if hit is not None:
                return hit
            # Stampede protection: serialise concurrent misses on the same key
            klock = cache.acquire_key_lock(key)
            with klock:
                # Re-check inside the lock — another thread may have filled it
                hit = cache.get(key)
                if hit is not None:
                    cache.note_stampede_block()
                    return hit
                result = fn(*args, **kwargs)
                if result is not None:
                    cache.set(key, result)
                return result

        wrapper.cache_clear = cache.clear  # type: ignore[attr-defined]
        wrapper.cache_stats = cache.stats  # type: ignore[attr-defined]
        return wrapper

    return decorator


def all_stats() -> list[dict]:
    """Snapshot of every registered cache."""
    return [c.stats() for c in _CACHES.values()]


def clear_all() -> None:
    for c in _CACHES.values():
        c.clear()


# ---------------------------------------------------------------------------
# Shared cached fetchers for hot nba_api endpoints.
#
# Rationale: ``analyze_prop_bet`` is called once per (player × prop) per
# request. The scoreboard + series-state + player-info calls are identical
# across every prop in one request, so caching them collapses ~14× redundant
# API hits down to one.
# ---------------------------------------------------------------------------

@ttl_cache(ttl_seconds=300, name="scoreboard_v2")
def fetch_scoreboard_v2(game_date: str):
    """Cached ScoreboardV2 call. 5-min TTL — live scores change intra-day."""
    from nba_api.stats.endpoints import ScoreboardV2  # local import: heavy
    return ScoreboardV2(game_date=game_date).get_data_frames()[0]


@ttl_cache(ttl_seconds=1800, name="league_game_finder_playoffs")
def fetch_playoff_games(team_id: int, opp_team_id: int, season: str | None = None):
    """Cached LeagueGameFinder for playoff head-to-head. 30-min TTL."""
    from nba_api.stats.endpoints import LeagueGameFinder
    kwargs = dict(
        team_id_nullable=team_id,
        vs_team_id_nullable=opp_team_id,
        season_type_nullable="Playoffs",
    )
    if season:
        kwargs["season_nullable"] = season
    return LeagueGameFinder(**kwargs).get_data_frames()[0]


@ttl_cache(ttl_seconds=86400, name="common_player_info", disk=True)
def fetch_player_info(player_id: int):
    """Cached CommonPlayerInfo. 24-h TTL — height/weight/position rarely change."""
    from nba_api.stats.endpoints import CommonPlayerInfo
    return CommonPlayerInfo(player_id=player_id).get_data_frames()[0]


@ttl_cache(ttl_seconds=900, name="player_game_log")
def fetch_player_game_log(player_id: int, season: str, season_type: str = "Regular Season"):
    """Cached PlayerGameLog. 15-min TTL during live games, fine otherwise."""
    from nba_api.stats.endpoints import playergamelog
    return playergamelog.PlayerGameLog(
        player_id=player_id,
        season=season,
        season_type_all_star=season_type,
        timeout=60,
    ).get_data_frames()[0]
