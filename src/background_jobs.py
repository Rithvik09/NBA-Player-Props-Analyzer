from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


@dataclass
class AutoJobsConfig:
    enabled: bool
    db_path: str
    models_dir: str
    season: str

    # Scheduling (local time)
    precompute_hour: int
    precompute_minute: int
    train_hour: int
    train_minute: int

    # Training params
    train_max_players: int
    train_mode: str  # "batch" | "incremental" | "hybrid"
    train_seasons: str

    # Optional weekly batch training (used when train_mode="hybrid", or when explicitly enabled)
    batch_weekly_enabled: bool
    batch_weekday: int  # 0=Mon .. 6=Sun
    batch_hour: int
    batch_minute: int
    batch_max_players: int
    batch_seasons: str
    batch_min_interval_seconds: int

    # Cooldowns (safety)
    min_interval_seconds: int


def load_config(db_path: str, season: str, models_dir: str = "models") -> AutoJobsConfig:
    train_mode = os.environ.get("AUTO_JOBS_TRAIN_MODE", "incremental").strip().lower()
    return AutoJobsConfig(
        enabled=_env_bool("AUTO_JOBS", True),
        db_path=db_path,
        models_dir=os.environ.get("AUTO_JOBS_MODELS_DIR", models_dir),
        season=os.environ.get("AUTO_JOBS_SEASON", season),
        precompute_hour=_env_int("AUTO_JOBS_PRECOMPUTE_HOUR", 5),
        precompute_minute=_env_int("AUTO_JOBS_PRECOMPUTE_MINUTE", 15),
        train_hour=_env_int("AUTO_JOBS_TRAIN_HOUR", 6),
        train_minute=_env_int("AUTO_JOBS_TRAIN_MINUTE", 0),
        train_max_players=_env_int("AUTO_JOBS_TRAIN_MAX_PLAYERS", 200),
        train_mode=train_mode,
        train_seasons=os.environ.get("AUTO_JOBS_TRAIN_SEASONS", ""),
        batch_weekly_enabled=_env_bool("AUTO_JOBS_BATCH_WEEKLY", train_mode == "hybrid"),
        batch_weekday=_env_int("AUTO_JOBS_BATCH_WEEKDAY", 0),  # 0=Mon
        batch_hour=_env_int("AUTO_JOBS_BATCH_HOUR", 18),
        batch_minute=_env_int("AUTO_JOBS_BATCH_MINUTE", 0),
        batch_max_players=_env_int("AUTO_JOBS_BATCH_MAX_PLAYERS", _env_int("AUTO_JOBS_TRAIN_MAX_PLAYERS", 200)),
        batch_seasons=os.environ.get("AUTO_JOBS_BATCH_SEASONS", os.environ.get("AUTO_JOBS_TRAIN_SEASONS", "")),
        batch_min_interval_seconds=_env_int("AUTO_JOBS_BATCH_MIN_INTERVAL_SECONDS", 60 * 60 * 24 * 6),  # ~6d
        min_interval_seconds=_env_int("AUTO_JOBS_MIN_INTERVAL_SECONDS", 60 * 60 * 20),  # ~20h
    )


def _next_local_time(hour: int, minute: int) -> datetime:
    now = datetime.now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target = target + timedelta(days=1)
    return target


def _next_local_weekday_time(weekday: int, hour: int, minute: int) -> datetime:
    """
    Next occurrence of weekday at (hour:minute) in local time.
    weekday: 0=Mon .. 6=Sun
    """
    now = datetime.now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    days_ahead = (int(weekday) - int(target.weekday())) % 7
    target = target + timedelta(days=days_ahead)
    if target <= now:
        target = target + timedelta(days=7)
    return target


class BackgroundJobRunner:
    """
    Runs daily background jobs without blocking requests.

    - Precompute refresh: scripts/update_precomputed.py
    - Model training: scripts/train_models.py

    Uses subprocess to isolate long-running training work from the web worker thread.
    """

    def __init__(self, cfg: AutoJobsConfig, on_models_updated=None, on_precompute_updated=None):
        self.cfg = cfg
        self.on_models_updated = on_models_updated
        self.on_precompute_updated = on_precompute_updated

        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = False

        self._last_precompute_attempt = 0
        self._last_train_attempt = 0
        self._last_batch_attempt = 0
        self._precompute_in_flight = False
        self._train_in_flight = False

        # Last run summaries for status/debugging
        self._last_precompute_result: dict | None = None
        self._last_train_result: dict | None = None
        self._last_batch_result: dict | None = None

    def status(self) -> dict:
        """
        Best-effort job status snapshot (safe to call from request thread).
        """
        with self._lock:
            return {
                "enabled": bool(self.cfg.enabled),
                "train_mode": str(self.cfg.train_mode),
                "season": str(self.cfg.season),
                "models_dir": str(self.cfg.models_dir),
                "db_path": str(self.cfg.db_path),
                "precompute_in_flight": bool(self._precompute_in_flight),
                "train_in_flight": bool(self._train_in_flight),
                "last_precompute": self._last_precompute_result,
                "last_train": self._last_train_result,
                "last_batch": self._last_batch_result,
            }

    def start(self):
        if not self.cfg.enabled:
            return
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop = False
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

    def stop(self):
        with self._lock:
            self._stop = True

    def _run_loop(self):
        next_precompute = _next_local_time(self.cfg.precompute_hour, self.cfg.precompute_minute)
        next_train = _next_local_time(self.cfg.train_hour, self.cfg.train_minute)
        next_batch = (
            _next_local_weekday_time(self.cfg.batch_weekday, self.cfg.batch_hour, self.cfg.batch_minute)
            if self.cfg.batch_weekly_enabled
            else datetime.max
        )

        print(
            f"[auto-jobs] enabled. next precompute={next_precompute}, next train={next_train} "
            f"(season={self.cfg.season}, max_players={self.cfg.train_max_players}, train_mode={self.cfg.train_mode})"
        )
        if self.cfg.batch_weekly_enabled:
            print(
                f"[auto-jobs] weekly batch enabled. next batch={next_batch} "
                f"(weekday={self.cfg.batch_weekday}, max_players={self.cfg.batch_max_players})"
            )

        while True:
            with self._lock:
                if self._stop:
                    return

            now = datetime.now()
            now_ts = int(time.time())

            if now >= next_precompute:
                self._kick_precompute(now_ts)
                next_precompute = _next_local_time(self.cfg.precompute_hour, self.cfg.precompute_minute)

            if self.cfg.batch_weekly_enabled and now >= next_batch:
                self._kick_batch(now_ts)
                next_batch = _next_local_weekday_time(self.cfg.batch_weekday, self.cfg.batch_hour, self.cfg.batch_minute)

            # If train + batch are scheduled for the same minute, prefer weekly batch.
            # (kick_batch sets _train_in_flight and _kick_train will no-op for that tick)
            if now >= next_train:
                self._kick_train(now_ts)
                next_train = _next_local_time(self.cfg.train_hour, self.cfg.train_minute)

            # sleep until next event or a short heartbeat
            soonest = min(next_precompute, next_train, next_batch)
            sleep_s = max(5, min(300, int((soonest - datetime.now()).total_seconds())))
            time.sleep(sleep_s)

    def _kick_precompute(self, now_ts: int):
        with self._lock:
            if self._precompute_in_flight:
                return
            if now_ts - int(self._last_precompute_attempt) < int(self.cfg.min_interval_seconds):
                return
            self._precompute_in_flight = True
            self._last_precompute_attempt = now_ts

        def _run():
            started_at = int(time.time())
            try:
                print("[auto-jobs] precompute starting...")
                cmd = [
                    sys.executable,
                    "scripts/update_precomputed.py",
                    "--db",
                    self.cfg.db_path,
                ]
                cp = subprocess.run(cmd, check=False)
                finished_at = int(time.time())
                with self._lock:
                    self._last_precompute_result = {
                        "started_at": started_at,
                        "finished_at": finished_at,
                        "returncode": int(getattr(cp, "returncode", 0) or 0),
                    }
                if callable(self.on_precompute_updated):
                    self.on_precompute_updated()
                print("[auto-jobs] precompute done.")
            except Exception as e:
                print("[auto-jobs] precompute failed:", e)
                finished_at = int(time.time())
                with self._lock:
                    self._last_precompute_result = {
                        "started_at": started_at,
                        "finished_at": finished_at,
                        "returncode": None,
                        "error": str(e),
                    }
            finally:
                with self._lock:
                    self._precompute_in_flight = False

        threading.Thread(target=_run, daemon=True).start()

    def _kick_train(self, now_ts: int):
        with self._lock:
            if self._train_in_flight:
                return
            if now_ts - int(self._last_train_attempt) < int(self.cfg.min_interval_seconds):
                return
            self._train_in_flight = True
            self._last_train_attempt = now_ts

        def _run():
            started_at = int(time.time())
            try:
                mode = (self.cfg.train_mode or "incremental").strip().lower()
                if mode == "hybrid":
                    mode = "incremental"  # hybrid means daily incremental + weekly batch
                print(f"[auto-jobs] training starting... (mode={mode})")
                if mode == "incremental":
                    cmd = [
                        sys.executable,
                        "scripts/update_incremental_models.py",
                        "--season",
                        self.cfg.season,
                        "--max-players",
                        str(self.cfg.train_max_players),
                        "--models-dir",
                        self.cfg.models_dir,
                        "--db",
                        self.cfg.db_path,
                    ]
                else:
                    cmd = [
                        sys.executable,
                        "scripts/train_models.py",
                        "--season",
                        self.cfg.season,
                        "--max-players",
                        str(self.cfg.train_max_players),
                        "--models-dir",
                        self.cfg.models_dir,
                        "--db",
                        self.cfg.db_path,
                    ]
                    if self.cfg.train_seasons.strip():
                        cmd.extend(["--seasons", self.cfg.train_seasons.strip()])
                cp = subprocess.run(cmd, check=False)
                finished_at = int(time.time())
                with self._lock:
                    self._last_train_result = {
                        "mode": mode,
                        "started_at": started_at,
                        "finished_at": finished_at,
                        "returncode": int(getattr(cp, "returncode", 0) or 0),
                    }
                if callable(self.on_models_updated):
                    self.on_models_updated()
                print("[auto-jobs] training done.")
            except Exception as e:
                print("[auto-jobs] training failed:", e)
                finished_at = int(time.time())
                with self._lock:
                    self._last_train_result = {
                        "started_at": started_at,
                        "finished_at": finished_at,
                        "returncode": None,
                        "error": str(e),
                    }
            finally:
                with self._lock:
                    self._train_in_flight = False

        threading.Thread(target=_run, daemon=True).start()

    def _kick_batch(self, now_ts: int):
        """
        Weekly batch training job (full retrain + evaluation).
        """
        with self._lock:
            if self._train_in_flight:
                return
            if now_ts - int(self._last_batch_attempt) < int(self.cfg.batch_min_interval_seconds):
                return
            self._train_in_flight = True
            self._last_batch_attempt = now_ts

        def _run():
            started_at = int(time.time())
            try:
                seasons = (self.cfg.batch_seasons or self.cfg.train_seasons or "").strip()
                cmd = [
                    sys.executable,
                    "scripts/train_models.py",
                    "--max-players",
                    str(self.cfg.batch_max_players),
                    "--models-dir",
                    self.cfg.models_dir,
                    "--db",
                    self.cfg.db_path,
                ]
                if seasons:
                    cmd.extend(["--seasons", seasons])
                else:
                    cmd.extend(["--season", self.cfg.season])
                print(f"[auto-jobs] weekly batch starting... (seasons={seasons or self.cfg.season})")
                cp = subprocess.run(cmd, check=False)
                finished_at = int(time.time())
                with self._lock:
                    self._last_batch_result = {
                        "started_at": started_at,
                        "finished_at": finished_at,
                        "returncode": int(getattr(cp, "returncode", 0) or 0),
                        "seasons": seasons or self.cfg.season,
                        "max_players": int(self.cfg.batch_max_players),
                    }
                if callable(self.on_models_updated):
                    self.on_models_updated()
                print("[auto-jobs] weekly batch done.")
            except Exception as e:
                print("[auto-jobs] weekly batch failed:", e)
                finished_at = int(time.time())
                with self._lock:
                    self._last_batch_result = {
                        "started_at": started_at,
                        "finished_at": finished_at,
                        "returncode": None,
                        "error": str(e),
                    }
            finally:
                with self._lock:
                    self._train_in_flight = False

        threading.Thread(target=_run, daemon=True).start()


