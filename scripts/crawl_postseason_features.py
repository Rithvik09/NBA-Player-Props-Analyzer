"""Recover postseason game logs so intensity features need not be dead flags."""
from pathlib import Path
import json
from nba_api.stats.endpoints import leaguegamelog

OUT=Path('logs/points_pilot_cache')
for season in ('2022-23','2023-24','2024-25','2025-26'):
    for kind,label in [('Playoffs','playoffs'),('PlayIn','playin')]:
        path=OUT/f'{label}_{season}.json'
        if path.exists():continue
        try:
            frame=leaguegamelog.LeagueGameLog(player_or_team_abbreviation='P',season=season,season_type_all_star=kind,timeout=15).get_data_frames()[0]
            if frame.empty:raise ValueError('Empty '+kind)
            path.write_text(frame.to_json(orient='records',date_format='iso'));print(season,kind,len(frame),frame.GAME_ID.iloc[0],flush=True)
        except Exception as exc:print(season,kind,type(exc).__name__,str(exc)[:100],flush=True)
