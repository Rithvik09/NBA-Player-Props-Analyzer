"""Validate and aggregate cached play-by-play into dated player observations."""
from pathlib import Path
import gzip,json
import pandas as pd
from src.pbp_features import analyze_game

OUT=Path('logs/feature_recovery')


def run():
    dates={}
    for path in Path('logs/points_pilot_cache').glob('players_*.json'):
        frame=pd.read_json(path,dtype={'GAME_ID':str})
        dates.update({str(g):str(pd.Timestamp(d).date()) for g,d in zip(frame.GAME_ID,frame.GAME_DATE)})
    rows=[];failed=[]
    for path in sorted((OUT/'pbp').glob('*.gz')):
        try:
            data=json.load(gzip.open(path));rows.extend(analyze_game(data,dates[data['game_id']]))
        except Exception as e:failed.append({'file':path.name,'error':str(e)})
    (OUT/'pbp_player_games.json').write_text(json.dumps(rows))
    report={'cached_games':len(list((OUT/'pbp').glob('*.gz'))),'player_rows':len(rows),'valid_point_rows':sum(r['points_valid'] for r in rows),'valid_shot_rows':sum(r['shots_valid'] for r in rows),'valid_stint_rows':sum(r['stints_valid'] for r in rows),'failed_games':failed}
    (OUT/'pbp_aggregation.json').write_text(json.dumps(report,indent=2));print(report)

if __name__=='__main__':run()
