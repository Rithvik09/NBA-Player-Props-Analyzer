"""Extract only pregame schedule/arena/broadcast facts from cached summaries."""
import json
from pathlib import Path

OUT=Path('logs/feature_recovery');records={}
for path in Path('logs/referee_cache').glob('*_summary.json'):
    raw=json.loads(path.read_text());summary=raw.get('boxScoreSummary',raw)
    gid=str(summary.get('gameId',''))
    if not gid:continue
    arena=summary.get('arena') or {};broadcast=summary.get('broadcasters') or {}
    value={'game_id':gid,'arena_id':arena.get('arenaId'),'arena_name':arena.get('arenaName'),'scheduled_at':summary.get('gameTimeUTC'),'source':str(path)}
    if 'nationalBroadcasters' in broadcast:value['national_tv']=bool(broadcast['nationalBroadcasters'] or broadcast.get('nationalOttBroadcasters'))
    records[gid]=value
(OUT/'game_metadata.json').write_text(json.dumps(records));print('game_metadata',len(records),'known national TV',sum('national_tv' in r for r in records.values()),'known arena',sum(bool(r['arena_id']) for r in records.values()))
