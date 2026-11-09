"""Import pinned public raw quotes; strictly pregame, independent of frozen studies."""
import json
from pathlib import Path
import pandas as pd
from src.historical_inputs import load_season_logs
from src.historical_injuries import normalize_name
from src.prospective_points import digest,save_new
from scripts.build_real_points_training import run as build
ROOT=Path('logs/real_points_source_search')
OUT=Path('logs/real_points_public_01')


def pair_quotes(frame):
    rows=[];rejected=[]
    keys=['event_id','bookmaker','player_name','point','snapshot_ts','commence_time','home_team','away_team']
    for key,g in frame[frame.market=='player_points'].groupby(keys,dropna=False,sort=True):
        try:
            event,book,name,line,snapshot,start,home,away=key
            observed=pd.Timestamp(snapshot);tip=pd.Timestamp(start)
            if observed.tzinfo is None or tip.tzinfo is None or observed>=tip:raise ValueError('Not pregame at capture')
            if set(g.outcome_name)!={'Over','Under'} or len(g)!=2:raise ValueError('Ambiguous/missing paired sides')
            updates=[pd.Timestamp(v) for v in g.last_update]
            if any(pd.isna(t) or t.tzinfo is None or t>observed or (observed-t).total_seconds()>900 for t in updates):raise ValueError('Missing/future/stale source update')
            sides=g.set_index('outcome_name')
            rows.append({'game_id':str(event),'game_date':tip.tz_convert('America/New_York').date().isoformat(),
                         'player_name':name,'market':'player_points','bookmaker':book,'line':float(line),
                         'over_odds':float(sides.loc['Over','price']),'under_odds':float(sides.loc['Under','price']),
                         'captured_at':observed.isoformat(),'source':'public_oddsapi_archive',
                         'home_team':home,'away_team':away,'provider_start':start,'source_updates':[t.isoformat() for t in updates]})
        except (ValueError,TypeError,KeyError) as exc:rejected.append({'key':list(key),'reason':str(exc)})
    return rows,rejected


def run():
    OUT.mkdir(exist_ok=True)
    dest=OUT/'quotes.json'
    if not dest.exists():
        sources=sorted(ROOT.glob('oddsapi_player_props_*.csv'))
        quotes,rejections=pair_quotes(pd.concat([pd.read_csv(p) for p in sources],ignore_index=True))
        logs=load_season_logs('2025-26',include_postseason=True)
        logs['_name']=logs.PLAYER_NAME.map(normalize_name);logs['_date']=pd.to_datetime(logs.GAME_DATE)
        accepted=[]
        for q in quotes:
            m=logs[(logs['_name']==normalize_name(q['player_name']))&(logs['_date']==pd.Timestamp(q['game_date']))]
            if len(m)!=1:
                rejections.append({'player':q['player_name'],'day':q['game_date'],'reason':'Ambiguous NBA identity'});continue
            q['source_player_name']=q['player_name'];q['player_name']=m.iloc[0].PLAYER_NAME
            accepted.append(q)
        save_new(dest,{'props':accepted,'repository':'https://github.com/mostgood1/NBA-Betting',
                       'git_tree_sha':json.loads((ROOT/'mostgood1_NBA-Betting_tree.json').read_text())['sha'],
                       'raw_sha256':{str(p):digest(p) for p in sources}})
        save_new(OUT/'source_rejections.json',rejections)
        print('pregame paired bookmaker records',len(accepted),flush=True)
    if not (OUT/'training').exists():build(dest,OUT/'training')

if __name__=='__main__':run()
