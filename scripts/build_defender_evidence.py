"""Aggregate real matchups and test the fixed defender rule before each season."""
from pathlib import Path
import gzip,json
import joblib
import pandas as pd
from src.defender_evidence import expected_defender

OUT=Path('logs/source_blocked')


def run():
    rows=[]
    for path in sorted((OUT/'matchups').glob('*.gz')):
        game=json.load(gzip.open(path))
        rows.extend({**r,'game_id':game['game_id'],'game_date':game['game_date']} for r in game['rows'])
    frame=pd.DataFrame(rows);frame['game_date']=pd.to_datetime(frame.game_date)
    logs=pd.concat([pd.read_json(p,dtype={'GAME_ID':str}) for p in Path('logs/points_pilot_cache').glob('players_*.json')])
    logs['GAME_DATE']=pd.to_datetime(logs.GAME_DATE)
    by_player={int(pid):r.sort_values('GAME_DATE') for pid,r in logs.groupby('PLAYER_ID')}
    off={int(pid):r for pid,r in frame.groupby('personIdOff')};results=[]
    # Each label is used only for scoring an earlier-data-only forecast.
    for day,part in frame.groupby('game_date',sort=True):
        earlier=logs[logs.GAME_DATE<day].sort_values('GAME_DATE')
        latest=earlier.groupby('PLAYER_ID').tail(1).set_index('PLAYER_ID').TEAM_ID.to_dict()
        minutes=earlier.groupby('PLAYER_ID').tail(10).groupby('PLAYER_ID').MIN.mean()
        baseline={int(tid):int(g.MIN.idxmax()) for tid,g in
            pd.DataFrame({'MIN':minutes,'TEAM':pd.Series(latest)}).dropna().groupby('TEAM')}
        for (gid,pid,opp),actual in part.groupby(['game_id','personIdOff','defTeamId']):
            chosen=expected_defender(off[int(pid)],player_id=pid,opponent_id=opp,day=day,latest_teams=latest)
            if chosen is None:continue
            target=int(actual.groupby('personIdDef').partialPossessions.sum().idxmax())
            results.append({'date':str(day.date()),'game_id':gid,'player_id':int(pid),
                'prediction':chosen,'actual':target,'correct':chosen==target,'baseline_correct':baseline.get(int(opp))==target})
    qualifications={}
    for year in range(2022,2027):
        season=f'{year}-{str(year+1)[2:]}';prior=[r for r in results if
            r['game_id'].startswith('002') and f'{year-1}-10-01'<=r['date']<f'{year}-10-01']
        n=len(prior);accuracy=sum(r['correct'] for r in prior)/n if n else None
        baseline=sum(r['baseline_correct'] for r in prior)/n if n else None
        qualifications[season]={'n':n,'accuracy':accuracy,'minutes_baseline_accuracy':baseline,
            'last_outcome_date':max((r['date'] for r in prior),default=f'{year-1}-10-01'),
            'qualified':bool(n>=100 and accuracy>=.5 and accuracy>=baseline+.05)}
    bundle={'rows':frame,'qualifications':qualifications,
            'switches_available':bool((frame.switchesOn>0).any())}
    joblib.dump(bundle,OUT/'defender_evidence.joblib')
    (OUT/'defender_qualification.json').write_text(json.dumps({'qualifications':qualifications,
        'measured_games':frame.game_id.nunique(),'matchup_rows':len(frame),
        'rule':'Prior400-day matchup possessions; min20, leading share>=40%, margin>=15%; current opponent membership from latest earlier appearance. Qualification uses preceding season only.'},indent=2))
    (OUT/'defender_backtest.json').write_text(json.dumps(results))
    print(qualifications,flush=True)

if __name__=='__main__':run()
