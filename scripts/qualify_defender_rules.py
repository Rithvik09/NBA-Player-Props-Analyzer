"""Select on October–December, qualify on January–April, deploy next season.

Never choose a rule using its qualification period. All candidate assignments
use strictly earlier games and latest earlier team membership. This evaluates
assignment accuracy, not incremental player-prop model usefulness.
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from src.defender_evidence import select_assignment

OUT=Path('logs/source_blocked')
RULES=[dict(last_games=n,half_life_days=h,min_share=s,min_margin=.15,min_possessions=20,regular_season_only=True)
       for n,h in [(None,None),(1,None),(3,None),(None,30),(None,90)]
       for s in [.4,.5,.6,.7]]


def summarize(rows):
    n=len(rows)
    if not n:return {'n':0,'accuracy':None,'minutes_baseline_accuracy':None,'wilson_lower':0}
    p=sum(r['correct'] for r in rows)/n
    lower=(p+1.96**2/(2*n)-1.96*np.sqrt(p*(1-p)/n+1.96**2/(4*n*n)))/(1+1.96**2/n)
    return {'n':n,'accuracy':p,'minutes_baseline_accuracy':sum(r['baseline_correct'] for r in rows)/n,
            'wilson_lower':float(lower)}


def qualify(records):
    qualifications={}
    for year in range(2022,2027):
        development=[r for r in records if f'{year-1}-10-01'<=r['date']<f'{year}-01-01']
        candidates=[summarize([r for r in development if r['rule_id']==i]) for i in range(len(RULES))]
        eligible=[i for i,s in enumerate(candidates) if s['n']>=100]
        chosen=max(eligible,key=lambda i:(candidates[i]['wilson_lower'],candidates[i]['n'],-i)) if eligible else None
        held=[r for r in records if r['rule_id']==chosen and f'{year}-01-01'<=r['date']<f'{year}-10-01']
        stats=summarize(held)
        qualifications[f'{year}-{str(year+1)[2:]}']={**stats,'rule_id':chosen,
            'rule':RULES[chosen] if chosen is not None else None,
            'development':candidates[chosen] if chosen is not None else None,
            'selected_before':f'{year}-01-01','last_outcome_date':max((r['date'] for r in held),default=f'{year}-01-01'),
            'qualified':bool(stats['n']>=100 and stats['accuracy']>=.5 and
                             stats['accuracy']>=stats['minutes_baseline_accuracy']+.05)}
    return qualifications


def run(promote=False):
    bundle=joblib.load(OUT/'defender_evidence.joblib')
    frame=bundle['rows'];frame=frame[frame.game_id.str.startswith('002')]
    logs=pd.concat([pd.read_json(p,dtype={'GAME_ID':str}) for p in Path('logs/points_pilot_cache').glob('players_*.json')])
    logs['GAME_DATE']=pd.to_datetime(logs.GAME_DATE)
    # Store earlier pairings as small numeric arrays; avoid a DataFrame scan
    # across every historical row for each candidate and target game.
    history={};records=[]
    for day,part in frame.groupby('game_date',sort=True):
        earlier=logs[logs.GAME_DATE<day].sort_values('GAME_DATE')
        latest=earlier.groupby('PLAYER_ID').tail(1).set_index('PLAYER_ID').TEAM_ID.to_dict()
        minutes=earlier.groupby('PLAYER_ID').tail(10).groupby('PLAYER_ID').MIN.mean()
        baseline={int(t):int(g.MIN.idxmax()) for t,g in pd.DataFrame({'MIN':minutes,'TEAM':pd.Series(latest)}).dropna().groupby('TEAM')}
        pending=[]
        for (gid,pid,opp),actual in part.groupby(['game_id','personIdOff','defTeamId']):
            key=(int(pid),int(opp));target=int(actual.groupby('personIdDef').partialPossessions.sum().idxmax())
            past=[g for g in history.get(key,[]) if (day-g[0]).days<=400]
            history[key]=past
            for i,rule in enumerate(RULES):
                subset=past[-rule['last_games']:] if rule['last_games'] else past
                if not subset:continue
                counts={};possessions=0.
                for date,pairs in subset:
                    weight=.5**((day-date).days/rule['half_life_days']) if rule['half_life_days'] else 1.
                    for defender,poss in pairs:
                        possessions+=poss;counts[defender]=counts.get(defender,0)+poss*weight
                chosen=select_assignment(counts,possessions,latest,opp,rule)
                if chosen is None:continue
                records.append({'date':str(day.date()),'game_id':gid,'player_id':int(pid),'rule_id':i,
                    'prediction':chosen,'actual':target,'correct':chosen==target,
                    'baseline_correct':baseline.get(int(opp))==target})
            pairs=[(int(d),float(p)) for d,p in actual.groupby('personIdDef').partialPossessions.sum().items()]
            pending.append((key,(day,pairs)))
        # All same-day outcomes are withheld until every forecast is computed.
        for key,game in pending:history.setdefault(key,[]).append(game)
    qualifications=qualify(records)
    (OUT/'defender_rule_research.json').write_text(json.dumps({'rules':RULES,'qualifications':qualifications,
        'protocol':'Select using Oct-Dec only; qualify Jan-Apr; deploy following season. Existing >=100, >=50%, baseline+5pp gates unchanged.'},indent=2))
    joblib.dump(records,OUT/'defender_rule_predictions.joblib')
    if promote:
        bundle['qualifications']=qualifications
        joblib.dump(bundle,OUT/'defender_evidence.joblib')
        (OUT/'defender_qualification.json').write_text(json.dumps({
            'qualifications':qualifications,'measured_games':bundle['rows'].game_id.nunique(),
            'matchup_rows':len(bundle['rows']),
            'rule':'Development Oct-Dec; untouched Jan-Apr qualification; deploy following season. Per-season selected rule; regular-season assignment history only.'},indent=2))
    print(json.dumps(qualifications,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--promote',action='store_true',help='Persist qualified season-specific rules in evidence bundle')
    run(parser.parse_args().promote)
