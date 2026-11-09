"""Restore reference inputs at each actual quote cutoff; no outcomes used."""
import argparse,json
from pathlib import Path
import joblib
import pandas as pd
from src.historical_inputs import HistoricalInputs,load_season_logs
from src.points_reference_inputs import legacy_injuries,REFERENCE_FIELDS
from src.prospective_points import digest,save_new
from scripts.train_models import _parse_matchup,_team_id


def run(dataset,output):
    if output.exists():raise ValueError('Refuse to overwrite reference evidence')
    data=joblib.load(dataset); provenance=json.loads((dataset.parent/'provenance.json').read_text())
    source_quotes={}
    for source,expected in provenance['sources'].items():
        if digest(source)!=expected:raise ValueError('Upstream dataset drift')
        d=joblib.load(source);report=json.loads((Path(source).parent/'report.json').read_text())
        if len(report['quotes'])!=len(d['games']):raise ValueError('Quote row alignment mismatch')
        source_quotes[source]={(str(g),int(p)):q for g,p,q in zip(d['games'],d['players'],report['quotes'])}
    frame=data['X'].copy();contexts={};cache={};cutoffs=[]
    snapshot=Path('logs/expanded_points_v4/source_snapshot.joblib');raw=joblib.load(snapshot)
    for i,(g,p,day,source) in enumerate(zip(data['games'],data['players'],data['dates'],provenance['row_sources'])):
        date=pd.Timestamp(day);y=date.year if date.month>=10 else date.year-1;season=f'{y}-{str(y+1)[2:]}'
        if season not in contexts:
            logs=load_season_logs(season,include_postseason=True)
            contexts[season]=(HistoricalInputs(season,raw,logs,audited=True),{(str(r.GAME_ID),int(r.PLAYER_ID)):r for r in logs.itertuples()})
        inputs,lookup=contexts[season];row=lookup[(str(g),int(p))];own,opp,_=_parse_matchup(row.MATCHUP)
        quote=source_quotes[source][(str(g),int(p))];cutoff=pd.Timestamp(quote['captured_at']);cutoffs.append(cutoff.isoformat())
        key=(str(g),own,cutoff.isoformat())
        if key not in cache:
            inputs.prediction_game_id=str(g);inputs.prediction_cutoff=cutoff
            inputs.recovery._cutoff_cache={};inputs.recovery._injury_cache={}
            cache[key]=legacy_injuries(inputs.recovery,_team_id(own),_team_id(opp),date)
        for k,v in cache[key].items():frame.loc[frame.index[i],k]=v
    side={'X':frame,'dataset_sha256':digest(dataset),'source_snapshot_sha256':digest(snapshot),'quote_cutoffs':cutoffs,
          'source_hashes':{p:digest(p) for p in ['src/points_reference_inputs.py','src/recovered_features.py','src/historical_inputs.py']},
          **{k:data[k] for k in ('games','players','dates')}}
    output.parent.mkdir(exist_ok=True,parents=True);joblib.dump(side,output)
    changed={k:int(((frame[k]!=data['X'][k])&~(frame[k].isna()&data['X'][k].isna())).sum()) for k in REFERENCE_FIELDS}
    save_new(output.with_suffix('.json'),{'rows':len(frame),'changed_fields':changed,'dataset_sha256':digest(dataset),'reference_sha256':digest(output),'rule':'Only original three injury fields restored; same player/game/line and quote cutoff. Unknown legacy role fallback retained only for frozen references.'})
    print(changed)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',type=Path);p.add_argument('output',type=Path);a=p.parse_args();run(a.dataset,a.output)
