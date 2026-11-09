"""Forensic audit of frozen v6 research outcomes; no model or source mutation."""
import json
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.metrics import log_loss,roc_auc_score
from src.points_v5 import prepare
from src.verified_training import classifier_frame
from src.prospective_points import digest,save_new
from src.historical_inputs import load_season_logs
from scripts.points_pilot import metrics

ROOT=Path('logs/points_v7_research')
DATA=Path('logs/real_points_merged_05/real_points.joblib')

def quotes_for(data):
    prov=json.loads((DATA.parent/'provenance.json').read_text());lookup={}
    for path,expected in prov['sources'].items():
        assert digest(path)==expected
        d=joblib.load(path);q=json.loads((Path(path).parent/'report.json').read_text())['quotes']
        assert len(q)==len(d['games'])
        lookup[path]={(str(g),int(p)):r for g,p,r in zip(d['games'],d['players'],q)}
    return [lookup[s][(str(g),int(p))] for s,g,p in zip(prov['row_sources'],data['games'],data['players'])]

def implied(odds):
    a=np.asarray(odds,dtype=float)
    out=np.empty_like(a);positive=a>0
    out[positive]=100/(a[positive]+100);out[~positive]=-a[~positive]/(100-a[~positive])
    return out

def run():
    ROOT.mkdir(exist_ok=False)
    d=joblib.load(DATA);q=quotes_for(d);x=prepare(d['X']);days=pd.to_datetime(d['dates'])
    masks={'fit':days<'2025-01-01','calfit':(days>='2025-03-01')&(days<'2025-05-01'),'selection':(days>='2025-05-01')&(days<'2025-07-01'),'evaluation':days>='2025-07-01'}
    over=implied([r['over_odds'] for r in q]);under=implied([r['under_odds'] for r in q]);market=over/(over+under)
    frame=pd.DataFrame({'game':d['games'],'player':d['players'],'date':d['dates'],'line':d['lines'],'points':d['points'],'over':d['y'],'market':market,'source':[r['source'] for r in q],'book':[r['bookmaker'] for r in q]})
    report={'splits':{},'models':{},'identity_errors':[],'synthetic_real_comparison':{}}
    for name,m in masks.items():
        report['splits'][name]={'n':int(m.sum()),'games':len(set(d['games'][m])),'over_rate':float(d['y'][m].mean()),'market':metrics(d['y'][m],market[m]),'short_history':int((x.games_played[m]<10).sum()),'source_counts':frame.source[m].value_counts().to_dict()}
    assert np.array_equal(d['y'],(d['points']>d['lines']).astype(int))
    for name in ['repaired','compact_retrained']:
        model=joblib.load(f'logs/points_v6_real_01/{name}.joblib');raw=model.classifier.predict_proba(classifier_frame(x,d['lines'],model.columns))[:,1];cal=model.predict_over(d['X'],d['lines'])
        frame[name+'_raw']=raw;frame[name]=cal
        report['models'][name]={'slope':model.calibrator.estimator.coef_.tolist(),'intercept':model.calibrator.estimator.intercept_.tolist(),'splits':{}}
        for split,m in masks.items():report['models'][name]['splits'][split]={'raw':metrics(d['y'][m],raw[m]),'calibrated':metrics(d['y'][m],cal[m]),'mean_raw':float(raw[m].mean()),'mean_calibrated':float(cal[m].mean()),'std_calibrated':float(cal[m].std())}
    for season in ['2024-25','2025-26']:
        logs=load_season_logs(season,include_postseason=True);lookup={(str(r.GAME_ID),int(r.PLAYER_ID)):r for r in logs.itertuples()}
        syn=joblib.load(f'logs/points_v5_data/{season}.joblib');index={(str(g),int(p)):i for i,(g,p) in enumerate(zip(syn['games'],syn['players']))}
        realids=[];synids=[]
        for i,(g,p) in enumerate(zip(d['games'],d['players'])):
            row=lookup.get((str(g),int(p)))
            if row is None:continue
            if float(row.PTS)!=float(d['points'][i]) or pd.Timestamp(row.GAME_DATE)!=days[i]:report['identity_errors'].append(i)
            if (str(g),int(p)) in index:realids.append(i);synids.append(index[(str(g),int(p))])
        a=x.iloc[realids].reset_index(drop=True);b=prepare(syn['X'].iloc[synids].reset_index(drop=True));cols=joblib.load('logs/points_v6_real_01/repaired.joblib').columns
        equal=np.isclose(a[cols].to_numpy(dtype=float),b[cols].to_numpy(dtype=float),equal_nan=True)
        differences={k:int(n) for k,n in zip(cols,(~equal).sum(axis=0)) if n}
        report['synthetic_real_comparison'][season]={'matched_rows':len(realids),'different_fields':differences}
    frame.to_csv(ROOT/'diagnostic_predictions.csv',index=False)
    save_new(ROOT/'diagnosis.json',report)
    joblib.dump({'data':d,'quotes':q,'market':market,'masks':masks},ROOT/'inputs.joblib')
    print('splits', {k:(v['n'],v['over_rate'],v['market']['log_loss']) for k,v in report['splits'].items()})
    for name,v in report['models'].items():print(name,'slope',v['slope'],'intercept',v['intercept'],{k:(p['raw']['log_loss'],p['calibrated']['log_loss'],p['mean_calibrated']) for k,p in v['splits'].items()})
    print('identity_errors',report['identity_errors']);print(report['synthetic_real_comparison'])

if __name__=='__main__':run()
