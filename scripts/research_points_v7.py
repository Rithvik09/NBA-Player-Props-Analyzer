"""Development-selected weighting/calibration experiment; exposed evaluation only."""
import json
from pathlib import Path
from datetime import datetime,timezone
import joblib,numpy as np,pandas as pd
from xgboost import XGBClassifier
from src.points_v5 import prepare
from src.points_v7 import MarketAnchoredPoints,anchored_probability,fit_anchor,mixture_weights
from src.verified_training import expand_points,classifier_frame
from src.prospective_points import save_new,digest
from scripts.train_selected_points import fit_calibration,paired_losses
from scripts.points_pilot import metrics
from scripts.train_points_v6 import candidate_groups
from scripts.investigate_points_features import PARAMS

ROOT=Path('logs/points_v7_research')

def run():
    obj=joblib.load(ROOT/'inputs.joblib');d=obj['data'];m=obj['masks'];market=obj['market'];x=prepare(d['X']);y=d['y']
    old={name:joblib.load('logs/expanded_points_experiment_v4/'+file) for name,file in [('frozen161','development_selected.joblib'),('frozen_compact','compact_expanded.joblib')]}
    groups=candidate_groups(old)
    plan={'created_at':datetime.now(timezone.utc).isoformat(),'fit_before':'2025-01-01','real_shares':[0,.01,.1,.5],
      'groups':groups,'duplicate_policy':'Remove any real fitting player/game from synthetic fitting rows before expansion. No new quote collection.',
      'calibration':'Fit no-intercept market-anchored shrinkage on March-April2025. Select weight/profile on May-June2025; refit shrinkage March-June only. No global intercept learned from pooled over frequency. Raw/sigmoid are diagnostic only.',
      'selection':'Include market-only fallback. Lowest selection log loss; exact ties prefer market, then fewer columns/lower real weight. Choices saved before evaluation. All research exposed, no promotion.',
      'data_sha256':digest('logs/real_points_merged_05/real_points.joblib'),'source_hashes':{p:digest(p) for p in [__file__,'src/points_v7.py','src/points_v5.py']}}
    save_new(ROOT/'experiment_plan.json',plan)
    parts=[joblib.load(f'logs/points_v5_data/{s}.joblib') for s in ['2022-23','2023-24','2024-25']]
    sx=pd.concat([p['X'] for p in parts],ignore_index=True);sy=np.concatenate([p['y'] for p in parts]);sd=pd.to_datetime(np.concatenate([p['dates'] for p in parts]));sg=np.concatenate([p['games'] for p in parts]);sp=np.concatenate([p['players'] for p in parts])
    realkeys=set(zip(d['games'][m['fit']],d['players'][m['fit']]))
    overlap=np.array([(g,p) in realkeys for g,p in zip(sg,sp)]);keep=(sd<'2025-01-01')&~overlap
    syn={'X':prepare(sx.loc[keep].reset_index(drop=True)),'y':sy[keep]}
    sx,sy,_,_=expand_points(syn,groups['repaired']);rx=classifier_frame(x,d['lines'],groups['repaired']);fitx=pd.concat([sx,rx.loc[m['fit']]],ignore_index=True).astype(np.float32);fity=np.r_[sy,y[m['fit']]]
    saved={};results={};options=[{'name':'market_only','score':metrics(y[m['selection']],market[m['selection']])['log_loss'],'columns':0,'share':0}]
    for name,cols in groups.items():
        for share in plan['real_shares']:
            tag=name+'_real'+str(share);weights=mixture_weights(len(sy),int(m['fit'].sum()),share)
            model=XGBClassifier(**{**PARAMS,'max_depth':3,'min_child_weight':50,'reg_lambda':50,'colsample_bytree':1},eval_metric='logloss')
            cc=cols+['decision_line','recent_minus_line','season_minus_line','standardized_edge']
            model.fit(fitx[cc],fity,sample_weight=weights)
            raw=model.predict_proba(rx[cc])[:,1];alpha=fit_anchor(raw[m['calfit']],market[m['calfit']],y[m['calfit']]);sig=fit_calibration('sigmoid',raw[m['calfit']],y[m['calfit']])
            result={'share':share,'alpha_calfit':alpha,'max_row_weight':float(weights.max()),'raw':{},'sigmoid':{},'anchored':{}}
            for split in ['fit','calfit','selection']:
                mask=m[split];result['raw'][split]=metrics(y[mask],raw[mask]);result['sigmoid'][split]=metrics(y[mask],sig.predict(raw[mask]));result['anchored'][split]=metrics(y[mask],anchored_probability(raw[mask],market[mask],alpha))
            score=result['anchored']['selection']['log_loss'];options.append({'name':tag,'score':score,'columns':len(cols),'share':share})
            results[tag]=result;saved[tag]=(model,cols,raw)
            print(tag,'raw selection',round(result['raw']['selection']['log_loss'],6),'anchor',round(score,6),'alpha',round(alpha,4),flush=True)
    chosen=min(options,key=lambda a:(round(a['score'],12),a['columns'],a['share']))
    # Also select a compact arm on the identical development criterion.
    compact=min([o for o in options if o['name']=='market_only' or o['name'].startswith('compact_retrained')],key=lambda a:(round(a['score'],12),a['columns'],a['share']))
    selection={'selected':chosen,'compact_control':compact,'options':options,'development':results,'removed_synthetic_overlaps':int(((sd<'2025-01-01')&overlap).sum()),'synthetic_fitting_player_games':int(keep.sum())}
    save_new(ROOT/'selection.json',selection)
    preds={'market_only':market,'coin_flip':np.full(len(y),.5)}
    for label,option in [('selected',chosen),('compact_control',compact)]:
        if option['name']=='market_only':preds[label]=market;continue
        model,cols,raw=saved[option['name']];cal=m['calfit']|m['selection'];alpha=fit_anchor(raw[cal],market[cal],y[cal]);candidate=MarketAnchoredPoints(cols,model,alpha)
        joblib.dump(candidate,ROOT/(label+'.joblib'));preds[label]=anchored_probability(raw,market,alpha)
    # Exposed evaluation is opened only after choices above are saved.
    for name in ['repaired','compact_retrained']:
        preds['v6_'+name]=joblib.load(f'logs/points_v6_real_01/{name}.joblib').predict_over(d['X'],d['lines'])
    report={'populations':{},'selected':chosen,'compact_control':compact,'promoted':False}
    for pop,mask in [('all',m['evaluation']),('eligible',m['evaluation']&(x.games_played.to_numpy()>=10)),('short',m['evaluation']&(x.games_played.to_numpy()<10))]:
        report['populations'][pop]={'n':int(mask.sum()),'games':len(set(d['games'][mask])),'arms':{name:metrics(y[mask],p[mask]) for name,p in preds.items()},'comparisons':{name:paired_losses(y[mask],preds['selected'][mask],preds[name][mask],d['games'][mask]) for name in ['compact_control','v6_repaired','market_only','coin_flip']}}
    pd.DataFrame({'game':d['games'],'player':d['players'],'date':d['dates'],'line':d['lines'],'over':y,**preds}).to_csv(ROOT/'predictions.csv',index=False)
    # Save all ablation evaluation scores only after selection, without using them to retune.
    save_new(ROOT/'ablation_evaluation.json',{name:{'raw':metrics(y[m['evaluation']],raw[m['evaluation']]),'anchored_calfit':metrics(y[m['evaluation']],anchored_probability(raw[m['evaluation']],market[m['evaluation']],results[name]['alpha_calfit']))} for name,(_,_,raw) in saved.items()})
    save_new(ROOT/'report.json',report);print('SELECTED',chosen,flush=True)

if __name__=='__main__':run()
