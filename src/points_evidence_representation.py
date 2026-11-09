"""Research-only shooting reliability and complete injury evidence representation."""
import numpy as np
import pandas as pd

RAW_SHOOTING=['fg_pct_recent','fg3_pct_recent','ft_pct_recent','true_shooting_pct','effective_fg_pct','points_per_shot','fg3a_per_game','fta_per_game','fta_rate_player']
INJURY=[side+'_'+k for side in ('team','opp') for k in ['injury_unknown','injury_report_available','out_count','out_unresolved_count','out_unknown_role_count','known_key_players_out','key_players_out']]+['opp_injury_impact','games_since_return','missed_games_before_return']


def smoothed(makes,attempts,prior,strength):
    makes=np.asarray(makes,float);attempts=np.asarray(attempts,float)
    if (attempts<0).any() or strength<=0:raise ValueError('Invalid reliability inputs')
    return (makes+prior*strength)/(attempts+strength)


def complete_injury(frame):
    out=pd.DataFrame(index=frame.index)
    for side in ('team','opp'):
        valid=(frame[side+'_injury_report_available']==1)&(frame[side+'_out_unresolved_count']==0)&(frame[side+'_out_unknown_role_count']==0)
        for k in ['out_count','known_key_players_out']:
            out['qualified_'+side+'_'+k]=frame[side+'_'+k].where(valid)
    return out


def prior_counts(logs):
    cols=['FGM','FGA','FG3M','FG3A','FTM','FTA','PTS']
    pieces=[]
    for pid,part in logs.groupby('PLAYER_ID',sort=False):
        part=part.sort_values(['GAME_DATE','GAME_ID'])
        if pd.to_datetime(part.GAME_DATE).duplicated().any():raise ValueError('Multiple same-day appearances require explicit cutoff')
        out=pd.DataFrame({'game':part.GAME_ID.astype(str).to_numpy(),'player':int(pid),'date':pd.to_datetime(part.GAME_DATE).to_numpy()})
        for window in [5,20]:
            counts=part[cols].astype(float).shift(1).rolling(window,min_periods=1).sum().fillna(0)
            for col in cols:out[f'{col}_{window}']=counts[col].to_numpy()
            out[f'n_{window}']=np.minimum(np.arange(len(part)),window)
        pieces.append(out)
    return pd.concat(pieces,ignore_index=True)


def shooting_features(counts,priors):
    out=pd.DataFrame(index=counts.index)
    for key,made,attempt,strength in [('fg','FGM','FGA',50),('fg3','FG3M','FG3A',20),('ft','FTM','FTA',20)]:
        out['reliable_'+key]=smoothed(counts[made+'_5'],counts[attempt+'_5'],priors[key],strength)
        out['support_'+key]=np.log1p(counts[attempt+'_5'])
    fga=counts.FGA_20;poss=2*(fga+.44*counts.FTA_20)
    out['reliable_ts']=smoothed(counts.PTS_20,poss,priors['ts'],100)
    out['reliable_efg']=smoothed(counts.FGM_20+.5*counts.FG3M_20,fga,priors['efg'],50)
    out['support_efficiency']=np.log1p(fga)
    for col in ['FGA','FG3A','FTA']:out['volume_'+col.lower()]=counts[col+'_20']/counts.n_20.replace(0,np.nan)
    return out
