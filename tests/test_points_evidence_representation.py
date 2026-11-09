import numpy as np
import pandas as pd
from src.points_evidence_representation import prior_counts,smoothed,complete_injury


def test_rolling_features_exclude_current_and_future_outcomes():
    rows=pd.DataFrame({'PLAYER_ID':[1,1,1],'GAME_ID':['a','b','c'],'GAME_DATE':['2023-01-01','2023-01-03','2023-01-05'],**{k:[2,4,999] for k in ['FGM','FGA','FG3M','FG3A','FTM','FTA','PTS']}})
    a=prior_counts(rows);rows.loc[1,'PTS']=9999;b=prior_counts(rows)
    assert a.PTS_5.iloc[1]==b.PTS_5.iloc[1]==2
    assert a.n_20.tolist()==[0,1,2]


def test_zero_attempts_are_prior_not_zero_accuracy():
    assert smoothed([0],[0],.36,20)[0]==.36
    assert abs(smoothed([60],[100],.36,20)[0]-.6)<abs(.36-.6)


def test_partial_zero_injury_is_not_complete_zero():
    f=pd.DataFrame({side+'_'+k:v for side in ['team','opp'] for k,v in {'injury_report_available':[1,1,0],'out_unresolved_count':[0,1,0],'out_unknown_role_count':[0,0,0],'out_count':[0,2,0],'known_key_players_out':[0,0,0]}.items()})
    x=complete_injury(f)
    assert (x.iloc[0]==0).all()
    assert x.iloc[1:].isna().all().all()
