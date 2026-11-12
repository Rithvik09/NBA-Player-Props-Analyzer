import numpy as np
import pandas as pd
import pytest
from scripts.build_expanded_points import eligible
from src.recovered_features import teammate_absence_usage


def history():
    return pd.DataFrame({'PLAYER_ID':[1]*10,'TEAM_ID':[10]*10,
        'GAME_DATE':pd.date_range('2025-01-01',periods=10),'GAME_ID':[str(i) for i in range(10)],
        'MIN':[24]*10,'FGA':[10]*10,'FTA':[2]*10,'TOV':[1]*10})


def test_eligibility_uses_prior_appearances_and_minutes_only():
    h=history();assert eligible(h)
    assert not eligible(h.iloc[:9],True)
    h.MIN=3;assert not eligible(h);assert eligible(h,True)
    h['PTS']=10000;assert not eligible(h)


def test_star_absence_does_not_count_games_before_joining_team():
    h=history()
    star=pd.DataFrame({'PLAYER_ID':[2,2],'TEAM_ID':[20,10],
        'GAME_DATE':pd.to_datetime(['2024-12-31','2025-01-09']),'GAME_ID':['old','8']})
    assert teammate_absence_usage(h,star,2,10,'2025-01-11') is None
    # A same-team observation on Jan6 establishes membership for later games.
    star.loc[1,['GAME_DATE','GAME_ID']]=[pd.Timestamp('2025-01-06'),'5']
    expected=(10+.44*2+1)*48/24
    assert teammate_absence_usage(h,star,2,10,'2025-01-11')==pytest.approx(expected)


def test_star_transfer_away_and_player_other_team_games_are_excluded():
    h=history();h.loc[:6,'TEAM_ID']=20
    star=pd.DataFrame({'PLAYER_ID':[2,2],'TEAM_ID':[10,20],
        'GAME_DATE':pd.to_datetime(['2024-12-31','2025-01-07']),'GAME_ID':['old','elsewhere']})
    assert teammate_absence_usage(h,star,2,10,'2025-01-11') is None


def test_future_roster_and_current_outcome_cannot_establish_past_membership():
    h=history();star=pd.DataFrame({'PLAYER_ID':[2],'TEAM_ID':[10],
        'GAME_DATE':pd.to_datetime(['2025-01-11']),'GAME_ID':['future']})
    assert teammate_absence_usage(h,star,2,10,'2025-01-11') is None
    star.GAME_DATE=pd.to_datetime(['2024-12-31'])
    expected=teammate_absence_usage(h,star,2,10,'2025-01-11')
    poisoned=pd.concat([h,h.iloc[[-1]].assign(GAME_DATE=pd.Timestamp('2025-01-11'),FGA=9999)],ignore_index=True)
    assert teammate_absence_usage(poisoned,star,2,10,'2025-01-11')==expected


def test_matchup_missing_mean_repair_preserves_real_zeros_and_inputs():
    from src.points_candidate import normalize_points_inputs
    x=pd.DataFrame({'matchup_games':[0,2,np.nan],'vs_opp_avg_min':[0.,0.,0.],
                    'vs_opp_fg_pct':[0.,0.,0.],'head_to_head_avg':[20.,0.,15.]})
    before=x.copy(deep=True);fixed=normalize_points_inputs(x)
    assert np.isnan(fixed.loc[0,'vs_opp_avg_min'])
    assert np.isnan(fixed.loc[2,'vs_opp_fg_pct'])
    assert fixed.loc[1,'vs_opp_fg_pct']==0.
    pd.testing.assert_frame_equal(x,before)
    pd.testing.assert_frame_equal(normalize_points_inputs(fixed),fixed)
    np.testing.assert_array_equal(fixed.head_to_head_avg,x.head_to_head_avg)
    with pytest.raises(ValueError,match='Matchup count'):
        normalize_points_inputs(x.drop(columns='matchup_games'))


def test_expanded_profiles_keep_compact_and_deferred_exclusions():
    from scripts.train_expanded_points import profiles
    from src.verified_training import DEFERRED_DEFENDERS,verified_columns
    from scripts.train_verified_points import COMPACT
    importance=pd.DataFrame({'feature':['ft_rate','ft_rate','recent_avg','recent_avg'],
                             'permutation_log_loss_increase':[-.1,-.2,-.1,-.2]})
    arms,bad=profiles(importance,COMPACT+['ft_rate','team_key_players_out'])
    for name,cols in arms.items():
        assert set(COMPACT).issubset(cols)
        assert set(cols).issubset(verified_columns())
        assert not set(cols)&DEFERRED_DEFENDERS
        if 'pruned' in name:assert not (set(cols)-set(COMPACT))&set(bad)


def test_expanded_candidate_serialization_and_raw_input_parity(tmp_path):
    import joblib
    from xgboost import XGBClassifier,XGBRegressor
    from src.points_candidate import ExpandedPointsCandidate,normalize_points_inputs
    from src.verified_training import classifier_frame,ProbabilityCalibration
    x=pd.DataFrame({'matchup_games':[0,2]*6,'vs_opp_avg_min':[0.,25.]*6,
                    'recent_avg':np.arange(12)+5.,'season_avg':np.arange(12)+6.,'stddev':[3.]*12})
    cols=['vs_opp_avg_min','recent_avg','season_avg','stddev'];fixed=normalize_points_inputs(x)
    lines=np.arange(12)+5.5;y=np.array([0,1]*6)
    params=dict(n_estimators=3,max_depth=1,n_jobs=1,random_state=42)
    clf=XGBClassifier(**params).fit(classifier_frame(fixed,lines,cols),y)
    reg=XGBRegressor(**params).fit(fixed[cols],np.arange(12)+5)
    model=ExpandedPointsCandidate(cols,reg,clf,ProbabilityCalibration(),regressor_columns=cols)
    path=tmp_path/'model.joblib';joblib.dump(model,path);loaded=joblib.load(path)
    np.testing.assert_allclose(loaded.predict_over(x,lines),clf.predict_proba(classifier_frame(fixed,lines,cols))[:,1])
    np.testing.assert_allclose(loaded.predict(x),reg.predict(fixed[cols]))
    loaded.input_normalization='unsupported'
    with pytest.raises(ValueError,match='Unsupported'):loaded.predict(x)


def test_cohort_weights_preserve_original_and_do_not_discard_added_players():
    from scripts.train_expanded_points import cohort_weights,normalized_weights
    core=np.array([True,False,False,True])
    np.testing.assert_array_equal(cohort_weights(core,'original'),[1,0,0,1])
    w=cohort_weights(core,'expanded_weighted')
    np.testing.assert_allclose(w,[1,.25,.25,1])
    scaled=normalized_weights(w,np.ones(4,dtype=bool))
    assert scaled.mean()==pytest.approx(1)
    assert scaled[0]/scaled[1]==pytest.approx(4)
    with pytest.raises(ValueError):cohort_weights(core,'unsupported')


@pytest.mark.parametrize('method',['sigmoid','isotonic'])
def test_calibration_honors_training_weights(method):
    from scripts.train_selected_points import fit_calibration
    p=np.full(8,.5);y=np.array([0]*4+[1]*4);weights=np.where(y==1,3.,1.)
    cal=fit_calibration(method,p,y,weights)
    assert cal.predict([.5])[0]==pytest.approx(.75,abs=.001)


def test_expanded_profile_label_tracks_the_feature_policy(monkeypatch):
    from scripts import train_expanded_points as mod
    from src.audited_features import POLICY_VERSION
    assert mod.profile_name('candidate')=='expanded_points_'+POLICY_VERSION+'_candidate'
    # Derived, not merely correct today: the label was hardcoded to v4 and survived
    # the bump to v5, so pinning only the current string would pin the same bug.
    monkeypatch.setattr(mod,'POLICY_VERSION','point_in_time_v99')
    assert mod.profile_name('candidate')=='expanded_points_point_in_time_v99_candidate', \
        'the profile label does not follow POLICY_VERSION, so a bump leaves it stale'


def test_negative_official_pie_is_preserved_as_signed_ratio():
    from tests.test_full_feature_audit import fixture,build
    from src.audited_features import SPECS
    logs,raw,inputs=fixture()
    inputs.precomputed['advanced'][(2544,'2024-25')]['pie']=-.286
    values,gaps=build(logs,inputs)
    assert SPECS['pie'].units=='signed ratio'
    assert values['pie']==pytest.approx(-.286)
    assert 'pie' not in gaps
