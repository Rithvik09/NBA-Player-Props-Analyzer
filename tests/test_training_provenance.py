from types import SimpleNamespace
import pandas as pd
from src.training_provenance import record_training_inputs


def test_training_digest_tracks_features_targets_and_order():
    a=SimpleNamespace(prop_type='points',player_id=1,game_id='001',game_date=pd.Timestamp('2025-01-02'),y=20.,minutes=30.,feature_policy='fixture',X=pd.DataFrame([{'recent_avg':18.}]))
    b=SimpleNamespace(**{**vars(a),'player_id':2,'y':15.})
    original=record_training_inputs([a,b],{'seed':42})
    assert original['examples_count']==2 and original['unique_player_games']==2
    assert original['examples_sha256']==record_training_inputs([a,b],{'seed':42})['examples_sha256']
    assert original['examples_sha256']!=record_training_inputs([b,a],{'seed':42})['examples_sha256']
    a.X.loc[0,'recent_avg']=19.
    assert original['examples_sha256']!=record_training_inputs([a,b],{'seed':42})['examples_sha256']
    a.X.loc[0,'recent_avg']=18.;a.y=21.
    assert original['examples_sha256']!=record_training_inputs([a,b],{'seed':42})['examples_sha256']
