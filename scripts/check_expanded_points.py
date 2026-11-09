"""Check expanded cohort identity, old-cohort parity, and source coverage."""
import argparse,json
from pathlib import Path
import joblib,numpy as np,pandas as pd
from src.audited_features import SPECS,POLICY_VERSION
from src.verified_training import verified_columns
from scripts.train_verified_points import COMPACT


def run(directory):
    report={'feature_policy':POLICY_VERSION,'seasons':{},
            'schema_metadata_corrections':{'pie':'Signed ratio, not bounded [0,1]; values unchanged. NBA Stats glossary formula permits negative contributions.'}};counts=[]
    for season in ['2022-23','2023-24','2024-25','2025-26']:
        new=joblib.load(directory/f'audited_{season}.joblib');old=joblib.load(Path('logs/full_feature_audit')/f'audited_{season}.joblib')
        assert new['feature_policy']==POLICY_VERSION
        keys=pd.MultiIndex.from_arrays([new['players'],new['games']]);oldkeys=pd.MultiIndex.from_arrays([old['players'],old['games']])
        assert keys.is_unique and oldkeys.is_unique
        positions=keys.get_indexer(oldkeys);assert (positions>=0).all()
        assert new['original_cohort'].sum()==len(oldkeys)
        np.testing.assert_array_equal(new['y'][positions],old['y'])
        np.testing.assert_array_equal(pd.to_datetime(new['dates'][positions]),pd.to_datetime(old['dates']))
        # Compact inputs must be identical; repairs may not change its benchmark.
        pd.testing.assert_frame_equal(new['X'].iloc[positions][COMPACT].reset_index(drop=True),old['X'][COMPACT].reset_index(drop=True))
        changed={}
        for col in verified_columns():
            a=new['X'].iloc[positions][col].to_numpy();b=old['X'][col].to_numpy()
            n=int((~np.isclose(a,b,equal_nan=True)).sum())
            if n:changed[col]=n
        unexpected=set(changed)-{'usage_rate_with_star_out','team_key_players_out','opp_key_players_out','opp_injury_impact'}
        assert not unexpected,unexpected
        missing=new['X'][verified_columns()].isna().mean()
        fractions=[c for c in verified_columns() if SPECS[c].units=='fraction']
        violations=[c for c in fractions if ((new['X'][c]<0)|(new['X'][c]>1)).any()]
        assert not violations,violations
        for col in verified_columns():counts.append({'season':season,'feature':col,'missing_rate':float(missing[col]),'unique':int(new['X'][col].nunique()),'matched_changed_rows':changed.get(col,0)})
        report['seasons'][season]={'player_games':len(keys),'players':int(len(np.unique(new['players']))),
            'original_cohort_rows':len(oldkeys),'changed_matched_features':changed,'compact_inputs_identical':True,
            'active_all_missing':missing[missing==1].index.tolist(),'fraction_range_violations':violations}
    pd.DataFrame(counts).to_csv(directory/'feature_coverage.csv',index=False)
    (directory/'quality_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);run(p.parse_args().dataset)
