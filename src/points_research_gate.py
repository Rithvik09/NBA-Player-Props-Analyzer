"""Evidence requirements for future research promotion; never alters frozen studies."""
import numpy as np


def promotion_gate(*,independent,player_games,independent_games,market_interval,compact_interval):
    """A development-selected score is not independent validation evidence."""
    reasons=[]
    if not independent:reasons.append('No untouched independent validation')
    if player_games<100 or independent_games<20:reasons.append('Insufficient independent validation coverage')
    for name,interval in [('market',market_interval),('compact',compact_interval)]:
        a=np.asarray(interval,dtype=float)
        if a.shape!=(2,) or not np.isfinite(a).all() or a[0]>a[1] or a[1]>=0:
            reasons.append('No established log-loss improvement over '+name)
    return {'eligible_for_promotion':not reasons,'reasons':reasons}


def calibration_support(history_appearances,games,calibration_fit,calibration_selection):
    """Check history cohorts separately; pooled counts cannot certify warmup."""
    n=np.asarray(history_appearances);g=np.asarray(games);result={}
    for name,cohort in [('short_history',n<10),('eligible_history',n>=10)]:
        splits={}
        for split,mask in [('fit',calibration_fit),('selection',calibration_selection)]:
            m=cohort&np.asarray(mask,dtype=bool);rows=int(m.sum());count=len(set(g[m]))
            splits[split]={'player_games':rows,'independent_games':count,'supported':rows>=100 and count>=20}
        result[name]={'splits':splits,'supported':all(v['supported'] for v in splits.values())}
    return result
