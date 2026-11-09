"""Finite-output and fallback status contract for the live prediction response."""
import math


def core_input_error(features,line):
    try:
        values=[float(line)]+[float(features[k]) for k in ['recent_avg','season_avg','stddev']]
    except (KeyError,ValueError,TypeError):
        return 'missing_or_invalid_core_inputs'
    if not all(math.isfinite(v) for v in values) or values[0]<0 or values[3]<0:
        return 'nonfinite_or_negative_core_inputs'
    return None


def unavailable_prediction(line,reason):
    try:
        value=float(line)
        if not math.isfinite(value) or value<0:value=0.
    except (TypeError,ValueError):value=0.
    return {'over_probability':.5,'predicted_value':value,'recommendation':'PASS','confidence':'LOW','edge':0.,'prediction_mode':'unavailable','prediction_available':False,'fallback_reason':reason,'model_inference_succeeded':False,'pi80_lower':None,'pi80_upper':None}
