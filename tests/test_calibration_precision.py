import numpy as np
from src.points_calibration_research_v2 import probability,fit


def test_float32_input_retains_small_parameter_perturbations():
    raw=np.array([.2,.8],dtype=np.float32)
    a=probability(raw,[0.,1.]);b=probability(raw,[1e-8,1.])
    assert a.dtype==np.float64
    assert (b>a).all()


def test_calibration_precision_independent_of_input_storage():
    raw=np.linspace(.05,.95,100,dtype=np.float32);y=(np.arange(100)%3==0).astype(int)
    for kind in ['identity','intercept','intercept_slope']:
        assert np.allclose(fit(raw,y,kind),fit(raw.astype(float),y,kind),atol=1e-10)
