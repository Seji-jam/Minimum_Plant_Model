import pathlib, numpy as np, pandas as pd
from mpm.plant.model_handler import ModelHandler

BASE = pathlib.Path(__file__).parent / "baseline" / "model_outputs.csv"
TOL  = 1e-6    # tighten if FP64 partout

def test_against_baseline(tmp_path):
    drivers = BASE.parents[2] / "data" / "drivers.csv"
    params  = BASE.parents[2] / "data" / "params.csv"
    pools   = BASE.parents[2] / "data" / "resource_pools.csv"

    m = ModelHandler(drivers, params, pools)
    m.run_simulation()
    m.save_outputs_to_csv(tmp_path / "new.csv")

    old = pd.read_csv(BASE)
    new = pd.read_csv(tmp_path / "new.csv")
    np.testing.assert_allclose(new.values, old.values, rtol=0, atol=TOL)