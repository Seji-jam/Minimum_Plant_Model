# Minimum Plant Model (MPM)

> **A lightweight, **_re‑configurable_** physiological processed based model that can adopted, forked, and extended**  
> The Minimum Plant Model (MPM) is a framework designed to simulate plant growth and development according to specific environmental conditions. It represents canopy photosynthesis, carbon assimilation, allocation, and the development of their organs through logistic growth curve. The architecture is designed to facilitate the repetition of simulations and the incorporation of new processes as required.

---
**Core Components**
```text
 carbon_pool ─┐                 ┌─► Leaf
              │  (thermal age)  │
              ├─► Priority Q ───┼─► Stem
              │                 │
              └─────────────────┴─► Root / Seed / …
```

1. Resource Pools (RPs)
An RP represents a distinct anatomical compartment (e.g., a leaf cohort, stem, root order, storage organ).
In summary each RP:
 Initiates growth when the plant’s accumulated thermal time reaches a user-defined threshold.
 Grows according to a three-parameter logistic function in its own thermal time.
 Reports carbon demand at each time step as the product of its current biomass, instantaneous relative growth rate, and thermal time increment.

2. Priority Queue (PQ)
The PQ determines the order in which RPs receive carbon from a central pool:

All initiated RPs enter the queue at each step.
RPs are sorted by an allocation priority (defined by user); demand is met in order until the carbon pool is depleted.
Priorities can be static or stage-dependent by subclassing the PQ.

3. Environment Abstraction
The Environment class encapsulates external drivers (e.g., air temperature, radiation, VPD, wind speed) and can include different subclasses depending on the plant environemnt (e.g., hydroponic). Such abstraction allows that aditional contexts (e.g., greenhouse, growth chamber) to be implemented without altering plant code.

4. Photosynthesis
Canopy carbon assimilation is computed using the biochemistry of Farquhar et al. (1980), scaled to the canopy via the sunlit–shaded leaf of De Pury and Farquhar (1997). Temperature dependencies of Vcmax and Jmax, together with a stomatal conductance model, are used to estimate intercellular CO₂ and net assimilation.



## Input Structure
MPM requires the following input files:

1. Driver data: time series of environmental variables at the model time step 
2. Global parameters: includes biophysical and physiological parameters of the plant
3. Resource pool parameters: one row per RP, specifying logistic growth parameters (i.e., initiation time, growth rate, and allocation priority).


---


## Project Layout

```
MPM_testing/
├── mpm/
│   ├── cli.py                  # Command‑line interface  →  python -m mpm.cli …
│   ├── plant/
│   │   ├── model_handler.py    # Orchestrates the run loop
│   │   ├── plant.py            # Plant composite object
│   │   ├── priority_queue.py   # Flexible allocation engine
│   │   ├── resource_pool.py    # Base logistic pool
│   │   └── carbon_assimilation.py
│   └── environment/
│       ├── atmosphere.py
│       ├── aboveground.py
│       └── base.py
├── data/                       # Sample drivers/params CSVs
├── tests/                      # Optional regression tests (pytest)
├── pyproject.toml              # PEP 621 config
└── README.md                   # You are here
```

---

## Quick Start

### 1  ·  Install

```bash
git clone https://github.com/Seji-jam/MPM_testing.git
cd MPM_testing
pip install -e .[dev]     # editable + dev extras
```

### 2  ·  Run a simulation

```bash
python -m mpm.cli  testdata/MPM_driver.csv testdata/parameters_CC.csv testdata/RP_CC.csv
```

*Shows plots and writes `model_outputs.csv`.*

Or from the script:

```python
from mpm.plant.model_handler import ModelHandler

sim = ModelHandler("MPM_driver.csv", "parameters_CC.csv", "RP_CC.csv")
sim.run_simulation()
sim.plot_outputs()
```
### 3  ·  Verify scientific parity (optional)

```bash
pytest        # compares output to tests/baseline/model_outputs.csv
```

---

---


## 📄 License

MPM is distributed under the [MIT License](LICENSE).

---

## 👥 Acknowledgements

Developed by **Diane Wang, Sajad Jamshidi, and Bryan Runck** as an open, minimal starting point for next‑generation crop simulators. Contributions and forks are welcome — feel free to open an issue or pull request!
