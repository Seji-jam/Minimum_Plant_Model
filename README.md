# Minimum Plant Model (MPM)

> **A lightweight, **_re‑configurable_** physiological based you can adopt, fork, and extend**  
> The Minimum Plant Model (MPM) is a simulation framework for modeling plant growth and development under different environmental conditions. It simulates carbon assimilation, resource allocation, and growth of different plant components (resource pools) over time based on environmental drivers like temperature, radiation, and other climate variables.

---

## 🌱 Why another crop model?

Conventional crop models are typically difficult to extend, codebase has not been optimally developed with hard wired a **fixed set of organs** (leaf, stem, root…)
**MPM emphasizes on flexibility and modularity** It is a *base engine* that trades complexity for **flexibility**:

| Design choice                 | What it means          |
| ----------------------------- | ---------------------- |
| **Priority‑queue allocation** | Carbon (and any nutrients) move through a queue – simply re‑order or re‑prioritise to create *new* architectures without touching core maths. |
| **Pluggable _Resource Pools_**| Each organ is a class with its own logistic growth curve. |
| **Environment abstraction**   | Above‑ and below‑ground conditions are objects. We can now swap a greenhouse light model for an outdoor PAR estimator, reuse the same plant code. |


The goal: **Provide the smallest credible core** upon which we can layer
canopy energy balance, water stress, genotype parameters, or a GUI – _without
fork‑lift refactoring_.

---

## ✨ Feature Highlights

* 🍃 **Sunlit / shaded Farquhar photosynthesis** (absorbed‑PAR based).
* 🌡️ **Thermal‑time growth** – logistic per organ, thermally integrated per plant.
* 🔄 **Priority queue carbon allocation** configurable per growth stage or user rule.
* 🧩 **Resource‑pool plug‑ins** – any number, any order, any parameter set.
* 🏞️ **Environment wrappers** – drop‑in replacements for greenhouse, field, or growth‑chamber scenarios.
* 🖼️ **Matplotlib quick‑look plots** + CSV export for downstream dashboards.
* 🧪 **Regression‑test harness** – check parity with a baseline run.

---

## 📦 Project Layout

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

## 🚀 Quick Start

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

## 🧬 Core Concepts

### Priority Queue Allocation

```text
 carbon_pool ─┐                 ┌─► Leaf
              │  (thermal age)  │
              ├─► Priority Q ───┼─► Stem
              │                 │
              └─────────────────┴─► Root / Seed / …
```

* Pools **enter** the queue once their initiation thermal time is reached.  
* A simple `sorted()` call allocates C based on `growth_allocation_priority`.  
* Override or subclass `PriorityQueue` for fancier rules (e.g. stage‑specific priorities, sink‑strength feedback).

### Resource Pools

```python
ResourcePool(
    name="L3",
    thermal_time_initiation=120,  # °C·day
    allocation_priority=1,        # low number ▶ high priority
    max_size=1.2,                 # g C
    initial_size=0.01,
    rate=0.04                     # logistic r
)
```

A pool **knows its own logistic curve** and reports demand each time step.

### Environment Abstraction

* `Atmosphere` → solar geometry only.  
* `AbovegroundEnvironment` → light partitioning, PAR absorption.  
* Want growth‑chamber pulses or PAR sensor data? Sub‑class `Environment` and feed it directly to the plant.

---

## 🔧 Extending MPM

| You want to… | Do this |
|--------------|---------|
| Add a new organ (e.g. tubers) | Sub‑class `ResourcePool`, list it in the `resource_pools.csv`. |
| Change allocation order mid‑season | Sub‑class `PriorityQueue`, swap the implementation in `Plant.create_resource_pools()`. |
| Use LiDAR‑based LAI | Inject your own `leaf_area_index` after each step. |
| Run at 30‑min steps | Change the drivers and time‑step loop – no model maths rely on 1 h. |

Because everything is **Python**, any object can be patched or subclassed without recompiling the core.

---

## 🧪 Testing & CI

A single `pytest` file (`tests/test_regression.py`) re‑runs MPM and compares
numeric output against a frozen baseline with `numpy.allclose(atol=1e‑6)`.

Add a GitHub Actions workflow:

```yaml
- uses: actions/setup-python@v5
  with: {python-version: '3.11'}
- run: pip install -e .[dev]
- run: pytest -q
```

---

## 📗 Documentation Roadmap

* **API Reference** generated via *pdoc* (planned)  
* Tutorial notebooks: greenhouse vs. field, adding a fruit pool (planned)

---

## 📄 License

MPM is distributed under the [MIT License](LICENSE).

---

## 👥 Acknowledgements

Developed by **Diane Wang, Sajad Jamshidi, and Bryan Runck** as an open, minimal starting point for next‑generation crop simulators. Contributions and forks are welcome — feel free to open an issue or pull request!
