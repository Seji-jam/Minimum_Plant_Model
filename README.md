# Minimum Plant Model (MPM)

## 🌱 Project Description

The **Minimum Plant Model (MPM)** is a modular, object-oriented simulation framework for modeling crop growth and development based on daily or sub-daily environmental inputs. It simulates **carbon assimilation**, **resource allocation**, and the **growth of plant components (resource pools)** over time using physiological and environmental parameters.

This project was refactored from a monolithic research prototype into a publish-ready, modular Python package with full support for reproducible simulation, visualization, and scientific benchmarking.

---

## ✨ Features

- 🌞 **Photosynthesis model** using a sunlit/shaded canopy approach
- 🌡️ **Thermal time-based development** for plant and organ growth
- 🌱 **Carbon allocation** among user-defined resource pools (e.g. leaf, stem, root)
- ⚙️ **Modular and extensible design** with separate environment and plant logic
- 📊 **Built-in output visualization** and CSV export
- ✅ **Regression testing** to ensure scientific consistency with baseline runs

---

## 📁 Project Structure

```
MPM_testing/
├── mpm/                      # Main package (importable as 'mpm')
│   ├── cli.py                # CLI entry point
│   ├── __init__.py
│   ├── environment/          # Environment classes (atmosphere, light)
│   └── plant/                # Plant logic (growth, carbon, allocation)
├── data/                     # Sample input files (drivers.csv, etc.)
├── testdata/                 #  test suite with baseline check
├── pyproject.toml            # Build system & install config
├── LICENSE
└── README.md                 # This file
```

---

## 🚀 Getting Started

### 📦 Prerequisites

- Python ≥ 3.9
- pip
- Optional (for development): `pytest`, `matplotlib`

Install the package locally in **editable mode**:

```bash
pip install -e .[dev]
```

---

### ▶️ Running a Simulation

You can run the model using the CLI interface:

```bash
 python -m mpm.cli  testdata/MPM_driver.csv testdata/parameters_CC.csv testdata/RP_CC.csv
```

This will:
- Run the hourly simulation
- Plot carbon pool over time
- Write `model_outputs.csv` in the working directory

You can also call the model programmatically:

```python
from mpm.plant.model_handler import ModelHandler

m = ModelHandler("data/drivers.csv", "data/params.csv", "data/resource_pools.csv")
m.run_simulation()
m.plot_outputs()
m.save_outputs_to_csv()
```

---

## 🧬 Model Components

### 📦 Resource Pools
- Each plant organ is modeled as a *resource pool* with a logistic growth curve.
- Examples: `leaf`, `stem`, `root`, `seed`

### 🔄 Carbon Allocation
- Photosynthesis supplies a carbon pool.
- A priority queue allocates carbon based on developmental stage and thermal time.

### ☀️ Photosynthesis
- Based on absorbed PAR using a **sunlit/shaded canopy model**
- Accounts for VPD, temperature, and light quality

### 🌡️ Thermal Development
- Plant and organs grow based on **thermal age**
- Daily increments are calculated from air temperature and base temperature

---

## 🧪 Testing

To check that the model’s output is identical to a known-good baseline:

```bash
pytest
```

Baseline CSV files should be stored in `tests/baseline/`.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙋‍♀️ Author & Acknowledgements

Developed by Sajad Jam, based on original research in plant modeling and carbon allocation.

Inspired by modular physiological models and structured for reproducibility, maintainability, and scientific sharing.