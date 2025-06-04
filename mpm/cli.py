"""Simple command-line wrapper:  `python -m mpm ...`"""
from __future__ import annotations

import argparse
from pathlib import Path

from .plant.model_handler import ModelHandler


def main() -> None:
    p = argparse.ArgumentParser(prog="mpm")
    p.add_argument("drivers")
    p.add_argument("params")
    p.add_argument("pools")
    args = p.parse_args()

    mdl = ModelHandler(Path(args.drivers), Path(args.params), Path(args.pools))
    mdl.run_simulation()
    mdl.plot_outputs()
    mdl.save_outputs_to_csv()


if __name__ == "__main__":
    main()