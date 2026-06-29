#!/usr/bin/env python
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, ROOT)

from zeng_repro.convective import run_convective_figure
from zeng_repro.io import MissingInputError, load_config


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    try:
        paths = run_convective_figure(cfg, "fig2_bd")
    except MissingInputError as exc:
        print("Missing required input: {}".format(exc))
        return 2
    print(paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
