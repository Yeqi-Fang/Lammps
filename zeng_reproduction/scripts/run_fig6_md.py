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
    status = 0
    for entry in cfg.get("panels", []):
        merged = dict(cfg)
        merged.update(entry)
        name = "fig6_{}".format(entry.get("name", entry["system"]).lower())
        try:
            paths = run_convective_figure(merged, name)
            print(paths)
        except MissingInputError as exc:
            print("[{}] Missing required input: {}".format(name, exc))
            status = 2
    return status


if __name__ == "__main__":
    raise SystemExit(main())
