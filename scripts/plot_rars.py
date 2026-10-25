#!/usr/bin/env python3
"""Render one confirmed 15x15 RARS parameter-pair surface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualization.rars_surface import load_rars_grid, plot_rars_pair


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", required=True, type=Path)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    output = plot_rars_pair(load_rars_grid(arguments.grid, arguments.pair_id), arguments.output)
    print(output)


if __name__ == "__main__":
    main()
