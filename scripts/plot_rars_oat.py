#!/usr/bin/env python3
"""Plot complete OAT Accuracy, macro-Recall and macro-F1 curves with seed intervals."""
from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from visualization.rars_oat import plot_oat


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    print(plot_oat(args.selection, args.output))


if __name__ == "__main__":
    main()
