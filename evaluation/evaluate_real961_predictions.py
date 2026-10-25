#!/usr/bin/env python3
"""Legacy real961 wrapper; requires --allow-legacy-protocol.

Uses GT-positive voting and shared metadata boxes. Use evaluate_paper_protocol.py
for phase-specific annotations and all eligible diagnostic votes.
"""

import sys

try:
    from .evaluate_sim999_predictions import main
except ImportError:  # Direct script execution from this directory.
    from evaluate_sim999_predictions import main


if __name__ == "__main__":
    sys.argv[1:1] = [
        "--expected-patients",
        "190",
        "--dataset-label",
        "real961",
        "--cluster-field",
        "statistical_patient_id",
    ]
    main()
