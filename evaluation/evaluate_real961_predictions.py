#!/usr/bin/env python3
"""Entry point for the audited 190-patient real961 test set."""

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
