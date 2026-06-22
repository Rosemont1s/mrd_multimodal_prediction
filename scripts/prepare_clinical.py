#!/usr/bin/env python3
"""Convert the cohort workbook into a canonical clinical CSV."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data.clinical_builder import attach_mrd_labels, build_baseline_clinical_table

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", required=True)
    parser.add_argument("--output", default="data/processed/clinical_baseline.csv")
    parser.add_argument(
        "--labels-csv",
        help="Independent patient-level MRD label CSV. Outdated workbook values "
        "are never interpreted as MRD labels.",
    )
    parser.add_argument("--label-id-column", default="patient_id")
    parser.add_argument("--label-column", default="mrd_status")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    clinical = build_baseline_clinical_table(args.workbook)
    if args.labels_csv:
        labels = pd.read_csv(args.labels_csv, dtype={args.label_id_column: str})
        clinical = attach_mrd_labels(
            clinical,
            labels,
            label_id_column=args.label_id_column,
            label_column=args.label_column,
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    clinical.to_csv(output, index=False)
    logger.info(
        "Wrote %d patients and %d columns to %s",
        len(clinical),
        len(clinical.columns),
        output,
    )
    if not args.labels_csv:
        logger.warning(
            "No MRD labels were attached. Add an independently sourced label "
            "table before using this CSV for model training."
        )


if __name__ == "__main__":
    main()
