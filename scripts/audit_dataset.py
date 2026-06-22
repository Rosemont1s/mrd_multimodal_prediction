#!/usr/bin/env python3
"""Create linked-table templates or audit/build the MRD analytical cohort."""

from __future__ import annotations

import argparse
import logging
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data.cohort_audit import (
    audit_and_build_cohort,
    save_audit_result,
    write_table_templates,
)
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--init-templates",
        metavar="DIRECTORY",
        help="Write empty linked-table CSV templates and exit.",
    )
    parser.add_argument(
        "--allow-not-ready",
        action="store_true",
        help="Write audit outputs even when readiness blockers remain.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if args.init_templates:
        write_table_templates(args.init_templates)
        return
    cfg = load_config(args.config)
    result = audit_and_build_cohort(cfg)
    data_cfg = cfg["data"]
    save_audit_result(
        result,
        data_cfg["clinical_csv"],
        data_cfg["readiness_report"],
        data_cfg["audit_issues"],
    )
    if not result.ready and not args.allow_not_ready:
        blockers = "\n- ".join(result.report["blockers"])
        raise SystemExit(f"Dataset is not ready:\n- {blockers}")


if __name__ == "__main__":
    main()
