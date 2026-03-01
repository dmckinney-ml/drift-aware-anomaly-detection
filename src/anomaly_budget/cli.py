# src/anomaly_budget/cli.py
from __future__ import annotations

import argparse
import sys
import traceback

from anomaly_budget.run import run_from_config_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="anomaly-budget",
        description="Alert-budget controlled anomaly detection",
    )
    sub = p.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run anomaly detection experiment")
    run_p.add_argument(
        "--config", "-c",
        default="configs/nyc_taxi_mad.yaml",
        help="Path to YAML config file",
    )

    # We pass just the config path into the library entrypoint.
    run_p.set_defaults(func=lambda a: run_from_config_path(a.config))

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    try:
        rc = args.func(args)
        return int(rc) if rc is not None else 0
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())