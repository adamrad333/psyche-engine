"""Command line interface: ``psyche ingest|update|show``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from psyche.enrich import NPPESClient
from psyche.pipeline import ingest, show, update


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="psyche",
        description="Bayesian persona engine (see docs/ALGORITHM.md).")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="ingest a leads CSV into the store")
    p_ingest.add_argument("leads_csv")
    p_ingest.add_argument("--store", default="store", help="store directory")
    p_ingest.add_argument("--margins", default=None, help="population margins CSV")
    p_ingest.add_argument("--offline", action="store_true",
                          help="skip NPPES enrichment entirely")
    p_ingest.add_argument("--cache", default=None, help="NPPES SQLite cache path")

    p_update = sub.add_parser("update", help="run one update cycle on confirmations")
    p_update.add_argument("confirmations_csv")
    p_update.add_argument("--store", default="store", help="store directory")
    p_update.add_argument("--margins", default=None,
                          help="population margins CSV (re-rake if leads changed)")

    p_show = sub.add_parser("show", help="print persona store summary")
    p_show.add_argument("--store", default="store", help="store directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store_dir = Path(args.store)

    if args.command == "ingest":
        enricher = None
        if not args.offline:
            enricher = NPPESClient(cache_path=args.cache
                                   or store_dir / "nppes_cache.sqlite")
        try:
            store = ingest(args.leads_csv, store_dir, margins_csv=args.margins,
                           enricher=enricher)
        finally:
            if enricher is not None:
                enricher.close()
        print(f"ingested leads into {store_dir} "
              f"({len(store.personas)} persona(s))")
        return 0

    if args.command == "update":
        summary = update(args.confirmations_csv, store_dir, margins_csv=args.margins)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.command == "show":
        print(show(store_dir))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
