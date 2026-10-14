"""Command-line entry point for StructSub MCP capability discovery."""

from __future__ import annotations

import argparse
from pathlib import Path

from structllm.catalogue import snapshot_from_environment, write_snapshot
from structllm.dataset import load_catalogue, write_dataset
from structllm.validation import capture_from_environment, write_observed_failure


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Discover StructSub MCP capabilities and prepare schema-only datasets."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("catalogues"),
        help="Directory for the immutable JSON snapshot (default: ./catalogues)",
    )
    commands = parser.add_subparsers(dest="command")
    generate = commands.add_parser(
        "generate-dataset",
        help="Generate non-executable schema examples from a verified catalogue snapshot.",
    )
    generate.add_argument("--catalogue", type=Path, required=True)
    generate.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Volumes/X9/scratch/structsub-mcp-dev/datasets"),
    )
    generate.add_argument("--seed", type=int, default=17)
    generate.add_argument("--holdout-fraction", type=float, default=0.2)
    capture = commands.add_parser(
        "capture-invalid-smiles",
        help="Run one cheap invalid-SMILES check and record the real backend failure.",
    )
    capture.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Volumes/X9/scratch/structsub-mcp-dev/observed-failures"),
    )
    requests = commands.add_parser(
        "generate-requests",
        help="Write synthetic user requests from scenario specs (calls the Anthropic API).",
    )
    requests.add_argument("--catalogue", type=Path, required=True)
    requests.add_argument("--scenarios", type=Path, default=None, help="Scenario YAML (default: packaged specs)")
    requests.add_argument("--output", type=Path, required=True, help="New .jsonl file; never overwritten")
    requests.add_argument("--seed", type=int, default=17)
    requests.add_argument("--model", default="claude-opus-5-5")
    trajectories = commands.add_parser(
        "generate-trajectories",
        help="Answer requests with a teacher model against the development MCP server (cheap operations only).",
    )
    trajectories.add_argument("--catalogue", type=Path, required=True)
    trajectories.add_argument("--requests", type=Path, required=True)
    trajectories.add_argument("--scenarios", type=Path, default=None)
    trajectories.add_argument(
        "--output-dir", type=Path, default=Path("/Volumes/X9/scratch/structsub-mcp-dev/trajectories")
    )
    trajectories.add_argument("--model", default="claude-opus-5-5")
    trajectories.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    trajectories.add_argument("--seed", type=int, default=17)
    trajectories.add_argument("--eval-fraction", type=float, default=0.2)
    trajectories.add_argument("--limit", type=int, default=None, help="Run at most this many new requests")
    trajectories.add_argument(
        "--shard", default="0/1", help="K/N: run every N-th request starting at K, for parallel workers"
    )
    export = commands.add_parser(
        "export-sft",
        help="Merge accepted trajectories into self-contained train/eval JSONL for fine-tuning.",
    )
    export.add_argument("--trajectories", type=Path, required=True, help="Trajectory directory (shards included)")
    export.add_argument("--output-dir", type=Path, default=Path("/Volumes/X9/scratch/structsub-mcp-dev/sft"))
    export.add_argument("--no-salvage", action="store_true", help="Do not add the passing opening turns of rejected conversations")
    export.add_argument("--allow-ungrounded", action="store_true", help="Keep conversations whose numbers no tool result supports")
    assemble = commands.add_parser("assemble-sft", help="Join several SFT exports (one per catalogue) into one training set.")
    assemble.add_argument("--exports", type=Path, nargs="+", required=True)
    assemble.add_argument("--output-dir", type=Path, required=True, help="New directory; never overwritten")
    args = parser.parse_args()

    if args.command == "assemble-sft":
        from structllm.datagen.export import assemble_sft

        print(f"Assembled training set at {assemble_sft(args.exports, args.output_dir)}")
        return
    if args.command == "export-sft":
        from structllm.datagen.export import export_sft

        destination = export_sft(args.trajectories, args.output_dir, require_grounded=not args.allow_ungrounded, salvage=not args.no_salvage)
        print(f"Exported fine-tuning data to {destination}")
        return

    if args.command == "generate-requests":
        from structllm.datagen.run import requests_command

        requests_command(
            catalogue_path=args.catalogue, scenarios_path=args.scenarios,
            output=args.output, seed=args.seed, model=args.model,
        )
        return
    if args.command == "generate-trajectories":
        from structllm.datagen.run import trajectories_command

        trajectories_command(
            catalogue_path=args.catalogue, requests_path=args.requests, scenarios_path=args.scenarios,
            output_dir=args.output_dir, model=args.model, effort=args.effort, seed=args.seed,
            eval_fraction=args.eval_fraction, limit=args.limit,
            shard=tuple(int(part) for part in args.shard.split("/", 1)),
        )
        return

    if args.command == "generate-dataset":
        snapshot = load_catalogue(args.catalogue)
        destination = write_dataset(
            snapshot,
            output_dir=args.output_dir,
            seed=args.seed,
            holdout_fraction=args.holdout_fraction,
        )
        print(
            f"Generated schema-only dataset package at {destination} "
            f"from catalogue SHA-256 {snapshot['catalogue_sha256']}"
        )
        return
    if args.command == "capture-invalid-smiles":
        record = capture_from_environment()
        destination = write_observed_failure(record, args.output_dir)
        print(
            f"Recorded real {record['observed_response']['status']} response from "
            f"{record['provenance']['tool_name']} to {destination}"
        )
        return

    snapshot = snapshot_from_environment()
    destination = write_snapshot(snapshot, args.output_dir)
    print(
        f"Saved {snapshot['tool_count']} authorized MCP tool definitions "
        f"(catalogue SHA-256 {snapshot['catalogue_sha256']}) to {destination}"
    )


if __name__ == "__main__":
    main()
