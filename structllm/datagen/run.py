"""Command-line drivers for request writing and teacher trajectories."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from structllm.datagen.policy import GenerationLimits
from structllm.datagen.requests import RequestWriter, generate_requests, load_requests, write_requests
from structllm.datagen.scenarios import check_against_catalogue, load_scenarios
from structllm.datagen.trajectories import (
    SetupError,
    TrajectoryWriter,
    model_failure,
    create_project,
    discover_ml_engines,
    run_request,
    scenario_split,
)
from structllm.dataset import load_catalogue
from structllm.toolspec import Catalogue


def requests_command(*, catalogue_path: Path, scenarios_path: Path | None, output: Path, seed: int, model: str) -> Path:
    catalogue = Catalogue(load_catalogue(catalogue_path))
    scenarios = load_scenarios(scenarios_path)
    check_against_catalogue(scenarios, catalogue)
    writer = RequestWriter(model=model)
    kept, dropped = generate_requests(scenarios, writer, seed=seed)
    path = write_requests(kept, dropped, output)
    print(
        f"Wrote {len(kept)} requests ({len(dropped)} dropped) to {path}; "
        f"writer tokens in/out {writer.input_tokens}/{writer.output_tokens}"
    )
    return path


async def _trajectories(
    *,
    catalogue: Catalogue,
    requests: list[dict[str, Any]],
    scenarios: dict[str, Any],
    output_dir: Path,
    model: str,
    effort: str,
    seed: int,
    eval_fraction: float,
    limit: int | None,
) -> Path:
    from structllm.adapters import AnthropicAdapter
    from structllm.mcp_client import MCPConnection, MCPSettings

    writer = TrajectoryWriter(output_dir)
    done = writer.done()
    pending = [request for request in requests if request["request_id"] not in done]
    if limit is not None:
        pending = pending[:limit]
    settings = MCPSettings.from_environment()
    async with MCPConnection(settings) as connection:
        live = await connection.snapshot()
        if live["catalogue_sha256"] != catalogue.sha256:
            raise SystemExit(
                f"Live catalogue {live['catalogue_sha256'][:12]} differs from the snapshot "
                f"{catalogue.sha256[:12]}; take a fresh snapshot and regenerate requests first."
            )
        probe = await create_project(connection, catalogue, "structllm-datagen engine listing")
        generation = GenerationLimits(ml_engines=await discover_ml_engines(connection, catalogue, probe))
        print(f"{len(pending)} requests to run ({len(done)} already done); ML engines from server: {sorted(generation.ml_engines)}")
        for number, request in enumerate(pending, 1):
            scenario = scenarios.get(request["scenario_id"])
            if scenario is None:
                writer.error(request, "scenario no longer exists")
                continue
            split = scenario_split(scenario.id, seed=seed, eval_fraction=eval_fraction)
            try:
                record = await run_request(
                    request, scenario, catalogue=catalogue, transport=connection,
                    teacher_factory=lambda: AnthropicAdapter(model=model, effort=effort, rationale_field="rationale"),
                    generation=GenerationLimits(ml_engines=generation.ml_engines),
                    split=split,
                )
            except SetupError as error:
                writer.error(request, str(error))
                print(f"[{number}/{len(pending)}] {scenario.id}: setup failed: {error}")
                continue
            failure = model_failure(record)
            if failure:
                # Not the teacher's answer: the API call failed. Keep it out of the
                # data and run it again next time.
                writer.error(request, failure, retryable=True)
                print(f"[{number}/{len(pending)}] {scenario.id}: model call failed: {failure[:160]}", flush=True)
                if "credit balance" in failure or "authentication" in failure.lower():
                    raise SystemExit("Stopping: the Anthropic account cannot serve requests (credit or authentication).")
                continue
            writer.add(record)
            print(
                f"[{number}/{len(pending)}] {scenario.id}: {record['outcome']} "
                f"{'accepted' if record['accepted'] else 'rejected ' + '; '.join(record['problems'])} "
                f"(executed {record['executed_calls']}, mocked {record['mocked_calls']})",
                flush=True,
            )
    return writer.manifest(
        {
            "catalogue_sha256": catalogue.sha256,
            "teacher_model": model,
            "teacher_effort": effort,
            "seed": seed,
            "eval_fraction": eval_fraction,
            "endpoint": settings.endpoint,
        }
    )


def trajectories_command(
    *,
    catalogue_path: Path,
    requests_path: Path,
    scenarios_path: Path | None,
    output_dir: Path,
    model: str,
    effort: str,
    seed: int,
    eval_fraction: float,
    limit: int | None,
    shard: tuple[int, int] = (0, 1),
) -> Path:
    catalogue = Catalogue(load_catalogue(catalogue_path))
    scenarios = load_scenarios(scenarios_path)
    check_against_catalogue(scenarios, catalogue)
    index, count = shard
    if not 0 <= index < count:
        raise SystemExit(f"shard {index}/{count} is out of range")
    directory = output_dir / f"trajectories-{catalogue.sha256[:12]}-seed{seed}"
    if count > 1:
        # Each worker process writes its own resumable directory.
        directory = directory / f"shard-{index}-of-{count}"
    requests = [request for number, request in enumerate(load_requests(requests_path)) if number % count == index]
    manifest = asyncio.run(
        _trajectories(
            catalogue=catalogue,
            requests=requests,
            scenarios={scenario.id: scenario for scenario in scenarios},
            output_dir=directory,
            model=model,
            effort=effort,
            seed=seed,
            eval_fraction=eval_fraction,
            limit=limit,
        )
    )
    print(f"Manifest written to {manifest}")
    return manifest
