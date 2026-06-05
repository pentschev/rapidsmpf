#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Callable


BINARY_UNITS = {
    "B": 1.0,
    "KiB": float(1 << 10),
    "MiB": float(1 << 20),
    "GiB": float(1 << 30),
    "TiB": float(1 << 40),
    "PiB": float(1 << 50),
}

DECIMAL_UNITS = {
    "B": 1.0,
    "KB": 1.0e3,
    "MB": 1.0e6,
    "GB": 1.0e9,
    "TB": 1.0e12,
    "PB": 1.0e15,
}

BYTES_PER_TIB = float(1 << 40)


@dataclass(frozen=True)
class Benchmark:
    series: str
    implementation: str
    variant: str
    parser: Callable[[str], list[float]]
    command_builder: Callable[[argparse.Namespace, int], tuple[list[str], dict[str, str]]]
    scale_by_gpus: bool = False


@dataclass
class Result:
    series: str
    implementation: str
    variant: str
    gpus: int
    status: str
    returncode: int | None
    elapsed_s: float
    bandwidth_tib_s: float | None
    sample_count: int
    sample_min_tib_s: float | None
    sample_mean_tib_s: float | None
    sample_median_tib_s: float | None
    sample_max_tib_s: float | None
    command: str
    log_path: Path
    error: str


def parse_size(value: str) -> int:
    match = re.fullmatch(r"\s*(\d+)\s*([A-Za-z]*)\s*", value)
    if match is None:
        raise argparse.ArgumentTypeError(f"invalid size: {value}")

    number = int(match.group(1))
    suffix = match.group(2).lower()
    if suffix in {"", "b"}:
        multiplier = 1
    elif suffix in {"k", "kb", "kib"}:
        multiplier = 1 << 10
    elif suffix in {"m", "mb", "mib"}:
        multiplier = 1 << 20
    elif suffix in {"g", "gb", "gib"}:
        multiplier = 1 << 30
    elif suffix in {"t", "tb", "tib"}:
        multiplier = 1 << 40
    else:
        raise argparse.ArgumentTypeError(f"invalid size suffix: {suffix}")
    return number * multiplier


def format_bytes(value: int) -> str:
    return str(value)


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def split_extra_args(value: str) -> list[str]:
    return shlex.split(value) if value else []


def parse_env_assignments(values: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError(f"environment entry must be KEY=VALUE: {value}")
        key, val = value.split("=", 1)
        if not key:
            raise argparse.ArgumentTypeError(f"environment key is empty: {value}")
        env[key] = val
    return env


def rate_to_tib_s(value: float, unit: str) -> float:
    if unit in BINARY_UNITS:
        return value * BINARY_UNITS[unit] / BYTES_PER_TIB
    if unit in DECIMAL_UNITS:
        return value * DECIMAL_UNITS[unit] / BYTES_PER_TIB
    raise ValueError(f"unknown bandwidth unit: {unit}")


def parse_rapids_aggregate(output: str) -> list[float]:
    results: list[float] = []
    pattern = re.compile(
        r"aggregate:\s*([0-9.+\-eE]+)\s+GiB/s\s+\(([0-9.+\-eE]+)\s+TiB/s\)"
    )
    for line in output.splitlines():
        if "(warmup run)" in line:
            continue
        match = pattern.search(line)
        if match is not None:
            results.append(float(match.group(2)))
    return results


def parse_bench_comm(output: str) -> list[float]:
    means: list[float] = []
    measured_runs: list[float] = []
    pattern = re.compile(
        r"global throughput:\s*([0-9.+\-eE]+)\s+([A-Za-z]+)(?:/s)?"
    )
    for line in output.splitlines():
        match = pattern.search(line)
        if match is None:
            continue
        value = rate_to_tib_s(float(match.group(1)), match.group(2))
        if "means:" in line:
            means.append(value)
        elif "(warmup run)" not in line:
            measured_runs.append(value)
    return means[-1:] if means else measured_runs


def parse_nccl_alltoall(output: str) -> list[float]:
    avg_pattern = re.compile(r"Avg bus bandwidth\s*:\s*([0-9.+\-eE]+)")
    for line in output.splitlines():
        match = avg_pattern.search(line)
        if match is not None:
            return [rate_to_tib_s(float(match.group(1)), "GB")]

    results: list[float] = []
    for line in output.splitlines():
        fields = line.split()
        if len(fields) < 8 or line.lstrip().startswith("#"):
            continue
        if not fields[0].isdigit():
            continue
        try:
            results.append(rate_to_tib_s(float(fields[7]), "GB"))
        except ValueError:
            continue
    return results


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def maybe_min(values: list[float]) -> float | None:
    return min(values) if values else None


def maybe_max(values: list[float]) -> float | None:
    return max(values) if values else None


def fmt(value: float | None, precision: int = 3) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.{precision}f}"


def default_rapidsmpf_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_workspace_root() -> Path:
    return default_rapidsmpf_root().parent


def resolve_nccl_binary(args: argparse.Namespace) -> Path:
    if args.nccl_alltoall_bin is not None:
        return args.nccl_alltoall_bin

    candidates = [
        args.nccl_tests_root / "build" / "alltoall_perf",
        args.nccl_tests_root / "build" / "alltoall_perf_mpi",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def rapids_binary(args: argparse.Namespace, name: str) -> Path:
    return args.rapidsmpf_build / "benchmarks" / name


def launcher_env_flags(args: argparse.Namespace, env: dict[str, str]) -> list[str]:
    flags: list[str] = []
    if args.launcher == "rrun":
        for key, value in sorted(env.items()):
            flags.extend(["-x", f"{key}={value}"])
    elif args.launcher == "mpirun":
        for key, value in sorted(env.items()):
            flags.extend(["-x", f"{key}={value}"])
    else:
        raise ValueError(f"unknown launcher: {args.launcher}")
    return flags


def launch_prefix(args: argparse.Namespace, gpus: int, env: dict[str, str]) -> list[str]:
    if args.launcher == "rrun":
        command = [
            str(args.rrun),
            "-n",
            str(gpus),
            "--bind-to",
            "cpu",
            "--bind-to",
            "memory",
        ]
        command.extend(launcher_env_flags(args, env))
        return command

    command = [args.mpirun, "-n", str(gpus), "/raid/pentschev/src/rapidsmpf/binder.sh"]
    command.extend(launcher_env_flags(args, env))
    return command


def common_env(args: argparse.Namespace) -> dict[str, str]:
    return dict(args.extra_env)


def ucx_env(args: argparse.Namespace) -> dict[str, str]:
    env = common_env(args)
    env.update(args.ucx_env)
    if args.ucx_rndv_scheme:
        env["UCX_RNDV_SCHEME"] = args.ucx_rndv_scheme
    return env


def build_nccl_command(args: argparse.Namespace, gpus: int) -> tuple[list[str], dict[str, str]]:
    total_rank_bytes = args.bytes_per_peer * gpus
    command = [
        str(resolve_nccl_binary(args)),
        "-g",
        str(gpus),
        "-b",
        format_bytes(total_rank_bytes),
        "-e",
        format_bytes(total_rank_bytes),
        "-f",
        "2",
        "-w",
        str(args.warmups),
        "-n",
        str(args.runs),
        "-m",
        str(args.ops),
        "-c",
        "0",
    ]
    command.extend(split_extra_args(args.nccl_extra_args))
    env = common_env(args)
    env.update(args.nccl_env)
    return command, env


def build_bench_comm_command(
    args: argparse.Namespace, gpus: int, post_order: str
) -> tuple[list[str], dict[str, str]]:
    env = ucx_env(args)
    command = launch_prefix(args, gpus, env)
    command.extend(
        [
            str(rapids_binary(args, "bench_comm")),
            "-C",
            args.bench_comm_communicator,
            "-O",
            "all-to-all",
            "-w",
            str(args.warmups),
            "-r",
            str(args.runs),
            "-n",
            format_bytes(args.bytes_per_peer),
            "-p",
            str(args.ops),
            "-m",
            args.bench_comm_memory_resource,
            "-P",
            post_order,
            "-T",
            args.bench_comm_tag_mode,
            "-W",
            args.completion_mode,
            "-G",
            args.progress_during_post,
        ]
    )
    command.extend(split_extra_args(args.bench_comm_extra_args))
    return command, env


def build_cuda_ipc_command(
    args: argparse.Namespace, gpus: int, post_order: str
) -> tuple[list[str], dict[str, str]]:
    env = common_env(args)
    command = launch_prefix(args, gpus, env)
    command.extend(
        [
            str(rapids_binary(args, "bench_cuda_ipc")),
            "-n",
            format_bytes(args.bytes_per_peer),
            "-p",
            str(args.ops),
            "-w",
            str(args.warmups),
            "-r",
            str(args.runs),
            "--mode",
            args.cuda_ipc_mode,
            "--streams",
            args.cuda_ipc_streams,
            "-P",
            post_order,
        ]
    )
    if args.cuda_ipc_no_gate:
        command.append("--no-gate")
    command.extend(split_extra_args(args.cuda_ipc_extra_args))
    return command, env


def build_ucp_tag_command(
    args: argparse.Namespace, gpus: int, post_order: str
) -> tuple[list[str], dict[str, str]]:
    env = ucx_env(args)
    command = launch_prefix(args, gpus, env)
    command.extend(
        [
            str(rapids_binary(args, "bench_ucp_tag")),
            "-n",
            format_bytes(args.bytes_per_peer),
            "-p",
            str(args.ops),
            "-w",
            str(args.warmups),
            "-r",
            str(args.runs),
            "-P",
            post_order,
            "-T",
            args.ucp_tag_mode,
            "-W",
            args.completion_mode,
            "-G",
            args.progress_during_post,
            "--memory-type",
            args.ucp_tag_memory_type,
        ]
    )
    command.extend(split_extra_args(args.ucp_tag_extra_args))
    return command, env


def build_ucp_rma_command(
    args: argparse.Namespace, gpus: int, post_order: str
) -> tuple[list[str], dict[str, str]]:
    env = ucx_env(args)
    command = launch_prefix(args, gpus, env)
    command.extend(
        [
            str(rapids_binary(args, "bench_ucp_rma")),
            "-n",
            format_bytes(args.bytes_per_peer),
            "-p",
            str(args.ops),
            "-w",
            str(args.warmups),
            "-r",
            str(args.runs),
            "--mode",
            args.ucp_rma_mode,
            "-P",
            post_order,
            "-W",
            args.completion_mode,
            "-G",
            args.progress_during_post,
            "--memory-type",
            args.ucp_rma_memory_type,
        ]
    )
    if args.ucp_rma_no_local_memh:
        command.append("--no-local-memh")
    if args.ucp_rma_no_flush:
        command.append("--no-flush")
    command.extend(split_extra_args(args.ucp_rma_extra_args))
    return command, env


def make_benchmarks(args: argparse.Namespace) -> list[Benchmark]:
    device_order = args.device_major_post_order
    balanced_order = args.balanced_post_order

    benchmarks = [
        Benchmark(
            "NCCL alltoall",
            "nccl-tests/src/alltoall.cu",
            "default",
            parse_nccl_alltoall,
            build_nccl_command,
            True,
        )
    ]

    rapids_benchmarks = [
        (
            "bench_comm",
            "rapidsmpf/cpp/benchmarks/bench_comm.cpp",
            parse_bench_comm,
            build_bench_comm_command,
        ),
        (
            "bench_cuda_ipc",
            "rapidsmpf/cpp/benchmarks/bench_cuda_ipc.cu",
            parse_rapids_aggregate,
            build_cuda_ipc_command,
        ),
        (
            "bench_ucp_tag",
            "rapidsmpf/cpp/benchmarks/bench_ucp_tag.cu",
            parse_rapids_aggregate,
            build_ucp_tag_command,
        ),
        (
            "bench_ucp_rma",
            "rapidsmpf/cpp/benchmarks/bench_ucp_rma.cu",
            parse_rapids_aggregate,
            build_ucp_rma_command,
        ),
    ]
    for series_base, implementation, parser, builder in rapids_benchmarks:
        benchmarks.append(
            Benchmark(
                f"{series_base} device-major",
                implementation,
                "device-major",
                parser,
                lambda ns, g, b=builder, order=device_order: b(ns, g, order),
            )
        )
        benchmarks.append(
            Benchmark(
                f"{series_base} balanced",
                implementation,
                "balanced",
                parser,
                lambda ns, g, b=builder, order=balanced_order: b(ns, g, order),
            )
        )

    wanted = set(args.only) if args.only else None
    skipped = set(args.skip)
    return [
        benchmark
        for benchmark in benchmarks
        if (wanted is None or benchmark.series in wanted) and benchmark.series not in skipped
    ]


def executable_from_command(command: list[str]) -> Path:
    return Path(command[0])


def command_is_missing(command: list[str]) -> bool:
    path = executable_from_command(command)
    if path.is_absolute() or "/" in command[0]:
        return not path.exists()
    return False


def run_one(
    args: argparse.Namespace,
    benchmark: Benchmark,
    gpus: int,
    index: int,
    total: int,
    output_dir: Path,
) -> Result:
    command, env_updates = benchmark.command_builder(args, gpus)
    command_text = shell_join(command)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{index:02d}_{benchmark.series}_{gpus}gpu")
    log_path = output_dir / "logs" / f"{safe_name}.log"

    print(f"[{index}/{total}] {benchmark.series} | GPUs={gpus}")
    print(f"  $ {command_text}")

    if args.dry_run:
        log_path.write_text("DRY RUN\n" + command_text + "\n", encoding="utf-8")
        return Result(
            benchmark.series,
            benchmark.implementation,
            benchmark.variant,
            gpus,
            "dry-run",
            None,
            0.0,
            None,
            0,
            None,
            None,
            None,
            None,
            command_text,
            log_path,
            "",
        )

    if command_is_missing(command):
        message = f"missing executable: {command[0]}"
        if not args.skip_missing:
            raise FileNotFoundError(message)
        log_path.write_text(message + "\n", encoding="utf-8")
        return Result(
            benchmark.series,
            benchmark.implementation,
            benchmark.variant,
            gpus,
            "missing",
            None,
            0.0,
            None,
            0,
            None,
            None,
            None,
            None,
            command_text,
            log_path,
            message,
        )

    env = os.environ.copy()
    env.update(env_updates)
    start = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=args.workspace_root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=args.timeout,
            check=False,
        )
        elapsed_s = time.monotonic() - start
        output = completed.stdout
        log_path.write_text(
            "$ " + command_text + "\n\n" + output,
            encoding="utf-8",
            errors="replace",
        )
        if args.verbose:
            print(output, end="" if output.endswith("\n") else "\n")
        if completed.returncode != 0:
            return Result(
                benchmark.series,
                benchmark.implementation,
                benchmark.variant,
                gpus,
                "failed",
                completed.returncode,
                elapsed_s,
                None,
                0,
                None,
                None,
                None,
                None,
                command_text,
                log_path,
                f"command exited with code {completed.returncode}",
            )

        samples = benchmark.parser(output)
        if benchmark.scale_by_gpus:
            samples = [sample * float(gpus) for sample in samples]
        primary = median(samples)
        status = "ok" if primary is not None else "parse-failed"
        error = "" if primary is not None else "no bandwidth result parsed"
        return Result(
            benchmark.series,
            benchmark.implementation,
            benchmark.variant,
            gpus,
            status,
            completed.returncode,
            elapsed_s,
            primary,
            len(samples),
            maybe_min(samples),
            mean(samples),
            median(samples),
            maybe_max(samples),
            command_text,
            log_path,
            error,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed_s = time.monotonic() - start
        output = exc.stdout or ""
        log_path.write_text(
            "$ " + command_text + "\n\nTIMEOUT\n\n" + output,
            encoding="utf-8",
            errors="replace",
        )
        return Result(
            benchmark.series,
            benchmark.implementation,
            benchmark.variant,
            gpus,
            "timeout",
            None,
            elapsed_s,
            None,
            0,
            None,
            None,
            None,
            None,
            command_text,
            log_path,
            f"timed out after {args.timeout} seconds",
        )


def write_csv(results: list[Result], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "series",
                "implementation",
                "variant",
                "gpus",
                "status",
                "bandwidth_tib_s",
                "bandwidth_gib_s",
                "sample_count",
                "sample_min_tib_s",
                "sample_mean_tib_s",
                "sample_median_tib_s",
                "sample_max_tib_s",
                "elapsed_s",
                "returncode",
                "log_path",
                "error",
                "command",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.series,
                    result.implementation,
                    result.variant,
                    result.gpus,
                    result.status,
                    fmt(result.bandwidth_tib_s, 6),
                    fmt(
                        None
                        if result.bandwidth_tib_s is None
                        else result.bandwidth_tib_s * 1024.0,
                        3,
                    ),
                    result.sample_count,
                    fmt(result.sample_min_tib_s, 6),
                    fmt(result.sample_mean_tib_s, 6),
                    fmt(result.sample_median_tib_s, 6),
                    fmt(result.sample_max_tib_s, 6),
                    f"{result.elapsed_s:.3f}",
                    "" if result.returncode is None else result.returncode,
                    result.log_path,
                    result.error,
                    result.command,
                ]
            )


def markdown_table(results: list[Result], gpus_values: list[int]) -> str:
    by_series_gpu = {(result.series, result.gpus): result for result in results}
    series_order = list(dict.fromkeys(result.series for result in results))
    header = "| Series | " + " | ".join(f"{gpus} GPUs" for gpus in gpus_values) + " |"
    separator = "|---|" + "|".join("---:" for _ in gpus_values) + "|"
    lines = [header, separator]
    for series in series_order:
        cells = []
        for gpus in gpus_values:
            result = by_series_gpu.get((series, gpus))
            if result is None:
                cells.append("")
            elif result.bandwidth_tib_s is not None:
                cells.append(fmt(result.bandwidth_tib_s))
            else:
                cells.append(result.status)
        lines.append("| " + series + " | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_markdown(
    args: argparse.Namespace,
    results: list[Result],
    output_dir: Path,
    csv_path: Path,
    plot_path: Path | None,
) -> Path:
    path = output_dir / "summary.md"
    lines = [
        "# All-to-All Bandwidth Matrix",
        "",
        f"Generated: {dt.datetime.now(dt.UTC).isoformat()}",
        "",
        "## Parameters",
        "",
        f"- GPU counts: {', '.join(str(g) for g in args.gpus)}",
        f"- Bytes per peer/op: {args.bytes_per_peer} B",
        f"- Concurrent ops: {args.ops}",
        f"- Warmups: {args.warmups}",
        f"- Runs: {args.runs}",
        f"- RAPIDS launcher: {args.launcher}",
        f"- Device-major post-order argument: `{args.device_major_post_order}`",
        f"- Balanced post-order argument: `{args.balanced_post_order}`",
        f"- NCCL per-rank size: `bytes_per_peer * nranks`",
        "- NCCL bandwidth is parsed as per-GPU bus bandwidth and multiplied by GPU count.",
        "",
        "## Results",
        "",
        "Bandwidth values are effective aggregate bandwidth in TiB/s.",
        "",
        markdown_table(results, args.gpus),
        "",
        "## Files",
        "",
        f"- CSV: `{csv_path}`",
    ]
    if plot_path is not None:
        lines.append(f"- Plot: `{plot_path}`")
    lines.extend(["- Logs: `logs/`", "", "## Commands", ""])
    for result in results:
        lines.extend(
            [
                f"### {result.series}, {result.gpus} GPUs",
                "",
                f"- Status: `{result.status}`",
                f"- Log: `{result.log_path}`",
                "",
                "```bash",
                result.command,
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def plot_results(results: list[Result], gpus_values: list[int], path: Path) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on local environment
        print(f"warning: could not import matplotlib, skipping plot: {exc}", file=sys.stderr)
        return None

    by_series_gpu = {(result.series, result.gpus): result for result in results}
    series_order = list(dict.fromkeys(result.series for result in results))

    fig, ax = plt.subplots(figsize=(12.5, 7.0))
    for series in series_order:
        xs: list[int] = []
        ys: list[float] = []
        for gpus in gpus_values:
            result = by_series_gpu.get((series, gpus))
            if result is not None and result.bandwidth_tib_s is not None:
                xs.append(gpus)
                ys.append(result.bandwidth_tib_s)
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=2, label=series)

    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("Effective aggregate bandwidth (TiB/s)")
    ax.set_xticks(gpus_values)
    ax.grid(True, which="major", axis="both", alpha=0.3)
    ax.set_title("All-to-All Bandwidth")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def parse_gpu_counts(value: str) -> list[int]:
    counts = [int(part) for part in value.split(",") if part]
    if not counts:
        raise argparse.ArgumentTypeError("at least one GPU count is required")
    if any(count <= 1 for count in counts):
        raise argparse.ArgumentTypeError("GPU counts must be greater than one")
    return counts


def parse_args() -> argparse.Namespace:
    rapidsmpf_root = default_rapidsmpf_root()
    workspace_root = default_workspace_root()
    parser = argparse.ArgumentParser(
        description="Run all-to-all bandwidth benchmarks across GPU counts and plot the result."
    )
    parser.add_argument("--workspace-root", type=Path, default=workspace_root)
    parser.add_argument("--rapidsmpf-root", type=Path, default=rapidsmpf_root)
    parser.add_argument("--rapidsmpf-build", type=Path, default=rapidsmpf_root / "cpp" / "build")
    parser.add_argument("--nccl-tests-root", type=Path, default=workspace_root / "nccl-tests")
    parser.add_argument("--nccl-alltoall-bin", type=Path, default=None)
    parser.add_argument(
        "--rrun",
        type=Path,
        default=rapidsmpf_root / "cpp" / "build" / "tools" / "rrun",
    )
    parser.add_argument("--mpirun", default="mpirun")
    parser.add_argument("--launcher", choices=["rrun", "mpirun"], default="rrun")
    parser.add_argument("--gpus", type=parse_gpu_counts, default=parse_gpu_counts("2,4,8"))
    parser.add_argument(
        "--bytes", dest="bytes_per_peer", type=parse_size, default=parse_size("512M")
    )
    parser.add_argument("--ops", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: rapidsmpf/cpp/benchmark-results/alltoall-matrix-<timestamp>",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--no-plot", action="store_true")

    parser.add_argument("--device-major-post-order", default="device-major")
    parser.add_argument("--balanced-post-order", default="balanced")
    parser.add_argument("--completion-mode", default="unordered")
    parser.add_argument("--progress-during-post", default="request")

    parser.add_argument("--env", dest="extra_env_values", action="append", default=[])
    parser.add_argument("--ucx-env", dest="ucx_env_values", action="append", default=[])
    parser.add_argument("--ucx-rndv-scheme", default="")
    parser.add_argument("--nccl-env", dest="nccl_env_values", action="append", default=[])

    parser.add_argument("--bench-comm-communicator", default="ucxx")
    parser.add_argument("--bench-comm-memory-resource", default="pool")
    parser.add_argument("--bench-comm-tag-mode", default="per-copy")
    parser.add_argument("--bench-comm-extra-args", default="")

    parser.add_argument("--cuda-ipc-mode", choices=["get", "put"], default="put")
    parser.add_argument("--cuda-ipc-streams", default="per-copy")
    parser.add_argument("--cuda-ipc-no-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cuda-ipc-extra-args", default="")

    parser.add_argument("--ucp-tag-mode", default="per-copy")
    parser.add_argument("--ucp-tag-memory-type", choices=["unknown", "cuda"], default="unknown")
    parser.add_argument("--ucp-tag-extra-args", default="")

    parser.add_argument("--ucp-rma-mode", choices=["get", "put"], default="put")
    parser.add_argument("--ucp-rma-memory-type", choices=["unknown", "cuda"], default="cuda")
    parser.add_argument("--ucp-rma-no-local-memh", action="store_true")
    parser.add_argument("--ucp-rma-no-flush", action="store_true")
    parser.add_argument("--ucp-rma-extra-args", default="")

    parser.add_argument("--nccl-extra-args", default="")
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Run only a series name. May be repeated; use --dry-run to list commands.",
    )
    parser.add_argument(
        "--skip", action="append", default=[], help="Skip a series name. May be repeated."
    )

    args = parser.parse_args()
    if args.ops <= 0 or args.warmups < 0 or args.runs <= 0:
        parser.error("--ops and --runs must be positive; --warmups must be non-negative")
    args.extra_env = parse_env_assignments(args.extra_env_values)
    args.ucx_env = parse_env_assignments(args.ucx_env_values)
    args.nccl_env = parse_env_assignments(args.nccl_env_values)
    args.workspace_root = args.workspace_root.resolve()
    args.rapidsmpf_root = args.rapidsmpf_root.resolve()
    args.rapidsmpf_build = args.rapidsmpf_build.resolve()
    args.nccl_tests_root = args.nccl_tests_root.resolve()
    args.rrun = args.rrun.resolve()
    if args.nccl_alltoall_bin is not None:
        args.nccl_alltoall_bin = args.nccl_alltoall_bin.resolve()
    if args.output_dir is None:
        timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_dir = (
            args.rapidsmpf_root
            / "cpp"
            / "benchmark-results"
            / f"alltoall-matrix-{timestamp}"
        )
    else:
        args.output_dir = args.output_dir.resolve()
    return args


def main() -> int:
    args = parse_args()
    output_dir: Path = args.output_dir
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    benchmarks = make_benchmarks(args)
    total = len(benchmarks) * len(args.gpus)
    results: list[Result] = []
    index = 0
    try:
        for gpus in args.gpus:
            for benchmark in benchmarks:
                index += 1
                result = run_one(args, benchmark, gpus, index, total, output_dir)
                results.append(result)
                if result.bandwidth_tib_s is not None:
                    print(f"  -> {result.bandwidth_tib_s:.3f} TiB/s")
                elif result.error:
                    print(f"  -> {result.status}: {result.error}")
                else:
                    print(f"  -> {result.status}")
    finally:
        csv_path = output_dir / "alltoall_bandwidth_results.csv"
        write_csv(results, csv_path)
        plot_path = None
        if not args.no_plot and not args.dry_run:
            plot_path = plot_results(results, args.gpus, output_dir / "alltoall_bandwidth.png")
        md_path = write_markdown(args, results, output_dir, csv_path, plot_path)
        print(f"\nWrote CSV: {csv_path}")
        print(f"Wrote Markdown summary: {md_path}")
        if plot_path is not None:
            print(f"Wrote plot: {plot_path}")

    failed = [result for result in results if result.status not in {"ok", "dry-run"}]
    return 1 if failed and not args.dry_run else 0


if __name__ == "__main__":
    raise SystemExit(main())
