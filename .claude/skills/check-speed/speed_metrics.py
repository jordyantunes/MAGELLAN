#!/usr/bin/env python
"""
Measure MAGELLAN training throughput (episodes/hour) from MLflow run metrics,
optionally cross-referenced with an nvidia-smi CSV trace, and compare against
previously recorded runs.

Usage:
    uv run .claude/skills/check-speed/speed_metrics.py <run_id> [--gpu-csv path.csv]
    uv run .claude/skills/check-speed/speed_metrics.py --list [--experiment magellan]
    uv run .claude/skills/check-speed/speed_metrics.py --compare
    uv run .claude/skills/check-speed/speed_metrics.py --changelog

Examples:
    uv run .claude/skills/check-speed/speed_metrics.py latest --gpu-csv gpu_report.csv
    uv run .claude/skills/check-speed/speed_metrics.py 2a717e4c09de4190b843ab55252a22da
    uv run .claude/skills/check-speed/speed_metrics.py --list
    uv run .claude/skills/check-speed/speed_metrics.py --compare
"""

import sys
import os
import csv
import json
import argparse
import datetime
import glob
import mlflow
import mlflow.tracking

# Local sqlite fallback; override with MLFLOW_TRACKING_URI (e.g. the Docker
# mlflow server at http://localhost:5000) when the venv's mlflow client is
# older than the schema of outputs/mlflow.db (alembic b7e4c1a90f23 error).
DB = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///outputs/mlflow.db")
FINDINGS_DIR = ".claude/findings/speed"
SPEED_METRIC = "test/grasp"  # any metric logged on a step==episode cadence works
THROUGHPUT_PARAMS = [
    "number_envs",
    "update_freq",
    "minibatch_size",
    "gradient_batch_size",
]


def _ts_str(ms):
    return datetime.datetime.utcfromtimestamp(ms / 1000).strftime("%Y%m%d_%H%M%S")


def _client():
    mlflow.set_tracking_uri(DB)
    return mlflow.tracking.MlflowClient()


def resolve_run_id(arg, experiment_name="magellan"):
    """Resolve 'latest' or a raw run_id/URL fragment to a run_id."""
    if arg != "latest":
        # Accept a full MLflow UI URL too.
        if "/runs/" in arg:
            arg = arg.split("/runs/")[1].split("/")[0]
        return arg

    client = _client()
    experiments = [e for e in client.search_experiments() if e.name == experiment_name]
    if not experiments:
        experiments = client.search_experiments()
    runs = client.search_runs(
        experiment_ids=[e.experiment_id for e in experiments],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        print("No runs found.", file=sys.stderr)
        sys.exit(1)
    return runs[0].info.run_id


def compute_ep_hr(run_id, metric=SPEED_METRIC):
    """
    Episodes/hour from first->last metric point (step == episode count).
    Returns None if fewer than 2 points are logged yet.
    """
    client = _client()
    hist = client.get_metric_history(run_id, metric)
    if len(hist) < 2:
        return None
    hist = sorted(hist, key=lambda h: h.step)
    s0, t0 = hist[0].step, hist[0].timestamp
    s1, t1 = hist[-1].step, hist[-1].timestamp
    dt_hr = (t1 - t0) / 1000 / 3600
    deps = s1 - s0
    if dt_hr <= 0:
        return None
    return {
        "metric_used": metric,
        "episode_start": s0,
        "episode_end": s1,
        "episodes": deps,
        "elapsed_hours": round(dt_hr, 4),
        "ep_per_hr": round(deps / dt_hr, 1),
        "n_points": len(hist),
    }


def run_params(run_id, keys=THROUGHPUT_PARAMS):
    run = _client().get_run(run_id)
    params = {k: run.data.params.get(k, "?") for k in keys}
    params["seed"] = run.data.params.get("seed", "?")
    params["goal_sampler"] = run.data.params.get("goal_sampler", "?")
    return params, run.info.start_time


def gpu_stats_from_csv(path):
    """
    Summarize an `nvidia-smi --query-gpu=... --format=csv` trace.
    Expected columns: timestamp, name, utilization.gpu [%], utilization.memory [%],
    memory.total [MiB], memory.free [MiB], memory.used [MiB]
    """
    util, mem_used = [], []
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) < 7:
                continue
            try:
                u = int(row[2].strip().replace("%", ""))
                m = int(row[6].strip().replace("MiB", ""))
            except ValueError:
                continue
            util.append(u)
            mem_used.append(m)
    if not util:
        return None
    n = len(util)
    zero = sum(1 for u in util if u == 0)
    low = sum(1 for u in util if u < 50)
    return {
        "source_csv": path,
        "n_samples": n,
        "util_avg_pct": round(sum(util) / n, 1),
        "util_min_pct": min(util),
        "util_max_pct": max(util),
        "mem_used_avg_mib": round(sum(mem_used) / n, 0),
        "mem_used_min_mib": min(mem_used),
        "mem_used_max_mib": max(mem_used),
        "zero_util_samples": zero,
        "zero_util_pct": round(100 * zero / n, 1),
        "low_util_samples_lt50": low,
        "low_util_pct": round(100 * low / n, 1),
    }


def list_runs(experiment_name="magellan"):
    client = _client()
    experiments = [e for e in client.search_experiments() if e.name == experiment_name]
    if not experiments:
        experiments = client.search_experiments()

    rows = []
    for exp in experiments:
        for run in client.search_runs(experiment_ids=[exp.experiment_id], order_by=["start_time DESC"]):
            speed = compute_ep_hr(run.info.run_id)
            params, start_ms = run_params(run.info.run_id)
            rows.append({
                "run_id": run.info.run_id[:8],
                "start": _ts_str(start_ms) if start_ms else "?",
                "ep_per_hr": speed["ep_per_hr"] if speed else "n/a",
                **params,
            })

    if not rows:
        print("No runs found.")
        return
    col_widths = {k: max(len(k), max(len(str(r[k])) for r in rows)) for k in rows[0]}
    header = "  ".join(k.ljust(col_widths[k]) for k in col_widths)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(str(r[k]).ljust(col_widths[k]) for k in col_widths))


def save_finding(run_id, speed, params, gpu=None, notes=None):
    os.makedirs(FINDINGS_DIR, exist_ok=True)
    ts = _ts_str(_client().get_run(run_id).info.start_time)
    out_path = os.path.join(FINDINGS_DIR, f"{ts}_{run_id}.json")
    payload = {
        "run_id": run_id,
        "recorded_at": datetime.datetime.utcnow().isoformat() + "Z",
        "start_time_str": ts,
        "params": params,
        "speed": speed,
        "gpu": gpu,
        "notes": notes,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[findings written to {out_path}]", file=sys.stderr)
    return out_path


def compare_all():
    """Print all saved speed findings sorted by ep_per_hr, fastest first."""
    files = sorted(glob.glob(os.path.join(FINDINGS_DIR, "*.json")))
    if not files:
        print("No saved speed findings yet. Run without --compare first to record one.", file=sys.stderr)
        return
    rows = []
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        speed = d.get("speed") or {}
        params = d.get("params") or {}
        rows.append({
            "run_id": d["run_id"][:8],
            "start": d.get("start_time_str", "?"),
            "ep_per_hr": speed.get("ep_per_hr", "n/a"),
            **{k: params.get(k, "?") for k in THROUGHPUT_PARAMS},
        })
    rows.sort(key=lambda r: r["ep_per_hr"] if isinstance(r["ep_per_hr"], (int, float)) else -1, reverse=True)
    col_widths = {k: max(len(k), max(len(str(r[k])) for r in rows)) for k in rows[0]}
    header = "  ".join(k.ljust(col_widths[k]) for k in col_widths)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(str(r[k]).ljust(col_widths[k]) for k in col_widths))


def generate_changelog(output_path=None):
    files = sorted(glob.glob(os.path.join(FINDINGS_DIR, "*.json")), reverse=True)
    if not files:
        print("No findings files found.", file=sys.stderr)
        return
    lines = ["# MAGELLAN Speed Log", "", "Newest first. Generated by `speed_metrics.py --changelog`.", ""]
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        speed = d.get("speed") or {}
        params = d.get("params") or {}
        gpu = d.get("gpu")
        lines.append("---")
        lines.append("")
        lines.append(
            f"## {d.get('start_time_str', '?')} UTC — `{d['run_id'][:8]}` "
            f"({params.get('goal_sampler','?')}, envs={params.get('number_envs','?')}, "
            f"update_freq={params.get('update_freq','?')}, "
            f"batch={params.get('minibatch_size','?')}/{params.get('gradient_batch_size','?')}, "
            f"seed={params.get('seed','?')})"
        )
        lines.append("")
        lines.append(f"**{speed.get('ep_per_hr', 'n/a')} ep/hr** "
                      f"({speed.get('episodes', '?')} episodes over {speed.get('elapsed_hours', '?')}h, "
                      f"{speed.get('n_points', '?')} data points)")
        lines.append("")
        if gpu:
            lines.append(
                f"GPU: avg util {gpu['util_avg_pct']}%, mem {gpu['mem_used_avg_mib']:.0f} MiB avg "
                f"({gpu['mem_used_min_mib']}-{gpu['mem_used_max_mib']} MiB), "
                f"{gpu['zero_util_pct']}% samples at 0% util, over {gpu['n_samples']} samples"
            )
            lines.append("")
        if d.get("notes"):
            lines.append(d["notes"])
            lines.append("")
    changelog = "\n".join(lines) + "\n"
    out = output_path or os.path.join(FINDINGS_DIR, "SPEED_LOG.md")
    with open(out, "w") as f:
        f.write(changelog)
    print(f"[speed log written to {out}]", file=sys.stderr)
    return out


def main():
    parser = argparse.ArgumentParser(description="Measure and compare MAGELLAN training throughput")
    parser.add_argument("run_id", nargs="?", help="MLflow run ID, URL, or 'latest'")
    parser.add_argument("--experiment", default="magellan", help="Experiment name filter for --list/'latest'")
    parser.add_argument("--gpu-csv", default=None, help="Path to an nvidia-smi --format=csv trace to summarize alongside the run")
    parser.add_argument("--metric", default=SPEED_METRIC, help=f"Metric to use as the episode-count clock (default: {SPEED_METRIC})")
    parser.add_argument("--list", action="store_true", help="List all runs with computed ep/hr")
    parser.add_argument("--compare", action="store_true", help="Print all previously saved speed findings, fastest first")
    parser.add_argument("--changelog", action="store_true", help="Regenerate .claude/findings/speed/SPEED_LOG.md")
    parser.add_argument("--no-save", action="store_true", help="Don't persist a findings file, just print")
    args = parser.parse_args()

    if args.list:
        list_runs(experiment_name=args.experiment)
        return
    if args.compare:
        compare_all()
        return
    if args.changelog:
        generate_changelog()
        return
    if not args.run_id:
        parser.print_help()
        sys.exit(1)

    run_id = resolve_run_id(args.run_id, experiment_name=args.experiment)
    speed = compute_ep_hr(run_id, metric=args.metric)
    params, _ = run_params(run_id)
    gpu = gpu_stats_from_csv(args.gpu_csv) if args.gpu_csv else None

    result = {"run_id": run_id, "params": params, "speed": speed, "gpu": gpu}
    print(json.dumps(result, indent=2))

    if speed and not args.no_save:
        save_finding(run_id, speed, params, gpu=gpu)


if __name__ == "__main__":
    main()
