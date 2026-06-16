#!/usr/bin/env python
"""
Query metric history and params from an MLflow run.

Usage:
    uv run .claude/skills/check-metrics/mlflow_metrics.py <run_id> [metric ...]
    uv run .claude/skills/check-metrics/mlflow_metrics.py --list [--experiment magellan]
    uv run .claude/skills/check-metrics/mlflow_metrics.py <run_id> --json
    uv run .claude/skills/check-metrics/mlflow_metrics.py <run_id> --summary

Examples:
    uv run .claude/skills/check-metrics/mlflow_metrics.py 5d2058d7f5b344deba1deea075b78bc1
    uv run .claude/skills/check-metrics/mlflow_metrics.py 5d2058d7f5b344deba1deea075b78bc1 test/grasp train/entropy
    uv run .claude/skills/check-metrics/mlflow_metrics.py 5d2058d7f5b344deba1deea075b78bc1 --json
    uv run .claude/skills/check-metrics/mlflow_metrics.py 5d2058d7f5b344deba1deea075b78bc1 --summary
    uv run .claude/skills/check-metrics/mlflow_metrics.py --list --experiment magellan
"""

import sys
import json
import argparse
import math
import os
import datetime
import mlflow
import mlflow.tracking

DB = "sqlite:///outputs/mlflow.db"

DEFAULT_METRICS = [
    "test/grasp",
    "test/grow_plants",
    "test/grow_herbivores",
    "test/grow_carnivores",
    "test/estimated_lp_grasp",
    "test/estimated_lp_grow_plants",
    "test/estimated_lp_grow_herbivores",
    "test/estimated_lp_grow_carnivores",
    "diag/raw_lp_grasp",
    "diag/raw_lp_grow_plants",
    "diag/raw_lp_grow_herbivores",
    "diag/raw_lp_grow_carnivores",
    "diag/sr_grasp",
    "diag/sr_impossibles",
    "train/policy_loss",
    "train/value_loss",
    "train/entropy",
    "train/alpha",
]


def _ts_str(ms):
    """Convert Unix millisecond timestamp to YYYYMMDD_HHMMSS string (UTC)."""
    return datetime.datetime.utcfromtimestamp(ms / 1000).strftime("%Y%m%d_%H%M%S")


def list_runs(experiment_name=None):
    mlflow.set_tracking_uri(DB)
    client = mlflow.tracking.MlflowClient()

    experiments = client.search_experiments()
    if experiment_name:
        experiments = [e for e in experiments if e.name == experiment_name]
    if not experiments:
        print("No experiments found.")
        return

    rows = []
    for exp in experiments:
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time ASC"],
        )
        for run in runs:
            ts = _ts_str(run.info.start_time) if run.info.start_time else "?"
            rows.append({
                "run_id": run.info.run_id,
                "experiment": exp.name,
                "start_time": ts,
                "status": run.info.status,
                "goal_sampler": run.data.params.get("goal_sampler", "?"),
                "num_episodes": run.data.params.get("num_episodes", "?"),
                "seed": run.data.params.get("seed", "?"),
                "grasp": run.data.metrics.get("test/grasp", ""),
                "grow_plants": run.data.metrics.get("test/grow_plants", ""),
            })

    col_widths = {k: max(len(k), max(len(str(r[k])) for r in rows)) for k in rows[0]}
    header = "  ".join(k.ljust(col_widths[k]) for k in col_widths)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(str(r[k]).ljust(col_widths[k]) for k in col_widths))


def fetch_run_data(run_id, metrics):
    mlflow.set_tracking_uri(DB)
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)

    history = {}
    for metric in metrics:
        hist = client.get_metric_history(run_id, metric)
        if hist:
            history[metric] = [{"step": h.step, "value": h.value} for h in hist]

    start_ms = run.info.start_time
    return {
        "run_id": run_id,
        "status": run.info.status,
        "start_time": start_ms,
        "start_time_str": _ts_str(start_ms) if start_ms else None,
        "end_time": run.info.end_time,
        "params": dict(run.data.params),
        "final_metrics": {k: v[-1]["value"] if v else None for k, v in history.items()},
        "history": history,
    }


THRESHOLDS = {
    "test/grasp": [0.5, 0.9, 1.0],
    "test/grow_plants": [0.05, 0.5, 1.0],
    "test/grow_herbivores": [0.05, 0.5, 1.0],
    "test/grow_carnivores": [0.05, 0.5, 1.0],
}


def _first_crossing(points, threshold):
    for p in points:
        if not math.isnan(p["value"]) and p["value"] >= threshold:
            return p["step"]
    return None


def _has_nan(points):
    return any(math.isnan(p["value"]) for p in points)


def _volatility(points, threshold=0.5):
    """
    Compute volatility stats for a metric curve.

    n_collapses: how many times the value dropped back below `threshold`
    after initially crossing it (counts distinct collapse events).
    fraction_above_last_quarter: fraction of points in the final 25% of
    the run where value >= threshold — 1.0 means stably solved, <1.0 means
    still oscillating.
    """
    valid = [p for p in points if not math.isnan(p["value"])]
    if not valid:
        return {}

    values = [p["value"] for p in valid]
    steps = [p["step"] for p in valid]

    max_value = max(values)
    step_of_max = steps[values.index(max_value)]

    first_cross_idx = next((i for i, v in enumerate(values) if v >= threshold), None)
    min_after_first_crossing = None
    if first_cross_idx is not None and first_cross_idx < len(values) - 1:
        after = values[first_cross_idx + 1:]
        min_after_first_crossing = min(after) if after else None

    n_collapses = 0
    if first_cross_idx is not None:
        above = False
        for v in values[first_cross_idx:]:
            if v >= threshold:
                above = True
            elif above:
                n_collapses += 1
                above = False

    quarter_start = steps[0] + (steps[-1] - steps[0]) * 0.75
    last_quarter = [p for p in valid if p["step"] >= quarter_start]
    fraction_above_last_quarter = None
    if last_quarter:
        fraction_above_last_quarter = sum(1 for p in last_quarter if p["value"] >= threshold) / len(last_quarter)

    return {
        "max_value": max_value,
        "step_of_max": step_of_max,
        "min_after_first_crossing": min_after_first_crossing,
        "n_collapses": n_collapses,
        "fraction_above_last_quarter": fraction_above_last_quarter,
        "collapse_threshold_used": threshold,
    }


def summarize(data):
    """Compact, pre-digested view — fits in a tool result buffer for 40k+ runs."""
    summary = {
        "run_id": data["run_id"],
        "status": data["status"],
        "start_time": data["start_time"],
        "start_time_str": data["start_time_str"],
        "end_time": data["end_time"],
        "params": data["params"],
        "final_metrics": data["final_metrics"],
        "milestones": {},
        "volatility": {},
        "trajectory": {},
        "lp_stats": {},
    }

    for metric, thresholds in THRESHOLDS.items():
        points = data["history"].get(metric, [])
        summary["milestones"][metric] = {
            f">={t}": _first_crossing(points, t) for t in thresholds
        }
        summary["volatility"][metric] = _volatility(points, threshold=0.5)

    for metric in ["train/entropy", "train/alpha", "train/value_loss", "train/policy_loss"]:
        points = data["history"].get(metric, [])
        if not points:
            continue
        summary["trajectory"][metric] = {
            "first_step": points[0]["step"] if points else None,
            "first_value": points[0]["value"] if points else None,
            "last_value": points[-1]["value"] if points else None,
            "has_nan": _has_nan(points),
            "n_points": len(points),
        }

    for metric in [k for k in data["history"] if k.split("/")[-1].startswith("lp") or "_lp_" in k]:
        points = data["history"].get(metric, [])
        nonzero = [p for p in points if not math.isnan(p["value"]) and p["value"] > 0.001]
        summary["lp_stats"][metric] = {
            "first": points[0]["value"] if points else None,
            "last": points[-1]["value"] if points else None,
            "nonzero_count": len(nonzero),
            "first_nonzero_step": nonzero[0]["step"] if nonzero else None,
        }

    for metric in ["diag/sr_impossibles"]:
        points = data["history"].get(metric, [])
        if points:
            summary["trajectory"][metric] = {
                "first_value": points[0]["value"],
                "last_value": points[-1]["value"],
                "has_nan": _has_nan(points),
            }

    return summary


def generate_changelog(findings_dir=".claude/findings", output_path=None):
    """
    Read all <timestamp>_<run_id>.json findings files and write CHANGELOG.md.
    Files are processed in filename order (which is chronological by timestamp prefix).
    """
    import glob
    pattern = os.path.join(findings_dir, "[0-9]*_*.json")
    files = sorted((f for f in glob.glob(pattern) if "_history" not in f), reverse=True)

    if not files:
        print("No findings files found.", file=sys.stderr)
        return

    lines = ["# MAGELLAN Experiment Changelog", ""]
    lines.append("Newest experiments first. Generated by `mlflow_metrics.py --changelog`.")
    lines.append("")

    for fpath in files:
        with open(fpath) as fh:
            try:
                d = json.load(fh)
            except json.JSONDecodeError:
                continue

        run_id = d.get("run_id", "unknown")
        ts = d.get("start_time_str", "?")
        params = d.get("params", {})
        fm = d.get("final_metrics", {})
        summary_text = d.get("summary", "")
        findings = d.get("findings", [])
        comparison = d.get("comparison")

        sampler = params.get("goal_sampler", "?")
        episodes = params.get("num_episodes", "?")
        envs = params.get("number_envs", "?")
        seed = params.get("seed", "?")
        buf = params.get("buffer_size", "?")

        # Format timestamp as readable date
        readable_ts = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]} {ts[9:11]}:{ts[11:13]}" if len(ts) >= 13 else ts

        lines.append(f"---")
        lines.append(f"")
        lines.append(f"## {readable_ts} UTC — `{run_id[:8]}` ({sampler}, {episodes} ep, {envs} envs, seed={seed}, buffer={buf})")
        lines.append("")

        # Key metrics table
        if fm:
            grasp = fm.get("test/grasp")
            grow_p = fm.get("test/grow_plants")
            grow_h = fm.get("test/grow_herbivores")
            grow_c = fm.get("test/grow_carnivores")
            entropy = fm.get("train/entropy")
            alpha = fm.get("train/alpha")

            def _fmt(v):
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    return "NaN"
                return f"{v:.3f}" if isinstance(v, float) else str(v)

            lines.append(f"| grasp | grow_plants | grow_herb | grow_carn | entropy | alpha |")
            lines.append(f"|---|---|---|---|---|---|")
            lines.append(f"| {_fmt(grasp)} | {_fmt(grow_p)} | {_fmt(grow_h)} | {_fmt(grow_c)} | {_fmt(entropy)} | {_fmt(alpha)} |")
            lines.append("")

        if summary_text:
            lines.append(summary_text)
            lines.append("")

        if comparison:
            lines.append(f"**vs. prior run:** {comparison}")
            lines.append("")

        # Findings — one line each, bugs first
        bugs = [f for f in findings if f.get("severity") == "BUG"]
        investigates = [f for f in findings if f.get("severity") == "INVESTIGATE"]
        for f in bugs + investigates:
            icon = "**BUG**" if f["severity"] == "BUG" else "**INVESTIGATE**"
            metric = f.get("metric") or ""
            desc = f.get("description", "")
            lines.append(f"- {icon} `{metric}`: {desc}" if metric else f"- {icon}: {desc}")
        if bugs or investigates:
            lines.append("")

    changelog = "\n".join(lines) + "\n"
    out = output_path or os.path.join(findings_dir, "CHANGELOG.md")
    with open(out, "w") as fh:
        fh.write(changelog)
    print(f"[changelog written to {out}]", file=sys.stderr)
    return out


def show_text(data):
    print(f"Run:    {data['run_id']}")
    print(f"Status: {data['status']}")
    print()
    print("=== PARAMS ===")
    for k, v in sorted(data["params"].items()):
        print(f"  {k}: {v}")
    print()
    print("=== METRIC HISTORY ===")
    for metric, points in data["history"].items():
        print(f"{metric}:")
        for p in points:
            val = p["value"]
            val_str = "nan" if math.isnan(val) else f"{val:.6f}"
            print(f"  ep {p['step']:7d}: {val_str}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Query MLflow run metrics and params")
    parser.add_argument("run_id", nargs="?", help="MLflow run ID")
    parser.add_argument("metrics", nargs="*", help="Metrics to display (default: standard training set)")
    parser.add_argument("--list", action="store_true", help="List all runs")
    parser.add_argument("--experiment", default=None, help="Filter --list by experiment name")
    parser.add_argument("--json", action="store_true", help="Output full history as JSON")
    parser.add_argument("--summary", action="store_true", help="Output compact pre-digested JSON (recommended for skill use; avoids buffer overflow on long runs)")
    parser.add_argument("--changelog", action="store_true", help="Regenerate .claude/findings/CHANGELOG.md from all findings files")
    args = parser.parse_args()

    if args.list:
        list_runs(experiment_name=args.experiment)
        return

    if args.changelog:
        generate_changelog()
        return

    if not args.run_id:
        parser.print_help()
        sys.exit(1)

    metrics = args.metrics if args.metrics else DEFAULT_METRICS
    data = fetch_run_data(args.run_id, metrics)

    if args.summary:
        ts = data["start_time_str"] or "unknown"
        findings_dir = ".claude/findings"
        os.makedirs(findings_dir, exist_ok=True)
        history_path = os.path.join(findings_dir, f"{ts}_{args.run_id}_history.json")
        with open(history_path, "w") as f:
            json.dump({"run_id": args.run_id, "start_time_str": ts, "history": data["history"]}, f, indent=2)
        print(json.dumps(summarize(data), indent=2), file=sys.stdout)
        print(f"[history written to {history_path}]", file=sys.stderr)
    elif args.json:
        print(json.dumps(data, indent=2))
    else:
        show_text(data)


if __name__ == "__main__":
    main()
