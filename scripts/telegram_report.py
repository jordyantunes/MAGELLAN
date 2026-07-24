#!/usr/bin/env python
"""
Send a Telegram progress report for the currently running MAGELLAN experiment.

Reads the MLflow tracking server (assumed running in Docker on localhost:5000,
see docker-compose.local.yml) rather than the sqlite file directly, since the
local venv's mlflow version doesn't match the one that created outputs/mlflow.db.

Usage:
    uv run scripts/telegram_report.py [--experiment magellan] [--dry-run]
    uv run scripts/telegram_report.py --raw [--experiment magellan]
    echo "message text" | uv run scripts/telegram_report.py --send

Requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID (loaded from .env if present).
Keeps small state in scripts/.telegram_report_state.json to report deltas
since the last check and to detect a stalled run (no new steps).

--raw prints a compact JSON blob (deltas, downsampled recent history for trend
shape, rule-based hints, milestones) for an LLM to turn into a narrative report,
and updates state, but does not send anything to Telegram. It also appends a
snapshot to scripts/.telegram_history.jsonl and includes the last few snapshots
plus the last few sent messages (scripts/.telegram_messages.jsonl) in its output,
so a caller can see what already changed and what was already said, and report
only what's new instead of repeating stable facts every cycle.

--send reads a message from stdin and sends it verbatim via the Telegram bot,
then appends it to scripts/.telegram_messages.jsonl. No metrics fetching. Pair
it with --raw for an LLM-authored report: fetch with --raw, compose text, pipe
that text into --send.
"""

import argparse
import json
import math
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MLFLOW_URL = "http://localhost:5000"
SCRIPT_DIR = Path(__file__).resolve().parent
STATE_PATH = SCRIPT_DIR / ".telegram_report_state.json"
HISTORY_PATH = SCRIPT_DIR / ".telegram_history.jsonl"
MESSAGES_PATH = SCRIPT_DIR / ".telegram_messages.jsonl"
HISTORY_MAX_LINES = 40
MESSAGES_MAX_LINES = 10

KEY_METRICS = [
    "test/grasp",
    "test/grow_plants",
    "test/grow_herbivores",
    "test/grow_carnivores",
]
TRAIN_METRICS = ["train/entropy", "train/alpha", "train/alpha_loss", "train/policy_loss", "train/value_loss"]
HISTORY_METRICS = ["train/entropy", "train/value_loss"]
COLLAPSE_THRESHOLD = 0.5
SR_IMPOSSIBLES_STUCK_EPISODE = 20000
SR_IMPOSSIBLES_STUCK_VALUE = 0.3


def read_jsonl_tail(path, n):
    if not path.exists():
        return []
    lines = path.read_text().splitlines()[-n:]
    return [json.loads(l) for l in lines if l.strip()]


def append_jsonl(path, obj, max_lines):
    lines = path.read_text().splitlines() if path.exists() else []
    lines.append(json.dumps(obj))
    lines = lines[-max_lines:]
    path.write_text("\n".join(lines) + "\n")


def load_env():
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def api_post(path, payload):
    req = urllib.request.Request(
        f"{MLFLOW_URL}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.load(resp)


def api_get(path):
    with urllib.request.urlopen(f"{MLFLOW_URL}{path}", timeout=10) as resp:
        return json.load(resp)


def get_metric_history(run_id, metric_key):
    data = api_get(f"/api/2.0/mlflow/metrics/get-history?run_id={run_id}&metric_key={metric_key}")
    return data.get("metrics", [])


def downsample(points, n=20):
    """Evenly-spaced sample of {step, value} points, always including first/last."""
    if len(points) <= n:
        return [{"step": p["step"], "value": p["value"]} for p in points]
    idxs = sorted({round(i * (len(points) - 1) / (n - 1)) for i in range(n)})
    return [{"step": points[i]["step"], "value": points[i]["value"]} for i in idxs]


def find_experiment_id(name):
    data = api_post("/api/2.0/mlflow/experiments/search", {"max_results": 50})
    for exp in data.get("experiments", []):
        if exp["name"] == name:
            return exp["experiment_id"]
    raise SystemExit(f"No experiment named '{name}' found on the MLflow server")


def find_latest_run(experiment_id):
    data = api_post(
        "/api/2.0/mlflow/runs/search",
        {"experiment_ids": [experiment_id], "max_results": 5, "order_by": ["start_time DESC"]},
    )
    runs = data.get("runs", [])
    if not runs:
        return None
    running = [r for r in runs if r["info"]["status"] == "RUNNING"]
    return (running or runs)[0]


def metrics_dict(run):
    return {m["key"]: (m["value"], m["step"]) for m in run["data"]["metrics"]}


MILESTONE_THRESHOLDS = {
    "test/grasp": [0.5, 0.9, 1.0],
    "test/grow_plants": [0.05, 0.5, 1.0],
    "test/grow_herbivores": [0.05, 0.5, 1.0],
    "test/grow_carnivores": [0.05, 0.5, 1.0],
}


def check_milestones(metrics, step, prev_crossed):
    """Return (good_news_lines, updated_crossed) for thresholds newly reached this check."""
    good = []
    crossed = {k: list(v) for k, v in prev_crossed.items()}
    for metric, thresholds in MILESTONE_THRESHOLDS.items():
        v, _ = metrics.get(metric, (None, None))
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        already = set(crossed.get(metric, []))
        for t in thresholds:
            if v >= t and t not in already:
                crossed.setdefault(metric, []).append(t)
                pct = int(t * 100)
                good.append(f"🎉 {metric.split('/')[-1]} crossed {pct}% (ep {step})")
    return good, crossed


def check_collapses(metrics, prev_crossed):
    """Flag a metric that had crossed 50% before but has now fallen well below it."""
    warnings = []
    for metric in MILESTONE_THRESHOLDS:
        v, _ = metrics.get(metric, (None, None))
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        crossed_50 = 0.5 in set(prev_crossed.get(metric, []))
        if crossed_50 and v < 0.3:
            warnings.append(f"📉 {metric.split('/')[-1]} regressed to {v:.3f} after previously passing 50% (possible forgetting)")
    return warnings


def check_history_anomalies(run_id):
    """Fetch recent history for a few train metrics and flag NaNs / sudden shifts vs. their own recent baseline."""
    warnings = []
    for metric in HISTORY_METRICS:
        points = get_metric_history(run_id, metric)
        if not points:
            continue
        values = [p["value"] for p in points]
        if any(isinstance(v, float) and math.isnan(v) for v in values):
            warnings.append(f"⚠️ NaN detected in recent `{metric}` history")
            continue
        if len(values) < 20:
            continue
        split = max(1, int(len(values) * 0.9))
        baseline, recent = values[:split], values[split:]
        if not baseline or not recent:
            continue
        base_avg = sum(baseline) / len(baseline)
        recent_avg = sum(recent) / len(recent)
        if metric == "train/entropy" and base_avg > 0.05 and recent_avg < 0.4 * base_avg:
            warnings.append(f"⚠️ train/entropy dropped sharply ({base_avg:.3f} → {recent_avg:.3f}) — policy may be collapsing")
        if metric == "train/value_loss":
            recent_max = max(recent)
            base_median = sorted(baseline)[len(baseline) // 2]
            if base_median > 1e-6 and recent_max > 5 * base_median and recent_max > 0.01:
                warnings.append(f"⚠️ train/value_loss spiked to {recent_max:.4f} (baseline median {base_median:.4f})")
    return warnings


def fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, float) and math.isnan(v):
        return "NaN"
    if v != 0 and abs(v) < 0.001:
        return f"{v:.2e}"
    return f"{v:.3f}"


def send_telegram(text):
    """Plain text, deliberately no parse_mode: Telegram's Markdown parser errors out
    on unbalanced '_'/'*' — which metric names like `grow_plants` and free-form,
    LLM-authored messages trigger constantly."""
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=json.dumps({"chat_id": chat_id, "text": text}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.load(resp)


def gather(experiment, persist_state):
    """Fetch the latest run + metrics, run the rule-based checks, and (optionally)
    persist state for delta/milestone tracking. Returns None if there's nothing to
    report (unreachable MLflow / no runs), in which case `error` explains why."""
    try:
        exp_id = find_experiment_id(experiment)
        run = find_latest_run(exp_id)
    except Exception as e:
        return {"error": f"couldn't reach MLflow ({e})"}

    if run is None:
        return {"error": f"no runs found in experiment '{experiment}'"}

    run_id = run["info"]["run_id"]
    status = run["info"]["status"]
    params = {p["key"]: p["value"] for p in run["data"].get("params", [])}
    metrics = metrics_dict(run)
    step = max((s for _, s in metrics.values()), default=0)

    state = {}
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text())
    prev = state.get(run_id, {})
    prev_step = prev.get("step")
    prev_crossed = prev.get("crossed", {})
    stalled = prev_step is not None and prev_step == step and status == "RUNNING"

    good_news, crossed = check_milestones(metrics, step, prev_crossed)
    warnings = check_collapses(metrics, prev_crossed)
    warnings += check_history_anomalies(run_id)

    sr_imposs, _ = metrics.get("test/estimated_sr_impossibles", (None, None))
    if sr_imposs is not None and step >= SR_IMPOSSIBLES_STUCK_EPISODE and sr_imposs >= SR_IMPOSSIBLES_STUCK_VALUE:
        warnings.append(f"⚠️ SR head not distinguishing impossible goals (sr_impossibles={sr_imposs:.3f} at ep {step})")

    if stalled:
        warnings.append("🛑 no new episodes since last check — run may be stalled or finished")

    if persist_state:
        state[run_id] = {
            "step": step,
            "metrics": {m: metrics[m][0] for m in KEY_METRICS if m in metrics},
            "crossed": crossed,
            "last_check": time.time(),
        }
        STATE_PATH.write_text(json.dumps(state, indent=2))

    return {
        "run_id": run_id,
        "status": status,
        "params": params,
        "metrics": metrics,
        "step": step,
        "prev_step": prev_step,
        "prev": prev,
        "good_news": good_news,
        "warnings": warnings,
    }


def render_text(data):
    run_id, status, params, metrics, step, prev_step, prev = (
        data["run_id"], data["status"], data["params"], data["metrics"],
        data["step"], data["prev_step"], data["prev"],
    )
    sampler = params.get("goal_sampler", "?")
    seed = params.get("seed", "?")

    lines = [f"MAGELLAN report — {run_id[:8]} ({sampler}, seed={seed})"]
    lines.append(f"status: {status} | episode: {step}" + (f" (+{step - prev_step})" if prev_step is not None else ""))
    lines.append("")
    lines.append("test SR")
    for m in KEY_METRICS:
        v, _ = metrics.get(m, (None, None))
        prev_v = prev.get("metrics", {}).get(m)
        delta = f" ({v - prev_v:+.3f})" if v is not None and prev_v is not None else ""
        lines.append(f"  {m.split('/')[-1]}: {fmt(v)}{delta}")
    lines.append("")
    lines.append("train")
    for m in TRAIN_METRICS:
        v, _ = metrics.get(m, (None, None))
        lines.append(f"  {m.split('/')[-1]}: {fmt(v)}")

    if data["good_news"]:
        lines.append("")
        lines.append("progress")
        lines.extend(f"  {g}" for g in data["good_news"])

    if data["warnings"]:
        lines.append("")
        lines.append("anomalies")
        lines.extend(f"  {w}" for w in data["warnings"])

    return "\n".join(lines)


def render_raw(data):
    run_id, status, params, metrics, step, prev_step, prev = (
        data["run_id"], data["status"], data["params"], data["metrics"],
        data["step"], data["prev_step"], data["prev"],
    )

    test_sr = {}
    for m in KEY_METRICS:
        v, _ = metrics.get(m, (None, None))
        prev_v = prev.get("metrics", {}).get(m)
        test_sr[m.split("/")[-1]] = {
            "value": v,
            "delta": (v - prev_v) if v is not None and prev_v is not None else None,
        }

    train = {m.split("/")[-1]: metrics.get(m, (None, None))[0] for m in TRAIN_METRICS}

    train_trend = {}
    for m in ["train/entropy", "train/value_loss", "train/policy_loss", "train/alpha"]:
        points = get_metric_history(run_id, m)
        train_trend[m] = downsample(points, n=20)

    prior_history = read_jsonl_tail(HISTORY_PATH, 6)
    prior_messages = [m["message"] for m in read_jsonl_tail(MESSAGES_PATH, 3)]

    metrics_lines = [f"ep {step} ({run_id[:8]}, {params.get('goal_sampler', '?')})"]
    for m in KEY_METRICS:
        v, _ = metrics.get(m, (None, None))
        prev_v = prev.get("metrics", {}).get(m)
        delta = f" ({v - prev_v:+.3f})" if v is not None and prev_v is not None else ""
        metrics_lines.append(f"{m.split('/')[-1]}: {fmt(v)}{delta}")
    for m in TRAIN_METRICS:
        v, _ = metrics.get(m, (None, None))
        metrics_lines.append(f"{m.split('/')[-1]}: {fmt(v)}")
    metrics_block = "\n".join(metrics_lines)

    raw = {
        "run_id": run_id,
        "status": status,
        "sampler": params.get("goal_sampler", "?"),
        "seed": params.get("seed", "?"),
        "episode": step,
        "episodes_since_last_check": (step - prev_step) if prev_step is not None else None,
        "test_sr": test_sr,
        "train": train,
        "train_trend": train_trend,
        "milestones_this_check": data["good_news"],
        "hints": data["warnings"],
        "prior_checks": prior_history,
        "previously_sent_messages": prior_messages,
        "metrics_block": metrics_block,
    }

    append_jsonl(
        HISTORY_PATH,
        {
            "timestamp": time.time(),
            "episode": step,
            "test_sr": {k: v["value"] for k, v in test_sr.items()},
            "train": train,
            "hints": data["warnings"],
        },
        HISTORY_MAX_LINES,
    )

    return raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="magellan")
    parser.add_argument("--dry-run", action="store_true", help="Print the report instead of sending it")
    parser.add_argument("--raw", action="store_true", help="Print compact JSON for an LLM to narrate, instead of sending a formatted report")
    parser.add_argument("--send", action="store_true", help="Read a message from stdin and send it verbatim to Telegram")
    args = parser.parse_args()

    load_env()

    if args.send:
        text = sys.stdin.read().strip()
        if not text:
            sys.exit("no message on stdin")
        send_telegram(text)
        append_jsonl(MESSAGES_PATH, {"timestamp": time.time(), "message": text}, MESSAGES_MAX_LINES)
        return

    data = gather(args.experiment, persist_state=args.raw or not args.dry_run)

    if "error" in data:
        msg = f"⚠️ MAGELLAN monitor: {data['error']}"
        print(msg)
        if not args.dry_run:
            send_telegram(msg)
        sys.exit(0)

    if args.raw:
        print(json.dumps(render_raw(data), indent=2))
        return

    report = render_text(data)
    print(report)

    if args.dry_run:
        return

    send_telegram(report)


if __name__ == "__main__":
    main()
