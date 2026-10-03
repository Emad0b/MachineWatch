"""Run-to-failure input and causal, history-only window features."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

SENSORS = ["temperature", "vibration", "pressure"]
WINDOW = 10
FEATURES = ["cycle"] + [f"{s}_{stat}" for stat in ["last", "mean", "slope"] for s in SENSORS]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def simulate(machines=60, seed=42):
    """Engineered synthetic signals, not a physics simulator or real failures."""
    if machines < 10:
        raise ValueError("At least ten machines are required")
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(machines):
        lifetime = int(rng.integers(140, 241))
        offset = rng.normal(0, [5, 0.15, 3])
        for cycle in range(1, lifetime + 1):
            age = cycle / lifetime
            values = np.array([45 + 80 * age**2, 0.5 + 5 * age**3, 70 - 30 * age])
            values += offset + rng.normal(0, [3, 0.18, 2])
            rows.append({"machine_id": f"M{index + 1:03d}", "cycle": cycle,
                         **dict(zip(SENSORS, (float(x) for x in values)))})
    return rows


def save_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["machine_id", "cycle", *SENSORS])
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            cycle = float(row["cycle"])
            if not cycle.is_integer():
                raise ValueError("Cycles must be integers")
            rows.append({"machine_id": row["machine_id"], "cycle": int(cycle),
                         **{s: float(row[s]) for s in SENSORS}})
    return rows


def validate_history(history):
    if len(history) < WINDOW:
        raise ValueError(f"At least {WINDOW} readings are required")
    cycles = [r["cycle"] for r in history]
    if any(not isinstance(c, int) or c < 1 for c in cycles) or any(a >= b for a, b in zip(cycles, cycles[1:])):
        raise ValueError("Cycles must be positive, unique and increasing")
    if not np.isfinite([[r[s] for s in SENSORS] for r in history]).all():
        raise ValueError("Sensors must be finite")


def features(history):
    validate_history(history)
    window = history[-WINDOW:]
    values = np.array([[r[s] for s in SENSORS] for r in window], dtype=float)
    delta = window[-1]["cycle"] - window[0]["cycle"]
    return np.concatenate(([window[-1]["cycle"]], values[-1], values.mean(axis=0), (values[-1] - values[0]) / delta))


def make_examples(rows, horizon=20):
    if horizon < 1:
        raise ValueError("Horizon must be positive")
    machines = {}
    for row in rows:
        if not row["machine_id"]:
            raise ValueError("Missing machine ID")
        machines.setdefault(row["machine_id"], []).append(row)
    x, rul, ids, cycles = [], [], [], []
    for machine_id, history in sorted(machines.items()):
        history = sorted(history, key=lambda r: r["cycle"])
        validate_history(history)
        if history[-1]["cycle"] - history[WINDOW - 1]["cycle"] <= horizon:
            raise ValueError("Each run-to-failure machine needs healthy and pre-failure windows")
        # Terminal cycle is used only to make labels, never as an input feature.
        terminal = history[-1]["cycle"]
        for end in range(WINDOW, len(history) + 1):
            x.append(features(history[end - WINDOW:end]))
            rul.append(terminal - history[end - 1]["cycle"])
            ids.append(machine_id)
            cycles.append(history[end - 1]["cycle"])
    if not x:
        raise ValueError("Empty dataset")
    rul = np.array(rul)
    return np.array(x), (rul <= horizon).astype(int), rul, np.array(ids), np.array(cycles), machines


def split_machines(ids, seed=42):
    machines = np.array(sorted(set(ids)))
    if len(machines) < 10:
        raise ValueError("At least ten distinct machines required for three held-out groups")
    machines = np.random.default_rng(seed).permutation(machines)
    holdout = max(2, len(machines) // 5)
    groups = {"test": sorted(machines[:holdout].tolist()),
              "validation": sorted(machines[holdout:2 * holdout].tolist()),
              "train": sorted(machines[2 * holdout:].tolist())}
    return {name: np.isin(ids, values) for name, values in groups.items()}, groups
