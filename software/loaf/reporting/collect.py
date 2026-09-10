"""Assembles the data a LOAF training-run HTML report is built from.

Reads a run directory written by loaf.training.Trainer (train_log.csv plus
best.pt/last.pt) and, when the training data is still available at
`data_dir`, re-runs the checkpoint's own validation split through the model
to recover the per-horizon/per-variable/scatter/residual detail that
RunningMetrics' running sums don't keep around (see loaf.training.evaluate).

A checkpoint written by loaf.pipeline.train_stage is self-describing enough
(region bounds, back_hrs, val_split, etc. - see "run_config") to rebuild the
exact same validation split without the original YAML config. The split
itself is chronological (see create_dataloaders), so it's deterministic
given just that metadata - no RNG state needs to be recovered.
"""

from __future__ import annotations

import csv
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from loaf.data.loaders import create_dataloaders
from loaf.training.evaluate import RunningMetrics
from loaf.training.trainer import build_model

# Cap on how many (actual, predicted) pairs a report embeds per variable, so
# the HTML file stays a reasonable size regardless of how much validation
# data a run has. Sampled uniformly (reservoir sampling) rather than
# truncated, so the scatter/residual plots stay representative.
MAX_SAMPLE_POINTS = 4000


@dataclass
class ReportData:
    """Everything render_html() needs to draw a training-run report."""

    run_name: str
    generated_at: str
    checkpoint_path: str
    model_type: str
    lead_times: list[int]
    target_vars: list[str]
    station_vars: list[str]
    hyperparams: dict[str, Any]
    training_curve: list[dict[str, float]]
    best_epoch: int | None
    best_val_loss: float | None
    final_metrics: dict[str, float]
    per_horizon: dict[str, list[dict[str, float]]]
    scatter: dict[str, list[list[float]]]
    residuals: dict[str, dict[str, Any]]
    notes: list[dict[str, str]]
    inference_error: str | None = None


class _ReservoirSampler:
    """Uniform reservoir sample of (actual, predicted) pairs, capped at `cap`.

    Standard Algorithm R: guarantees every point seen so far has equal
    probability of being in the sample, without needing to know the total
    count up front or hold every point in memory.
    """

    def __init__(self, cap: int, seed: int = 0):
        self.cap = cap
        self._rng = random.Random(seed)
        self._seen = 0
        self.items: list[tuple[float, float]] = []

    def offer(self, actual: float, predicted: float) -> None:
        self._seen += 1
        if len(self.items) < self.cap:
            self.items.append((actual, predicted))
        else:
            j = self._rng.randint(0, self._seen - 1)
            if j < self.cap:
                self.items[j] = (actual, predicted)


def _read_train_log(path: Path) -> list[dict[str, float]]:
    """Read Trainer's per-epoch train_log.csv, or [] if the run has none."""
    if not path.exists():
        return []
    rows: list[dict[str, float]] = []
    with open(path, newline="") as f:
        for raw_row in csv.DictReader(f):
            row = {k: float(v) for k, v in raw_row.items()}
            row["epoch"] = int(row["epoch"])
            rows.append(row)
    return rows


def _forward(
    model: torch.nn.Module,
    model_type: str,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    n_lead_times: int,
    n_target_vars: int,
) -> torch.Tensor:
    """Run the model and reshape output to (batch, n_stations, n_lead_times, n_vars).

    Mirrors Trainer._forward - kept as a standalone function here (like
    Predictor.predict_all's own copy of this dispatch) so building a report
    doesn't require constructing a full Trainer.
    """
    madis_x = batch["madis_x"].to(device)
    ex_x = batch.get("ex_x")

    if model_type == "mpnn":
        madis_lon = batch["madis_lon"].to(device)
        madis_lat = batch["madis_lat"].to(device)
        edge_index = batch["edge_index"].to(device)

        ex_lon = ex_lat = edge_index_e2m = None
        if ex_x is not None:
            ex_x = ex_x.to(device)
            ex_lon = batch["ex_lon"].to(device)
            ex_lat = batch["ex_lat"].to(device)
            edge_index_e2m = batch["edge_index_e2m"].to(device)

        preds = model(
            madis_x, madis_lon, madis_lat, edge_index, ex_lon, ex_lat, ex_x, edge_index_e2m
        )
    else:  # vit
        if ex_x is not None:
            ex_x = ex_x.to(device)
        preds, _ = model(madis_x, era5_x=ex_x)

    n_batch, n_stations, _ = preds.shape
    return preds.view(n_batch, n_stations, n_lead_times, n_target_vars)


def _persistence_baseline(
    batch: dict[str, torch.Tensor],
    station_vars: list[str],
    target_vars: list[str],
    n_lead_times: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Naive "no change from now" forecast, for the skill-score baseline."""
    try:
        indices = [station_vars.index(var) for var in target_vars]
    except ValueError:
        return None
    madis_x = batch["madis_x"].to(device)
    last_values = madis_x[:, :, -1, indices]  # (batch, n_stations, n_target_vars)
    return last_values.unsqueeze(2).expand(-1, -1, n_lead_times, -1)


def _denormalize(
    values: torch.Tensor, var: str, stats: dict[str, dict[str, float]]
) -> torch.Tensor:
    var_stats = stats.get(var)
    if var_stats is None:
        return values
    eps = 1e-5
    return values * (var_stats["max"] - var_stats["min"] + eps) + var_stats["min"]


def _denormalize_targets(
    values: torch.Tensor, target_vars: list[str], stats: dict[str, dict[str, float]]
) -> torch.Tensor:
    if not stats:
        return values
    slices = [_denormalize(values[..., i], var, stats) for i, var in enumerate(target_vars)]
    return torch.stack(slices, dim=-1)


def run_inference(
    checkpoint: dict[str, Any],
    data_dir: str | Path,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 0,
) -> dict[str, Any]:
    """Re-run a checkpoint's saved validation split through the model.

    Returns per-horizon/per-variable metrics (incl. skill vs a persistence
    baseline), plus subsampled scatter/residual data for the report's plots.

    Raises whatever create_dataloaders()/torch raise if the run's training
    data isn't available at data_dir (e.g. a checkpoint copied off the
    training machine) - callers should catch and fall back to a
    metrics-only report built from train_log.csv alone.
    """
    run_config: dict[str, Any] = checkpoint.get("run_config") or {}
    lead_times: list[int] = checkpoint["lead_times"]
    target_vars: list[str] = checkpoint["target_vars"]
    station_vars: list[str] = checkpoint["station_vars"]
    station_stats: dict[str, dict[str, float]] = (
        checkpoint.get("station_stats") or checkpoint.get("target_stats") or {}
    )

    year = run_config.get("year")
    data_dir = Path(data_dir)
    if year is None:
        from loaf.pipeline import infer_year

        year = infer_year(data_dir / "iem")

    bundle = create_dataloaders(
        data_dir=data_dir,
        year=year,
        back_hrs=run_config.get("back_hrs", 24),
        lead_times=lead_times,
        batch_size=batch_size,
        val_split=run_config.get("val_split", 0.15),
        num_workers=num_workers,
        lat_bounds=run_config.get("lat_bounds"),
        lon_bounds=run_config.get("lon_bounds"),
        use_era5=run_config.get("use_era5", False),
        use_hrrr=run_config.get("use_hrrr", False),
        min_observations=run_config.get("min_observations", 24),
    )

    model_cfg = dict(run_config.get("model_cfg") or {})
    model_cfg.setdefault("type", checkpoint["model_type"])
    model = build_model(
        model_cfg,
        n_stations=bundle.dataset.n_stations,
        in_hrs=run_config.get("back_hrs", 24),
        n_station_vars=len(bundle.dataset.station_vars),
        n_out_features=len(lead_times) * len(target_vars),
        grid_vars=bundle.dataset.grid_vars if bundle.dataset.grid_loader is not None else None,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    n_lead, n_vars = len(lead_times), len(target_vars)
    metrics = RunningMetrics(lead_times, target_vars)
    samplers = {
        var: _ReservoirSampler(MAX_SAMPLE_POINTS, seed=i) for i, var in enumerate(target_vars)
    }

    with torch.no_grad():
        for batch in bundle.val_loader:
            preds = _forward(model, model_cfg["type"], batch, device, n_lead, n_vars)
            target = batch["target"].to(device)
            mask = batch["target_mask"].to(device)
            persistence = _persistence_baseline(batch, station_vars, target_vars, n_lead, device)

            d_preds = _denormalize_targets(preds, target_vars, station_stats)
            d_target = _denormalize_targets(target, target_vars, station_stats)
            d_persistence = (
                _denormalize_targets(persistence, target_vars, station_stats)
                if persistence is not None
                else None
            )
            metrics.update(d_preds, d_target, mask, persistence=d_persistence)

            mask_np = mask.cpu().numpy() > 0
            preds_np = d_preds.cpu().numpy()
            target_np = d_target.cpu().numpy()
            for vi, var in enumerate(target_vars):
                for b, s, lead_idx in np.argwhere(mask_np[..., vi]):
                    samplers[var].offer(
                        float(target_np[b, s, lead_idx, vi]), float(preds_np[b, s, lead_idx, vi])
                    )

    computed = metrics.compute()
    per_horizon = {
        var: [
            {
                "lead_hr": lead_hr,
                "rmse": computed[f"rmse_{var}_{lead_hr}h"],
                "mae": computed[f"mae_{var}_{lead_hr}h"],
                "skill": computed[f"skill_{var}_{lead_hr}h"],
            }
            for lead_hr in lead_times
        ]
        for var in target_vars
    }

    scatter = {var: [[a, p] for a, p in samplers[var].items] for var in target_vars}
    residuals: dict[str, dict[str, Any]] = {}
    for var in target_vars:
        errors = np.array([p - a for a, p in samplers[var].items], dtype=float)
        if errors.size == 0:
            residuals[var] = {"bin_edges": [], "counts": [], "mean": 0.0, "std": 0.0}
            continue
        counts, edges = np.histogram(errors, bins=30)
        residuals[var] = {
            "bin_edges": edges.tolist(),
            "counts": counts.tolist(),
            "mean": float(errors.mean()),
            "std": float(errors.std()),
        }

    return {
        "overall": {k: computed[k] for k in ("mse", "mae", "rmse", "skill")},
        "per_horizon": per_horizon,
        "scatter": scatter,
        "residuals": residuals,
        "n_val_samples": len(bundle.val_loader.dataset),
        "n_train_samples": len(bundle.train_loader.dataset),
        "n_stations": bundle.dataset.n_stations,
    }


def _build_tuning_notes(
    training_curve: list[dict[str, float]],
    final_metrics: dict[str, float],
    per_horizon: dict[str, list[dict[str, float]]],
) -> list[dict[str, str]]:
    """Heuristic, human-readable suggestions for where to focus fine-tuning."""
    notes: list[dict[str, str]] = []

    if not training_curve:
        notes.append(
            {
                "severity": "info",
                "text": "No train_log.csv found in this run directory, so training-curve "
                "diagnostics (overfitting, early-stopping headroom) aren't available.",
            }
        )
    else:
        last = training_curve[-1]
        train_loss, val_loss = last["train_loss"], last["val_loss"]

        if train_loss > 0 and val_loss > train_loss * 1.5:
            notes.append(
                {
                    "severity": "warning",
                    "text": (
                        f"Validation loss ({val_loss:.3f}) is {val_loss / train_loss:.1f}x "
                        f"training loss ({train_loss:.3f}) - a sign of overfitting. Try "
                        "increasing training.weight_decay, adding dropout, lowering "
                        "model.hidden_dim, or gathering more training data."
                    ),
                }
            )

        if len(training_curve) >= 5:
            recent = training_curve[-5:]
            if recent[-1]["val_loss"] < recent[0]["val_loss"] * 0.98:
                notes.append(
                    {
                        "severity": "info",
                        "text": "Validation loss was still trending down at the last logged "
                        "epoch - training may benefit from a higher --epochs and/or "
                        "training.patience.",
                    }
                )

    skill = final_metrics.get("skill")
    if skill is not None and skill == skill:  # not NaN
        if skill < 0:
            notes.append(
                {
                    "severity": "critical",
                    "text": (
                        f"Skill score is negative ({skill:.2f}) - the model is worse than a "
                        "naive 'no change from now' forecast. Double-check normalization/target "
                        "alignment before tuning hyperparameters; a too-short data.back_hrs "
                        "window can also starve the model of useful signal."
                    ),
                }
            )
        elif skill < 0.1:
            notes.append(
                {
                    "severity": "warning",
                    "text": (
                        f"Skill score is only {skill:.2f} - barely better than persistence. "
                        "Consider fusing grid data (--use-hrrr/--use-era5), a larger "
                        "model.hidden_dim, or more training data."
                    ),
                }
            )
        else:
            notes.append(
                {
                    "severity": "good",
                    "text": f"Skill score is {skill:.2f} - the model beats a naive persistence "
                    "forecast.",
                }
            )

    for var, rows in per_horizon.items():
        if len(rows) < 2:
            continue
        first_rmse, last_rmse = rows[0]["rmse"], rows[-1]["rmse"]
        if first_rmse > 0 and last_rmse / first_rmse > 2.0:
            notes.append(
                {
                    "severity": "info",
                    "text": (
                        f"{var}: RMSE grows {last_rmse / first_rmse:.1f}x from the "
                        f"{rows[0]['lead_hr']}h to the {rows[-1]['lead_hr']}h horizon. "
                        "Long-horizon forecasts are the weak point here - grid data fusion "
                        "or a longer data.back_hrs window usually helps more at long lead "
                        "times than at short ones."
                    ),
                }
            )

    if not notes:
        notes.append(
            {
                "severity": "good",
                "text": "No obvious red flags: validation loss tracks training loss and the "
                "model beats the persistence baseline.",
            }
        )

    return notes


def build_report_data(
    run_dir: str | Path,
    data_dir: str | Path | None = "data",
    run_inference_eval: bool = True,
    device: str | None = None,
) -> ReportData:
    """Assemble a ReportData for the checkpoint in `run_dir`.

    Args:
        run_dir: A directory written by loaf.training.Trainer (contains
            best.pt/last.pt and, if training got that far, train_log.csv).
        data_dir: Base directory containing downloaded data (iem/, hrrr/,
            era5/) - needed to re-run the validation split for per-horizon,
            scatter, and residual detail. Pass None to skip inference
            entirely and build a metrics-only report from train_log.csv.
        run_inference_eval: Whether to re-run validation inference at all.
            Set to False to build a fast, data-independent report.
        device: Torch device string. Defaults to CUDA if available, else CPU.

    Returns:
        A ReportData ready for loaf.reporting.render.render_html().
    """
    run_dir = Path(run_dir)
    checkpoint_path = run_dir / "best.pt"
    if not checkpoint_path.exists():
        checkpoint_path = run_dir / "last.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No best.pt or last.pt checkpoint found in {run_dir}")

    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location=torch_device, weights_only=False)
    run_config: dict[str, Any] = checkpoint.get("run_config") or {}

    training_curve = _read_train_log(run_dir / "train_log.csv")
    best_epoch = None
    best_val_loss = None
    final_metrics: dict[str, float] = {}
    if training_curve:
        best_row = min(training_curve, key=lambda r: r["val_loss"])
        best_epoch = int(best_row["epoch"])
        best_val_loss = best_row["val_loss"]
        final_metrics = {
            "mse": best_row["val_loss"],
            "mae": best_row["val_mae"],
            "rmse": best_row["val_rmse"],
            "skill": best_row["val_skill"],
        }

    model_cfg = run_config.get("model_cfg") or {}
    hyperparams: dict[str, Any] = {
        "region": run_config.get("region_name", "unknown"),
        "model_type": checkpoint["model_type"],
        **{f"model.{k}": v for k, v in model_cfg.items() if k != "type"},
        "back_hrs": run_config.get("back_hrs"),
        "lead_times": checkpoint["lead_times"],
        "target_vars": checkpoint["target_vars"],
        "station_vars": checkpoint["station_vars"],
        "use_hrrr": run_config.get("use_hrrr", False),
        "use_era5": run_config.get("use_era5", False),
        "grid_vars": run_config.get("grid_vars"),
        "min_observations": run_config.get("min_observations"),
        "batch_size": run_config.get("batch_size"),
        "learning_rate": run_config.get("learning_rate"),
        "weight_decay": run_config.get("weight_decay"),
        "val_split": run_config.get("val_split"),
        "patience": run_config.get("patience"),
        "seed": run_config.get("seed"),
        "n_params": run_config.get("n_params"),
        "year": run_config.get("year"),
    }

    per_horizon: dict[str, list[dict[str, float]]] = {}
    scatter: dict[str, list[list[float]]] = {}
    residuals: dict[str, dict[str, Any]] = {}
    inference_error: str | None = None

    if run_inference_eval and data_dir is not None:
        try:
            result = run_inference(checkpoint, data_dir, torch_device)
        except Exception as exc:  # noqa: BLE001 - surfaced in the report, not fatal
            inference_error = str(exc)
        else:
            per_horizon = result["per_horizon"]
            scatter = result["scatter"]
            residuals = result["residuals"]
            final_metrics = {**final_metrics, **result["overall"]}
            hyperparams["n_val_samples"] = result["n_val_samples"]
            hyperparams["n_train_samples"] = result["n_train_samples"]
            hyperparams["n_stations"] = result["n_stations"]

    notes = _build_tuning_notes(training_curve, final_metrics, per_horizon)

    return ReportData(
        run_name=run_dir.name,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        checkpoint_path=str(checkpoint_path),
        model_type=checkpoint["model_type"],
        lead_times=checkpoint["lead_times"],
        target_vars=checkpoint["target_vars"],
        station_vars=checkpoint["station_vars"],
        hyperparams=hyperparams,
        training_curve=training_curve,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        final_metrics=final_metrics,
        per_horizon=per_horizon,
        scatter=scatter,
        residuals=residuals,
        notes=notes,
        inference_error=inference_error,
    )


# ---------------------------------------------------------------------------
# Cross-run "master report" - a fast, at-a-glance comparison across every run
# under a runs/ directory. See render.render_index_html().
# ---------------------------------------------------------------------------

# Trainer writes output dirs as "<region>_<YYYYMMDD>_<HHMMSS>" (see
# loaf.pipeline.train_stage) - parsed here to sort/label runs chronologically
# without relying on filesystem mtimes (which don't survive a copy/clone).
_RUN_NAME_RE = re.compile(r"^(?P<region>.+)_(?P<date>\d{8})_(?P<time>\d{6})$")


@dataclass
class RunSummary:
    """One row of the cross-run index/master report."""

    run_name: str
    run_dir: str
    sort_key: str
    display_date: str
    region: str
    model_type: str | None
    best_epoch: int | None
    epochs_trained: int
    lead_times: list[int]
    target_vars: list[str]
    hyperparams: dict[str, Any]
    final_metrics: dict[str, float]
    training_curve: list[dict[str, float]]
    has_report: bool
    error: str | None = None


@dataclass
class IndexData:
    """Everything render_index_html() needs to draw the cross-run summary."""

    generated_at: str
    runs_dir: str
    runs: list[RunSummary]


def _run_sort_key(run_dir: Path) -> str:
    """Chronological sort key for a run directory.

    Uses the timestamp embedded in Trainer's "<region>_<YYYYMMDD>_<HHMMSS>"
    naming convention when the name matches it, else falls back to the
    directory's mtime (still sortable, just less precise and won't survive
    a copy/clone the way the embedded timestamp does).
    """
    match = _RUN_NAME_RE.match(run_dir.name)
    if match:
        return match.group("date") + match.group("time")
    return f"{run_dir.stat().st_mtime:020.6f}"


def _run_display_date(run_dir: Path) -> str:
    """Short human-readable label for chart axes/table rows, e.g. "Aug 27, 21:32"."""
    match = _RUN_NAME_RE.match(run_dir.name)
    if not match:
        return run_dir.name
    try:
        dt = datetime.strptime(match.group("date") + match.group("time"), "%Y%m%d%H%M%S")
        return dt.strftime("%b %d, %H:%M")
    except ValueError:
        return run_dir.name


def discover_runs(runs_dir: str | Path) -> list[Path]:
    """Every run directory under `runs_dir` (any subdir with a best.pt/last.pt), oldest first."""
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return []
    candidates = [
        d
        for d in runs_dir.iterdir()
        if d.is_dir() and ((d / "best.pt").exists() or (d / "last.pt").exists())
    ]
    return sorted(candidates, key=_run_sort_key)


def build_run_summary(
    run_dir: str | Path,
    data_dir: str | Path | None = None,
    run_inference_eval: bool = False,
) -> RunSummary:
    """A lightweight per-run row for the cross-run index report.

    Defaults to skipping validation inference (run_inference_eval=False) -
    the index is meant to stay fast across many runs. final_metrics then
    come from the best epoch logged in train_log.csv, the same fallback a
    single-run report uses when inference isn't re-run. Pass
    run_inference_eval=True for metrics recomputed fresh against data_dir,
    at the cost of one validation pass per run.

    A run this can't read (corrupt checkpoint, missing data, etc.) doesn't
    fail the whole index - it comes back as a RunSummary with `error` set
    and every metric field empty, so one bad run doesn't hide the rest.
    """
    run_dir = Path(run_dir)
    try:
        data = build_report_data(run_dir, data_dir=data_dir, run_inference_eval=run_inference_eval)
    except Exception as exc:  # noqa: BLE001 - surfaced per-row, shouldn't sink the whole index
        return RunSummary(
            run_name=run_dir.name,
            run_dir=str(run_dir),
            sort_key=_run_sort_key(run_dir),
            display_date=_run_display_date(run_dir),
            region="unknown",
            model_type=None,
            best_epoch=None,
            epochs_trained=0,
            lead_times=[],
            target_vars=[],
            hyperparams={},
            final_metrics={},
            training_curve=[],
            has_report=(run_dir / "report.html").exists(),
            error=str(exc),
        )

    return RunSummary(
        run_name=data.run_name,
        run_dir=str(run_dir),
        sort_key=_run_sort_key(run_dir),
        display_date=_run_display_date(run_dir),
        region=str(data.hyperparams.get("region", "unknown")),
        model_type=data.model_type,
        best_epoch=data.best_epoch,
        epochs_trained=len(data.training_curve),
        lead_times=data.lead_times,
        target_vars=data.target_vars,
        hyperparams=data.hyperparams,
        final_metrics=data.final_metrics,
        training_curve=data.training_curve,
        has_report=(run_dir / "report.html").exists(),
    )


def build_index_data(
    runs_dir: str | Path,
    data_dir: str | Path | None = None,
    run_inference_eval: bool = False,
) -> IndexData:
    """Assemble an IndexData summarizing every run under `runs_dir`, oldest first."""
    runs_dir = Path(runs_dir)
    run_dirs = discover_runs(runs_dir)
    runs = [
        build_run_summary(d, data_dir=data_dir, run_inference_eval=run_inference_eval)
        for d in run_dirs
    ]
    return IndexData(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        runs_dir=str(runs_dir),
        runs=runs,
    )
