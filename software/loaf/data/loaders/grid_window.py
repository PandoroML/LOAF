"""Shared external (HRRR/ERA5) grid-window fetching for training and inference.

Both WeatherDataset (training, see .dataset) and Predictor (live inference,
see loaf.inference.predictor) need the external grid input to include, when
the underlying loader supports it, the *future* forecast horizon - not just
a second copy of the same historical window the station input already
carries. That's the actual fusion mechanism the reference implementation's
HRRR.getSample provides (LocalizedWeather Source/Dataloader/HRRR.py):
historical analysis (step=0) up to "now", concatenated with the forecast
issued at "now" for the steps ahead that make up the lead-time window - i.e.
the model gets to see what the regional forecast model thinks will happen,
not just what already happened. Centralizing the windowing logic here means
training and inference can't drift apart on what "the grid input window" is,
which matters because the model's input width is fixed at training time.

ERA5 has no forecast product in this pipeline (pure reanalysis, no `step`
dimension), so it keeps its original historical-only window: extending it to
include "future" reanalysis, the way the reference implementation does for
its offline paper experiments, would train on information that plain
doesn't exist yet at live inference time.
"""

from __future__ import annotations

import numpy as np
import torch

from .hrrr import HRRRLoader


def _n_forecast(grid_loader: object, lead_hours: int) -> int:
    """How many forecast steps beyond "now" this loader can supply, capped by lead_hours."""
    if not isinstance(grid_loader, HRRRLoader):
        return 0
    steps = grid_loader.steps
    if steps is None or len(steps) <= 1:
        return 0
    return min(lead_hours, len(steps) - 1)


def grid_window_len(grid_loader: object, back_hrs: int, lead_hours: int) -> int:
    """Number of time steps compute_grid_window() returns for this loader/config."""
    return back_hrs + _n_forecast(grid_loader, lead_hours)


def compute_grid_window(
    grid_loader: object,
    hist_start: np.datetime64 | str,
    now_time: np.datetime64 | str,
    back_hrs: int,
    lead_hours: int,
    variables: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Fetch the external grid input window for one sample.

    For an HRRRLoader with forecast steps archived, this is `back_hrs` hours
    of historical analysis ending at `now_time`, concatenated with the
    forecast issued at `now_time` for up to `lead_hours` ahead (see module
    docstring). Otherwise (ERA5, or HRRR data with no forecast steps
    downloaded), it's just the `back_hrs`-hour historical window, matching
    the station input.

    Args:
        grid_loader: HRRRLoader or ERA5Loader instance.
        hist_start: Start of the historical window (inclusive).
        now_time: End of the historical window / forecast initialization time.
        back_hrs: Length of the historical window, in hours.
        lead_hours: Forecast horizon needed, in hours.
        variables: Grid variables to extract.

    Returns:
        Dict of variable -> (n_nodes, grid_window_len(...)) tensor.
    """
    n_forecast = _n_forecast(grid_loader, lead_hours)
    if n_forecast > 0:
        return grid_loader.get_sample_with_forecast(
            hist_start,
            now_time,
            n_historical=back_hrs,
            n_forecast=n_forecast,
            variables=variables,
        )
    return grid_loader.get_sample(hist_start, now_time, variables)
