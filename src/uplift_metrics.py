"""Uplift-modeling evaluation helpers (Qini, uplift@k, policy simulation).

These are thin, dependency-light utilities used by the modeling notebook so the
core logic is reusable and unit-testable outside a notebook.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# numpy>=2.0 renamed trapz -> trapezoid; keep both working.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


def uplift_by_percentile(y_true, treatment, uplift, bins: int = 10) -> pd.DataFrame:
    """Observed uplift within each score decile (top scores first).

    For each bin we report the response rate of the treated vs control units and
    their difference (the realized uplift). A monotonically decreasing 'uplift'
    column across bins indicates a well-ranked model.
    """
    df = pd.DataFrame(
        {"y": np.asarray(y_true), "w": np.asarray(treatment), "s": np.asarray(uplift)}
    )
    df["bin"] = pd.qcut(df["s"].rank(method="first", ascending=False), bins, labels=False)
    rows = []
    for b, g in df.groupby("bin"):
        t, c = g[g.w == 1], g[g.w == 0]
        rt = t.y.mean() if len(t) else np.nan
        rc = c.y.mean() if len(c) else np.nan
        rows.append(
            {
                "decile": int(b) + 1,
                "n": len(g),
                "n_treated": len(t),
                "n_control": len(c),
                "resp_treated": rt,
                "resp_control": rc,
                "uplift": rt - rc,
            }
        )
    return pd.DataFrame(rows)


def qini_curve(y_true, treatment, uplift):
    """Return (x, y) points of the Qini curve (cumulative incremental gains).

    x is the fraction of the population targeted (sorted by descending score);
    y is the cumulative number of *incremental* positive outcomes captured.
    """
    df = pd.DataFrame(
        {"y": np.asarray(y_true), "w": np.asarray(treatment), "s": np.asarray(uplift)}
    ).sort_values("s", ascending=False, kind="mergesort").reset_index(drop=True)

    cum_y_t = (df.y * df.w).cumsum()
    cum_y_c = (df.y * (1 - df.w)).cumsum()
    cum_n_t = df.w.cumsum()
    cum_n_c = (1 - df.w).cumsum()
    ratio = np.where(cum_n_c == 0, 0, cum_n_t / np.where(cum_n_c == 0, 1, cum_n_c))
    qini = cum_y_t - cum_y_c * ratio

    n = len(df)
    x = np.arange(1, n + 1) / n
    return np.concatenate([[0], x]), np.concatenate([[0], qini.values])


def qini_auc(y_true, treatment, uplift) -> float:
    """Area between the model Qini curve and the random (diagonal) baseline."""
    x, y = qini_curve(y_true, treatment, uplift)
    area_model = _trapz(y, x)
    area_rand = _trapz(np.linspace(0, y[-1], len(y)), x)
    return float(area_model - area_rand)


def uplift_at_k(y_true, treatment, uplift, k: float = 0.3) -> float:
    """Realized uplift among the top-k fraction ranked by predicted uplift."""
    df = pd.DataFrame(
        {"y": np.asarray(y_true), "w": np.asarray(treatment), "s": np.asarray(uplift)}
    ).sort_values("s", ascending=False, kind="mergesort")
    top = df.head(max(1, int(len(df) * k)))
    t, c = top[top.w == 1], top[top.w == 0]
    rt = t.y.mean() if len(t) else 0.0
    rc = c.y.mean() if len(c) else 0.0
    return float(rt - rc)
