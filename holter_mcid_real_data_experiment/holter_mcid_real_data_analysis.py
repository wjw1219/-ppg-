"""Real-data, patient-level discovery of a study-specific AF-burden MCID.

The script implements the prespecified data-driven analysis plan:

1. Quality-control paired Holter AF-burden measurements.
2. Calculate the change as second Holter minus first Holter, in percentage
   points (pp), not relative percentage change.
3. Use an independent clinical anchor supplied by the user. The anchor can be
   a binary column or a continuous score, in which case the highest third of
   the observed score distribution is labelled anchor-positive.
4. Search candidate thresholds from 0 to 20 pp in 0.5-pp increments using the
   Youden index, sensitivity, specificity and anchor-positive rates.
5. Assess the selected threshold using 2,000 patient-level bootstrap samples.
6. Estimate MDC95 from a supplied stable subgroup or a supplied repeat-
   measurement difference column. MDC95 is skipped when the required stable
   repeatability information is absent; it is never fabricated.
7. Export source tables, logs and three vector figures (SVG and PDF), together
   with PNG/TIFF previews.

This is a real-data analysis pipeline. It does not simulate clinical anchors,
events, outcomes or model predictions.

Typical use:

    python reports/holter_mcid_real_data_analysis.py \
        --input data/paired_holter.csv \
        --clinical data/clinical_anchor.csv \
        --cohort data/cohort.csv \
        --output-dir results/holter_mcid_real \
        --anchor-positive-column anchor_positive \
        --stable-column stable_subgroup \
        --bootstrap 2000 \
        --seed 20260818

See reports/README_holter_mcid_real_data.md for the input schema.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "_mplconfig_real_mcid"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.linear_model import LogisticRegression


THRESHOLD_GRID = np.arange(0.0, 20.01, 0.5)
DEFAULT_BOOTSTRAP = 2000

PATIENT_ID_ALIASES = ("patient_id", "patientid", "subject_id", "id")
HOLTER_1_ALIASES = (
    "holter1_af_burden_pct",
    "holter_1_af_burden_pct",
    "discharge_af_burden_pct",
    "baseline_af_burden_pct",
    "first_af_burden_pct",
    "af_burden_holter1_pct",
)
HOLTER_2_ALIASES = (
    "holter2_af_burden_pct",
    "holter_2_af_burden_pct",
    "month6_af_burden_pct",
    "followup_af_burden_pct",
    "second_af_burden_pct",
    "af_burden_holter2_pct",
)
ANCHOR_BINARY_ALIASES = (
    "anchor_positive",
    "clinical_anchor_positive",
    "anchor_label",
    "clinical_deterioration",
    "cv_event_followup",
    "clinical_event",
    "composite_event",
)
ANCHOR_SCORE_ALIASES = (
    "anchor_score",
    "clinical_anchor_score",
    "global_change_score",
    "clinical_change_score",
)
STABLE_ALIASES = (
    "stable_subgroup",
    "stable_patient",
    "clinically_stable",
    "stable_repeatability",
)
COMPLETE_ALIASES = (
    "completed_6m_followup",
    "complete_6m_followup",
    "six_month_followup_complete",
    "model_eligible",
)

COLORS = {
    "positive": "#B64342",
    "negative": "#767676",
    "threshold": "#7A5A00",
    "dark": "#272727",
    "blue": "#1764A5",
    "teal": "#42949E",
    "light": "#E8EEF5",
    "improvement": "#3775BA",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 7.5,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Merged patient-level CSV.")
    source.add_argument("--holter", help="CSV containing paired Holter values.")
    parser.add_argument("--clinical", help="Optional clinical anchor CSV, merged by patient_id.")
    parser.add_argument("--cohort", help="Optional enrollment CSV used to report the enrolled denominator.")
    parser.add_argument("--output-dir", required=True, help="Directory for tables, figures and logs.")
    parser.add_argument("--patient-id-column", help="Override patient ID column name.")
    parser.add_argument("--holter-1-column", help="Override first Holter AF-burden column name.")
    parser.add_argument("--holter-2-column", help="Override second Holter AF-burden column name.")
    parser.add_argument("--anchor-positive-column", help="Binary clinical anchor column, coded 0/1 or yes/no.")
    parser.add_argument("--anchor-score-column", help="Continuous clinical anchor score; highest third is positive.")
    parser.add_argument("--stable-column", help="Stable-subgroup indicator column.")
    parser.add_argument("--stable-repeat-diff-column", help="Optional signed repeat-measurement difference in pp.")
    parser.add_argument("--complete-column", help="Optional 6-month completion/model-eligibility indicator.")
    parser.add_argument("--min-valid-hours", type=float, default=0.0)
    parser.add_argument("--max-artifact-fraction", type=float, default=1.0)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    available = set(columns)
    return next((candidate for candidate in candidates if candidate in available), None)


def resolve_column(
    columns: Iterable[str], explicit: str | None, aliases: Iterable[str], description: str, required: bool = False
) -> str | None:
    if explicit:
        if explicit not in columns:
            raise ValueError(f"{description} column '{explicit}' was not found. Available columns: {list(columns)}")
        return explicit
    found = first_existing(columns, aliases)
    if required and found is None:
        raise ValueError(f"Could not identify {description} column. Available columns: {list(columns)}")
    return found


def read_csv_checked(path: str | Path, label: str) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"{label} file does not exist: {source}")
    frame = pd.read_csv(source)
    if frame.empty:
        raise ValueError(f"{label} file is empty: {source}")
    return frame


def normalize_patient_id(frame: pd.DataFrame, explicit: str | None, label: str) -> tuple[pd.DataFrame, str]:
    column = resolve_column(frame.columns, explicit, PATIENT_ID_ALIASES, f"{label} patient ID", required=True)
    frame = frame.copy()
    if frame[column].isna().any():
        raise ValueError(f"{label} contains missing patient IDs.")
    frame["patient_id"] = frame[column].astype(str).str.strip()
    if frame["patient_id"].eq("").any():
        raise ValueError(f"{label} contains missing patient IDs.")
    if frame["patient_id"].duplicated().any():
        duplicate_ids = frame.loc[frame["patient_id"].duplicated(), "patient_id"].head(5).tolist()
        raise ValueError(f"{label} must contain one row per patient; duplicated IDs include {duplicate_ids}.")
    return frame, column


def parse_binary(series: pd.Series, column_name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype("Int64")
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == series.notna().sum() and numeric.dropna().isin([0, 1]).all():
        return numeric.astype("Int64")
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {
        "yes": 1,
        "y": 1,
        "true": 1,
        "positive": 1,
        "event": 1,
        "worsening": 1,
        "no": 0,
        "n": 0,
        "false": 0,
        "negative": 0,
        "none": 0,
        "stable": 0,
    }
    converted = normalized.map(mapping).astype("Int64")
    invalid = series.notna() & converted.isna()
    if invalid.any():
        values = series.loc[invalid].astype(str).unique().tolist()[:10]
        raise ValueError(f"Anchor column '{column_name}' is not binary 0/1 or recognized yes/no values: {values}")
    return converted


def load_data(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, object]]:
    if args.input:
        data = read_csv_checked(args.input, "input")
        data, patient_source_column = normalize_patient_id(data, args.patient_id_column, "input")
        input_sources = {"input": str(Path(args.input).resolve())}
    else:
        data = read_csv_checked(args.holter, "Holter")
        data, patient_source_column = normalize_patient_id(data, args.patient_id_column, "Holter")
        input_sources = {"holter": str(Path(args.holter).resolve())}

    if args.clinical:
        clinical = read_csv_checked(args.clinical, "clinical anchor")
        clinical, _ = normalize_patient_id(clinical, args.patient_id_column, "clinical anchor")
        keep = [column for column in clinical.columns if column != "patient_id"]
        data = data.merge(clinical[["patient_id", *keep]], on="patient_id", how="left", validate="one_to_one")
        input_sources["clinical"] = str(Path(args.clinical).resolve())

    holter_1 = resolve_column(data.columns, args.holter_1_column, HOLTER_1_ALIASES, "first Holter AF-burden", required=True)
    holter_2 = resolve_column(data.columns, args.holter_2_column, HOLTER_2_ALIASES, "second Holter AF-burden", required=True)
    data = data.copy()
    data["holter_1_af_burden_pct"] = pd.to_numeric(data[holter_1], errors="coerce")
    data["holter_2_af_burden_pct"] = pd.to_numeric(data[holter_2], errors="coerce")
    data["delta_burden_pp"] = data["holter_2_af_burden_pct"] - data["holter_1_af_burden_pct"]

    cohort_n = int(len(data))
    if args.cohort:
        cohort = read_csv_checked(args.cohort, "cohort")
        cohort, _ = normalize_patient_id(cohort, args.patient_id_column, "cohort")
        cohort_n = int(cohort["patient_id"].nunique())
        input_sources["cohort"] = str(Path(args.cohort).resolve())

    complete_column = resolve_column(data.columns, args.complete_column, COMPLETE_ALIASES, "follow-up completion", required=False)
    if complete_column:
        complete = parse_binary(data[complete_column], complete_column)
        data["followup_complete"] = complete
    else:
        data["followup_complete"] = 1

    anchor_binary = resolve_column(
        data.columns, args.anchor_positive_column, ANCHOR_BINARY_ALIASES, "binary clinical anchor", required=False
    )
    anchor_score = resolve_column(
        data.columns, args.anchor_score_column, ANCHOR_SCORE_ALIASES, "clinical anchor score", required=False
    )
    if args.anchor_positive_column and anchor_binary is None:
        raise ValueError("The requested binary clinical anchor column could not be resolved.")
    if args.anchor_score_column and anchor_score is None:
        raise ValueError("The requested clinical anchor score column could not be resolved.")
    if anchor_binary and anchor_score:
        logging.warning("Both binary anchor and anchor score were found; binary anchor will be used.")

    if anchor_binary:
        data["anchor_positive"] = parse_binary(data[anchor_binary], anchor_binary)
        anchor_source = f"binary column: {anchor_binary}"
    elif anchor_score:
        score = pd.to_numeric(data[anchor_score], errors="coerce")
        data["anchor_score"] = score
        ranks = score.rank(method="first", ascending=False, na_option="keep")
        eligible_n = int(score.notna().sum())
        top_n = int(math.ceil(eligible_n / 3)) if eligible_n else 0
        data["anchor_positive"] = pd.Series(pd.NA, index=data.index, dtype="Int64")
        if top_n:
            data.loc[score.notna(), "anchor_positive"] = (ranks[score.notna()] <= top_n).astype(int).to_numpy()
        anchor_source = f"highest third of continuous score: {anchor_score}"
    else:
        data["anchor_positive"] = pd.Series(pd.NA, index=data.index, dtype="Int64")
        anchor_source = "not supplied"
        logging.warning("No clinical anchor column or score was supplied; data-driven MCID discovery will be skipped.")

    stable_column = resolve_column(data.columns, args.stable_column, STABLE_ALIASES, "stable-subgroup", required=False)
    if stable_column:
        data["stable_subgroup"] = parse_binary(data[stable_column], stable_column)
    else:
        data["stable_subgroup"] = pd.Series(pd.NA, index=data.index, dtype="Int64")

    metadata = {
        "input_sources": input_sources,
        "patient_id_source_column": patient_source_column,
        "first_holter_source_column": holter_1,
        "second_holter_source_column": holter_2,
        "anchor_source": anchor_source,
        "stable_source_column": stable_column,
        "completion_source_column": complete_column,
        "n_enrolled_from_cohort_or_input": cohort_n,
    }
    return data, metadata


def apply_quality_control(data: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = data.copy()
    status = pd.Series("included", index=work.index, dtype="object")
    for column in ("holter_1_af_burden_pct", "holter_2_af_burden_pct"):
        status.loc[work[column].isna()] = f"missing_{column}"
        status.loc[work[column].notna() & ~work[column].between(0, 100)] = f"invalid_{column}"
    status.loc[work["delta_burden_pp"].isna()] = "missing_delta"
    status.loc[work["followup_complete"].isna() | (work["followup_complete"] != 1)] = "incomplete_followup"

    for column in ("discharge_holter_valid_hours", "month6_holter_valid_hours", "holter1_valid_hours", "holter2_valid_hours"):
        if column in work.columns:
            values = pd.to_numeric(work[column], errors="coerce")
            if args.min_valid_hours > 0:
                status.loc[values.notna() & (values < args.min_valid_hours)] = f"below_minimum_{column}"

    for column in (
        "holter_artifact_fraction_discharge",
        "holter_artifact_fraction_month6",
        "holter1_artifact_fraction",
        "holter2_artifact_fraction",
    ):
        if column in work.columns:
            values = pd.to_numeric(work[column], errors="coerce")
            status.loc[values.notna() & (values > args.max_artifact_fraction)] = f"high_artifact_{column}"

    work["qc_status"] = status
    qc_log = work[["patient_id", "qc_status"]].copy()
    included = work.loc[work["qc_status"] == "included"].copy()
    if included["patient_id"].duplicated().any():
        raise ValueError("Patient-level analysis requires one included row per patient.")
    return included, qc_log


def add_change_groups(data: pd.DataFrame, cutoff: float = 10.0) -> pd.DataFrame:
    work = data.copy()
    delta = work["delta_burden_pp"]
    work["change_group_at_10pp"] = np.select(
        [delta <= -cutoff, delta < cutoff],
        ["clinically_relevant_improvement", "minor_change_or_fluctuation"],
        default="clinically_relevant_worsening",
    )
    return work


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else math.nan


def threshold_metrics(delta: np.ndarray, anchor: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = delta >= threshold
    positive = anchor == 1
    negative = anchor == 0
    tp = int((predicted & positive).sum())
    fn = int((~predicted & positive).sum())
    fp = int((predicted & negative).sum())
    tn = int((~predicted & negative).sum())
    sensitivity = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    above_n = int(predicted.sum())
    below_n = int((~predicted).sum())
    above_anchor_rate = safe_divide(int(anchor[predicted].sum()), above_n)
    below_anchor_rate = safe_divide(int(anchor[~predicted].sum()), below_n)
    youden = sensitivity + specificity - 1 if np.isfinite(sensitivity) and np.isfinite(specificity) else math.nan
    odds_ratio = safe_divide((tp + 0.5) * (tn + 0.5), (fp + 0.5) * (fn + 0.5))
    return {
        "threshold_pp": float(threshold),
        "above_threshold_n": above_n,
        "below_threshold_n": below_n,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "youden_index": youden,
        "anchor_positive_rate_above": above_anchor_rate,
        "anchor_positive_rate_below": below_anchor_rate,
        "risk_difference": above_anchor_rate - below_anchor_rate
        if np.isfinite(above_anchor_rate) and np.isfinite(below_anchor_rate)
        else math.nan,
        "continuity_corrected_odds_ratio": odds_ratio,
    }


def threshold_search(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    eligible = data["anchor_positive"].notna()
    if eligible.sum() < 10 or data.loc[eligible, "anchor_positive"].nunique() < 2:
        return pd.DataFrame(), None
    delta = data.loc[eligible, "delta_burden_pp"].to_numpy(float)
    anchor = data.loc[eligible, "anchor_positive"].to_numpy(int)
    grid = pd.DataFrame([threshold_metrics(delta, anchor, t) for t in THRESHOLD_GRID])
    valid = grid[grid["youden_index"].notna()]
    if valid.empty:
        return grid, None
    # Prespecified tie-break: choose the lowest threshold among exact Youden ties.
    best = valid.sort_values(["youden_index", "threshold_pp"], ascending=[False, True]).iloc[0]
    return grid, best


def bootstrap_thresholds(data: pd.DataFrame, n_bootstrap: int, seed: int) -> pd.DataFrame:
    eligible = data["anchor_positive"].notna()
    work = data.loc[eligible, ["delta_burden_pp", "anchor_positive"]].copy()
    if len(work) < 10 or work["anchor_positive"].nunique() < 2:
        return pd.DataFrame(columns=["bootstrap_replicate", "selected_threshold_pp", "selected_youden_index"])
    delta = work["delta_burden_pp"].to_numpy(float)
    anchor = work["anchor_positive"].to_numpy(int)
    rng = np.random.default_rng(seed)
    rows = []
    for replicate in range(1, n_bootstrap + 1):
        indices = rng.integers(0, len(work), len(work))
        if np.unique(anchor[indices]).size < 2:
            continue
        boot_grid = pd.DataFrame([threshold_metrics(delta[indices], anchor[indices], t) for t in THRESHOLD_GRID])
        valid = boot_grid[boot_grid["youden_index"].notna()]
        if valid.empty:
            continue
        best = valid.sort_values(["youden_index", "threshold_pp"], ascending=[False, True]).iloc[0]
        rows.append(
            {
                "bootstrap_replicate": replicate,
                "selected_threshold_pp": float(best["threshold_pp"]),
                "selected_youden_index": float(best["youden_index"]),
            }
        )
    return pd.DataFrame(rows)


def estimate_mdc95(data: pd.DataFrame, stable_repeat_diff_column: str | None) -> tuple[pd.DataFrame, pd.Series | None]:
    if stable_repeat_diff_column:
        if stable_repeat_diff_column not in data.columns:
            raise ValueError(f"Stable repeat-difference column not found: {stable_repeat_diff_column}")
        differences = pd.to_numeric(data[stable_repeat_diff_column], errors="coerce").dropna().to_numpy(float)
        source = f"repeat-difference column: {stable_repeat_diff_column}"
    else:
        stable = data.loc[data["stable_subgroup"] == 1, "delta_burden_pp"].dropna().to_numpy(float)
        differences = stable
        source = "delta_burden_pp among stable_subgroup == 1"

    if len(differences) < 2:
        summary = pd.DataFrame(
            [{"status": "not_estimable", "reason": "fewer than two stable repeat differences", "n": len(differences)}]
        )
        return summary, None
    sd_difference = float(np.std(differences, ddof=1))
    mdc95 = float(1.96 * sd_difference)
    row = {
        "status": "estimable",
        "source": source,
        "n_stable_repeat_differences": len(differences),
        "mean_repeat_difference_pp": float(np.mean(differences)),
        "sd_repeat_difference_pp": sd_difference,
        "mdc95_pp": mdc95,
    }
    return pd.DataFrame([row]), pd.Series(row)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt((p * (1 - p) / total) + (z * z / (4 * total * total))) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def anchor_group_summary(data: pd.DataFrame, cutoff: float) -> pd.DataFrame:
    eligible = data[data["anchor_positive"].notna()].copy()
    if eligible.empty:
        return pd.DataFrame()
    eligible["threshold_group"] = np.where(
        eligible["delta_burden_pp"] >= cutoff, f"delta >= {cutoff:g} pp", f"delta < {cutoff:g} pp"
    )
    rows = []
    for group, subset in eligible.groupby("threshold_group", sort=False):
        n = len(subset)
        events = int(subset["anchor_positive"].sum())
        low, high = wilson_interval(events, n)
        rows.append(
            {
                "threshold_pp": cutoff,
                "threshold_group": group,
                "n": n,
                "anchor_positive_n": events,
                "anchor_positive_rate_pct": events / n * 100,
                "wilson_low_pct": low * 100,
                "wilson_high_pct": high * 100,
            }
        )
    return pd.DataFrame(rows)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.15, 1.04, label, transform=ax.transAxes, fontsize=9, fontweight="bold", color=COLORS["dark"])


def clean_ax(ax: plt.Axes) -> None:
    ax.tick_params(width=0.7, length=3, color="#555555")
    ax.grid(axis="y", color="#D9D9D9", lw=0.5, alpha=0.55)
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    base = output_dir / stem
    fig.savefig(base.with_suffix(".png"), dpi=600)
    fig.savefig(base.with_suffix(".tiff"), dpi=600)
    fig.savefig(base.with_suffix(".svg"))
    fig.savefig(base.with_suffix(".pdf"))
    plt.close(fig)


def make_figure_1(data: pd.DataFrame, metadata: dict[str, object], selected_cutoff: float | None, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.6), gridspec_kw={"width_ratios": [0.95, 1.15, 1.0]})
    ax = axes[0]
    ax.axis("off")
    n_enrolled = int(metadata["n_enrolled_from_cohort_or_input"])
    n_complete = len(data)
    anchor_positive_n = int(data["anchor_positive"].eq(1).sum())
    anchor_negative_n = int(data["anchor_positive"].eq(0).sum())
    boxes = [
        (0.06, 0.63, 0.88, 0.20, f"Enrolled\nn = {n_enrolled}", "#E8EEF5"),
        (0.06, 0.34, 0.88, 0.20, f"Paired Holter and eligible analysis\nn = {n_complete}", "#F2F4F7"),
        (0.06, 0.03, 0.40, 0.20, f"Anchor-\npositive\nn = {anchor_positive_n}", "#F7DFDF"),
        (0.54, 0.03, 0.40, 0.20, f"Anchor-\nnegative\nn = {anchor_negative_n}", "#E6EEF8"),
    ]
    for x, y, w, h, text, color in boxes:
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02", transform=ax.transAxes,
            facecolor=color, edgecolor="#66717D", linewidth=0.8,
        )
        ax.add_patch(patch)
        fontsize = 6.5 if y < 0.1 else 7.2
        ax.text(x + w / 2, y + h / 2, text, transform=ax.transAxes, ha="center", va="center", fontsize=fontsize)
    for start, end in [((0.5, 0.63), (0.5, 0.54)), ((0.5, 0.34), (0.28, 0.23)), ((0.5, 0.34), (0.72, 0.23))]:
        ax.add_patch(FancyArrowPatch(start, end, transform=ax.transAxes, arrowstyle="-|>", mutation_scale=8, linewidth=0.8, color="#555555"))
    panel_label(ax, "a")

    ax = axes[1]
    if data["anchor_positive"].notna().any():
        for label, color, legend in [(1, COLORS["positive"], "Anchor-positive"), (0, COLORS["negative"], "Anchor-negative")]:
            subset = data[data["anchor_positive"] == label]
            ax.scatter(subset["holter_1_af_burden_pct"], subset["holter_2_af_burden_pct"], s=11, alpha=0.65, color=color, edgecolors="white", linewidths=0.25, label=legend)
    else:
        ax.scatter(data["holter_1_af_burden_pct"], data["holter_2_af_burden_pct"], s=11, alpha=0.65, color=COLORS["negative"], edgecolors="white", linewidths=0.25)
    ax.plot([0, 100], [0, 100], "--", lw=0.9, color="#555555")
    ax.set(xlim=(0, 100), ylim=(0, 100), xlabel="First Holter AF burden (%)", ylabel="Second Holter AF burden (%)")
    ax.text(0.04, 0.93, "Each point = 1 patient", transform=ax.transAxes, fontsize=6.7, color="#555555")
    clean_ax(ax)
    panel_label(ax, "b")
    if data["anchor_positive"].notna().any():
        ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.03), ncol=2, handlelength=1.0, columnspacing=0.8)

    ax = axes[2]
    ax.hist(data["delta_burden_pp"], bins=np.linspace(-40, 40, 33), color="#A7B1BC", edgecolor="white", linewidth=0.35)
    if selected_cutoff is not None:
        ax.axvline(selected_cutoff, color=COLORS["threshold"], lw=1.1, ls="--")
        ax.text(selected_cutoff + 0.7, 0.98, f"selected cutoff\n{selected_cutoff:g} pp", transform=ax.get_xaxis_transform(), ha="left", va="top", fontsize=6.7, color=COLORS["threshold"])
    ax.axvline(10, color="#555555", lw=0.8, ls=":")
    ax.text(10.4, 0.80, "10 pp", transform=ax.get_xaxis_transform(), ha="left", va="top", fontsize=6.5, color="#555555")
    ax.set(xlabel="AF burden change (pp)", ylabel="Patients")
    clean_ax(ax)
    panel_label(ax, "c")
    fig.tight_layout(rect=(0, 0.02, 1, 0.92), w_pad=1.8)
    save_figure(fig, output_dir, "figure_1_cohort_and_holter_change")


def make_figure_2(grid: pd.DataFrame, selected: pd.Series, output_dir: Path) -> None:
    if grid.empty or selected is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.55), sharex=True)
    x = grid["threshold_pp"].to_numpy(float)
    cutoff = float(selected["threshold_pp"])
    for ax in axes:
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.set_xlabel("Candidate threshold (pp)")
        ax.axvline(cutoff, color=COLORS["threshold"], ls="--", lw=1.0)
        clean_ax(ax)
    axes[0].plot(x, grid["youden_index"], "o-", color=COLORS["dark"], lw=1.6, ms=3.4)
    axes[0].scatter([cutoff], [float(selected["youden_index"])], s=32, color=COLORS["threshold"], zorder=3)
    axes[0].set(ylabel="Youden index", ylim=(0, 1.0))
    axes[0].text(cutoff + 0.3, 0.94, f"selected = {cutoff:g} pp", color=COLORS["threshold"], fontsize=6.7, va="top")
    panel_label(axes[0], "a")

    axes[1].plot(x, grid["sensitivity"] * 100, "o-", color=COLORS["blue"], lw=1.5, ms=3.4, label="Sensitivity")
    axes[1].plot(x, grid["specificity"] * 100, "s-", color=COLORS["teal"], lw=1.5, ms=3.2, label="Specificity")
    axes[1].set(ylabel="Anchor discrimination (%)", ylim=(0, 105))
    axes[1].legend(loc="lower left", handlelength=1.2)
    panel_label(axes[1], "b")

    axes[2].plot(x, grid["anchor_positive_rate_above"] * 100, "o-", color=COLORS["positive"], lw=1.5, ms=3.4, label="Above threshold")
    axes[2].plot(x, grid["anchor_positive_rate_below"] * 100, "s-", color=COLORS["negative"], lw=1.5, ms=3.2, label="Below threshold")
    axes[2].set(ylabel="Anchor-positive rate (%)", ylim=(0, 105))
    axes[2].legend(loc="upper right", handlelength=1.2)
    panel_label(axes[2], "c")
    fig.tight_layout(rect=(0, 0.0, 1, 0.96), w_pad=2.0)
    save_figure(fig, output_dir, "figure_2_threshold_discovery")


def make_figure_3(
    data: pd.DataFrame,
    selected: pd.Series,
    bootstrap: pd.DataFrame,
    mdc: pd.Series | None,
    output_dir: Path,
) -> None:
    if selected is None or bootstrap.empty:
        return
    cutoff = float(selected["threshold_pp"])
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), gridspec_kw={"width_ratios": [1.0, 1.0, 1.15]})

    ax = axes[0]
    boot_values = bootstrap["selected_threshold_pp"].to_numpy(float)
    ax.hist(boot_values, bins=np.arange(-0.25, 20.76, 0.5), color="#9AA8B5", edgecolor="white", linewidth=0.25)
    median = float(np.median(boot_values))
    q025, q975 = np.percentile(boot_values, [2.5, 97.5])
    ax.axvline(cutoff, color=COLORS["threshold"], ls="--", lw=1.1)
    ax.axvline(10, color="#555555", ls=":", lw=0.8)
    ax.text(cutoff + 0.3, 0.95, f"median {median:g} pp", transform=ax.get_xaxis_transform(), color=COLORS["threshold"], fontsize=6.6, va="top")
    ax.set(xlabel="Bootstrap-estimated cutoff (pp)", ylabel="Bootstrap replicates")
    clean_ax(ax)
    panel_label(ax, "a")

    ax = axes[1]
    if mdc is not None and np.isfinite(float(mdc["mdc95_pp"])):
        values = [float(mdc["mdc95_pp"]), cutoff]
        bars = ax.bar([0, 1], values, color=["#A7B1BC", COLORS["threshold"]], width=0.58, edgecolor="white")
        ax.axhline(10, color="#555555", ls=":", lw=0.8)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.35, f"{value:.1f}", ha="center", va="bottom", fontsize=7)
        ax.set_ylim(0, max(12, max(values) * 1.25))
    else:
        ax.text(0.5, 0.52, "MDC95 not estimable\n(no stable repeat data)", transform=ax.transAxes, ha="center", va="center", fontsize=7)
        ax.set_ylim(0, 1)
    ax.set_xticks([0, 1], ["MDC95", "Estimated\ncutoff"])
    ax.set_ylabel("Percentage points")
    clean_ax(ax)
    panel_label(ax, "b")

    ax = axes[2]
    eligible = data[data["anchor_positive"].notna()].copy()
    x = eligible["delta_burden_pp"].to_numpy(float)[:, None]
    y = eligible["anchor_positive"].to_numpy(int)
    if len(np.unique(y)) == 2:
        model = LogisticRegression(C=10.0, solver="lbfgs", max_iter=2000).fit(x, y)
        grid = np.linspace(float(eligible["delta_burden_pp"].min()) - 2, float(eligible["delta_burden_pp"].max()) + 2, 241)
        risk = model.predict_proba(grid[:, None])[:, 1]
        ax.plot(grid, risk, color=COLORS["positive"], lw=1.8)
    ax.axvline(cutoff, color=COLORS["threshold"], ls="--", lw=1.0)
    ax.axvspan(-10, 10, color="#D9D9D9", alpha=0.30, lw=0)
    ax.text(cutoff + 0.8, 0.94, f"{cutoff:g} pp", color=COLORS["threshold"], fontsize=6.7, va="top")
    ax.text(0, 0.86, "minor-change\nreference region", color="#555555", fontsize=6.5, ha="center", va="top")
    ax.set(xlabel="AF burden change (pp)", ylabel="Anchor-positive probability", ylim=(0, 1.0))
    clean_ax(ax)
    panel_label(ax, "c")
    fig.tight_layout(rect=(0, 0.0, 1, 0.96), w_pad=2.3)
    save_figure(fig, output_dir, "figure_3_clinical_anchor")


def write_captions(output_dir: Path) -> None:
    captions = r"""\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{figure_1_cohort_and_holter_change.pdf}
    \caption{\textbf{Cohort flow and continuous paired-Holter change distribution.} (a) Enrolled cohort and the patient-level analysis cohort after follow-up and quality-control criteria. (b) Paired first and second Holter AF-burden values, coloured by the independent clinical anchor when available. (c) Continuous AF-burden change distribution; the selected data-driven cutoff and the 10-percentage-point reference are shown.}
    \label{fig:real-cohort-holter-change}
\end{figure*}

\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{figure_2_threshold_discovery.pdf}
    \caption{\textbf{Data-driven threshold discovery.} (a) Youden index across candidate thresholds from 0 to 20 percentage points. (b) Sensitivity and specificity for the independent clinical anchor. (c) Anchor-positive rates above and below each candidate threshold. The selected cutoff is determined by the prespecified maximum-Youden rule; no uncertainty lines are displayed.}
    \label{fig:real-threshold-discovery}
\end{figure*}

\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{figure_3_clinical_anchor.pdf}
    \caption{\textbf{Threshold stability, measurement-error context and clinical anchoring.} (a) Patient-level bootstrap distribution of the selected cutoff. (b) MDC95 estimated from stable repeat measurements compared with the selected cutoff; the panel reports non-estimability when stable repeat data are unavailable. (c) Smooth anchor-positive probability across continuous AF-burden change, with the $-10$ to $+10$ percentage-point region shown for interpretation.}
    \label{fig:real-clinical-anchor}
\end{figure*}
"""
    (output_dir / "figure_captions_latex.tex").write_text(captions, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap resamples; the planned analysis uses 2000.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=output_dir / "analysis_log.txt",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.info("Starting real-data paired-Holter MCID analysis")
    data, metadata = load_data(args)
    included, qc_log = apply_quality_control(data, args)
    if included.empty:
        raise ValueError("No patients remain after quality control and follow-up filtering.")
    included = add_change_groups(included)
    qc_log.to_csv(output_dir / "quality_control_log.csv", index=False)
    included.to_csv(output_dir / "patient_level_analysis.csv", index=False)

    delta_summary = included["delta_burden_pp"].describe(percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).rename("value").reset_index().rename(columns={"index": "statistic"})
    delta_summary.to_csv(output_dir / "delta_burden_summary.csv", index=False)
    group_counts = included["change_group_at_10pp"].value_counts().rename_axis("change_group").reset_index(name="n")
    group_counts["percentage"] = group_counts["n"] / len(included) * 100
    group_counts.to_csv(output_dir / "change_group_counts_at_10pp.csv", index=False)

    grid, selected = threshold_search(included)
    bootstrap = pd.DataFrame()
    anchor_summary = pd.DataFrame()
    if selected is not None:
        grid.to_csv(output_dir / "threshold_grid.csv", index=False)
        pd.DataFrame([selected]).to_csv(output_dir / "selected_threshold.csv", index=False)
        bootstrap = bootstrap_thresholds(included, args.bootstrap, args.seed)
        bootstrap.to_csv(output_dir / "bootstrap_selected_thresholds.csv", index=False)
        if not bootstrap.empty:
            pd.DataFrame(
                [
                    {
                        "n_bootstrap_success": len(bootstrap),
                        "median_selected_threshold_pp": float(bootstrap["selected_threshold_pp"].median()),
                        "q025_selected_threshold_pp": float(bootstrap["selected_threshold_pp"].quantile(0.025)),
                        "q975_selected_threshold_pp": float(bootstrap["selected_threshold_pp"].quantile(0.975)),
                        "selected_point_estimate_pp": float(selected["threshold_pp"]),
                    }
                ]
            ).to_csv(output_dir / "bootstrap_threshold_summary.csv", index=False)
        anchor_summary = anchor_group_summary(included, float(selected["threshold_pp"]))
        anchor_summary.to_csv(output_dir / "anchor_summary_at_selected_cutoff.csv", index=False)
    else:
        logging.warning("Threshold discovery skipped because a valid binary clinical anchor is unavailable.")
        pd.DataFrame([{"status": "not_estimable", "reason": "missing or insufficient binary clinical anchor"}]).to_csv(output_dir / "threshold_grid.csv", index=False)

    stable_diff_column = args.stable_repeat_diff_column
    mdc_summary, mdc_row = estimate_mdc95(included, stable_diff_column)
    mdc_summary.to_csv(output_dir / "mdc95_summary.csv", index=False)

    if not args.no_plots:
        selected_cutoff = float(selected["threshold_pp"]) if selected is not None else None
        make_figure_1(included, metadata, selected_cutoff, output_dir)
        if selected is not None:
            make_figure_2(grid, selected, output_dir)
            make_figure_3(included, selected, bootstrap, mdc_row, output_dir)
        write_captions(output_dir)

    selected_cutoff_value = float(selected["threshold_pp"]) if selected is not None else None
    metadata.update(
        {
            "analysis_type": "real_data_patient_level_data_driven_threshold_discovery",
            "simulated_values_used": False,
            "n_input_rows": int(len(data)),
            "n_included_rows": int(len(included)),
            "n_anchor_positive": int(included["anchor_positive"].eq(1).sum()),
            "n_anchor_negative": int(included["anchor_positive"].eq(0).sum()),
            "candidate_threshold_grid_pp": [float(x) for x in THRESHOLD_GRID],
            "bootstrap_resamples_requested": int(args.bootstrap),
            "bootstrap_resamples_successful": int(len(bootstrap)) if not bootstrap.empty else 0,
            "selected_threshold_pp": selected_cutoff_value,
            "selected_youden_index": float(selected["youden_index"]) if selected is not None else None,
            "mdc95_pp": float(mdc_row["mdc95_pp"]) if mdc_row is not None else None,
            "threshold_rule": "maximum Youden index; lowest threshold chosen for exact ties",
            "delta_definition": "second Holter minus first Holter, in percentage points",
        }
    )
    (output_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    logging.info("Completed real-data paired-Holter MCID analysis")
    print(f"Completed. Results written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
