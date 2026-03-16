"""Visualisation helpers for performance and data distribution plots."""

import os

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

# ---------------------------------------------------------------------------
# Publication-quality matplotlib defaults
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# ---------------------------------------------------------------------------
# Variant styling
# ---------------------------------------------------------------------------

# Tweedie variants (of either RA formula) are distinguished by a red outline.
_TWEEDIE_EDGE = "#CC3232"

# Performance plot styling for all four RA/objective variants.
# Acharki variants use a much lighter blue; residual correction uses a deeper blue.
VARIANT_PLOT_STYLE = {
    "acharki_mse":     {"fill": "#A8C4F0C0", "edge": "#7DA8E0C0", "lw": 1.0},
    "acharki_tweedie": {"fill": "#A8C4F0C0", "edge": _TWEEDIE_EDGE,  "lw": 1.5},
    "ours_mse":        {"fill": "#4350FFC0", "edge": "#2834D6C0", "lw": 1.0},
    "ours_tweedie":    {"fill": "#4350FFC0", "edge": _TWEEDIE_EDGE,  "lw": 1.5},
}


def _plot_performance(
    raw_scores_dgp: dict,
    dgp_key: str,
    dgp_name: str,
    output_dir: str,
) -> None:
    """Generates side-by-side box plots for all four RA/objective variants.

    For each metric (RMSE and bias), creates a figure with learner labels
    (X-learner, HyX-learner) on the x-axis and four box plots per learner,
    ordered: acharki_mse, acharki_tweedie, ours_mse, ours_tweedie.
    Light blue fill = Acharki et al. (2023); blue fill = Residual correction.
    Red outline = Tweedie first-stage objective.

    Args:
        raw_scores_dgp: Dict keyed by learner identifier, each containing
            one key per RA/objective variant, each holding lists ``'rmse'``
            and ``'bias'`` of per-replication values.
        dgp_key: DGP identifier used for output file naming (``'dgp_a'`` or
            ``'dgp_b'``).
        dgp_name: Human-readable DGP label used as the plot title.
        output_dir: Base results directory. Plots are saved under
            ``performance/``.
    """
    perf_dir = os.path.join(output_dir, "performance")
    os.makedirs(perf_dir, exist_ok=True)

    learner_keys = ["x_learner", "hyx_learner"]
    learner_labels = {"x_learner": "X-learner", "hyx_learner": "HyX-learner"}
    learner_positions = [0, 1.6]

    plot_variants = ["acharki_mse", "acharki_tweedie", "ours_mse", "ours_tweedie"]
    n_variants = len(plot_variants)

    group_span = 0.75
    box_offsets = np.linspace(-group_span / 2, group_span / 2, n_variants)
    box_width = group_span / n_variants * 0.80

    for metric in ["rmse", "bias"]:
        fig, ax = plt.subplots(figsize=(7, 5))

        for j, learner_key in enumerate(learner_keys):
            for v_idx, v_key in enumerate(plot_variants):
                style = VARIANT_PLOT_STYLE[v_key]
                pos = learner_positions[j] + box_offsets[v_idx]
                ax.boxplot(
                    raw_scores_dgp[learner_key][v_key][metric],
                    positions=[pos],
                    widths=box_width,
                    patch_artist=True,
                    manage_ticks=False,
                    boxprops=dict(
                        facecolor=style["fill"],
                        color=style["edge"],
                        linewidth=style["lw"],
                    ),
                    medianprops=dict(color=style["edge"], linewidth=style["lw"] + 1.0),
                    whiskerprops=dict(color=style["edge"], linewidth=style["lw"]),
                    capprops=dict(color=style["edge"], linewidth=style["lw"]),
                    flierprops=dict(
                        marker="o",
                        markerfacecolor=style["fill"],
                        markeredgecolor=style["edge"],
                        markersize=3,
                        linewidth=0.5,
                    ),
                )

        if metric == "bias":
            ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)

        ax.set_xticks(learner_positions)
        ax.set_xticklabels([learner_labels[k] for k in learner_keys])
        ax.set_xlim(learner_positions[0] - 0.6, learner_positions[-1] + 0.6)
        ax.set_ylabel("RMSE" if metric == "rmse" else "bias")

        legend_elements = [
            Patch(
                facecolor=VARIANT_PLOT_STYLE["acharki_mse"]["fill"],
                edgecolor=VARIANT_PLOT_STYLE["acharki_mse"]["edge"],
                linewidth=1.0,
                label="Acharki et al. (2023)",
            ),
            Patch(
                facecolor=VARIANT_PLOT_STYLE["ours_mse"]["fill"],
                edgecolor=VARIANT_PLOT_STYLE["ours_mse"]["edge"],
                linewidth=1.0,
                label="CRC",
            ),
            Patch(
                facecolor="#FFFFFF00",
                edgecolor=_TWEEDIE_EDGE,
                linewidth=2.0,
                label="Tweedie loss",
            ),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

        fig.savefig(os.path.join(perf_dir, f"{dgp_key}_{metric}.png"), dpi=300)
        plt.close(fig)


def _plot_per_treatment_rmse(
    per_treatment_raw: dict,
    output_dir: str,
) -> None:
    """Generates per-treatment-level RMSE box plots comparing X-learner and HyX-learner on DGP B.

    Both learners use Tweedie loss with residual correction.  For each
    non-control treatment arm (discount depth on x-axis), draws two
    side-by-side boxes: light blue for X-learner, deep blue for HyX-learner.
    The distribution of each box spans the Monte Carlo replications.

    Args:
        per_treatment_raw: Dict with keys ``'x_learner'`` and ``'hyx_learner'``,
            each mapping treatment value (float) to a list of per-replication
            RMSE values for that arm (DGP B only).
        output_dir: Base results directory. Plot is saved under
            ``performance/per_treatment_rmse.png``.
    """
    perf_dir = os.path.join(output_dir, "performance")
    os.makedirs(perf_dir, exist_ok=True)

    treatment_vals = sorted(per_treatment_raw["x_learner"].keys())
    n_arms = len(treatment_vals)

    # --- Spacing controls -------------------------------------------
    # group_spacing : distance between consecutive treatment-level groups.
    #   Increase to add more horizontal whitespace between pairs.
    #   1.0 = groups touching, 1.5 = half-group gap between pairs.
    # offset        : how far each learner box is nudged left/right from
    #   the group centre.  Keep < group_spacing / 2.
    # box_width     : width of each individual box.  Keep < 2 * offset.
    group_spacing = 1.5
    offset = 0.2
    box_width = 0.35
    # ----------------------------------------------------------------

    positions = np.arange(n_arms, dtype=float) * group_spacing

    learner_styles = {
        "x_learner":   {"fill": "#A8C4F0C0", "edge": "#7DA8E0", "label": "X-learner"},
        "hyx_learner": {"fill": "#4350FFC0", "edge": "#2834D6", "label": "HyX-learner"},
    }
    learner_offsets = {"x_learner": -offset, "hyx_learner": +offset}

    fig, ax = plt.subplots(figsize=(13, 5))

    for learner_key in ["x_learner", "hyx_learner"]:
        style = learner_styles[learner_key]
        off = learner_offsets[learner_key]
        for t_idx, d_val in enumerate(treatment_vals):
            pos = positions[t_idx] + off
            ax.boxplot(
                per_treatment_raw[learner_key][d_val],
                positions=[pos],
                widths=box_width,
                patch_artist=True,
                manage_ticks=False,
                boxprops=dict(facecolor=style["fill"], color=style["edge"], linewidth=1.0),
                medianprops=dict(color=style["edge"], linewidth=2.0),
                whiskerprops=dict(color=style["edge"], linewidth=1.0),
                capprops=dict(color=style["edge"], linewidth=1.0),
                flierprops=dict(
                    marker="o",
                    markerfacecolor=style["fill"],
                    markeredgecolor=style["edge"],
                    markersize=3,
                    linewidth=0.5,
                ),
            )

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{d:.1f}" for d in treatment_vals])
    ax.set_xlabel("discount depth")
    ax.set_ylabel("RMSE")
    ax.set_xlim(positions[0] - group_spacing * 0.6, positions[-1] + group_spacing * 0.6)

    legend_elements = [
        Patch(
            facecolor=learner_styles["x_learner"]["fill"],
            edgecolor=learner_styles["x_learner"]["edge"],
            linewidth=1.0,
            label="X-learner",
        ),
        Patch(
            facecolor=learner_styles["hyx_learner"]["fill"],
            edgecolor=learner_styles["hyx_learner"]["edge"],
            linewidth=1.0,
            label="HyX-learner",
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    fig.savefig(os.path.join(perf_dir, "per_treatment_rmse.png"), dpi=300)
    plt.close(fig)


def _save_real_life_data_plots(
    Y: np.ndarray,
    D: np.ndarray,
    treatment_labels: dict,
    output_dir: str,
) -> None:
    """Saves outcome and treatment distribution plots for the real-life dataset.

    Produces two plots in ``<output_dir>/data/``:
        - ``outcome.png``: histogram of the outcome variable (units sold).
        - ``treatment.png``: bar chart of treatment arm counts, bars sorted
          by frequency descending, labelled with treatment names.

    Styling matches the DGP plots (blue fill ``#8AB3FFC5``, edge ``#0040B7``).

    Args:
        Y: Outcome array for the full dataset (train + test).
        D: Treatment array for the full dataset (train + test).
        treatment_labels: Dict mapping non-zero treatment int codes to names.
            Control (D=0) is always included as ``'Control'``.
        output_dir: Base results directory. Plots are saved under ``data/``.
    """
    data_dir = os.path.join(output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # --- Outcome plot -------------------------------------------------------
    # Clip to 99th percentile to suppress outlier-driven scale compression.
    x_max = np.percentile(Y, 99)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(Y, bins=50, color="#8AB3FF80", edgecolor="#0040B7", linewidth=0.5,
            range=(0, x_max))
    ax.set_xlabel("sales")
    ax.set_ylabel("density")
    ax.set_xlim(0, x_max)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.set_yticks([])
    fig.savefig(os.path.join(data_dir, "outcome.png"), dpi=300)
    plt.close(fig)

    # --- Treatment plot -----------------------------------------------------
    all_labels = {0: "Control", **{int(k): v for k, v in treatment_labels.items()}}
    counts = {tv: int(np.sum(D == tv)) for tv in all_labels}
    # Sort bars by count descending
    sorted_arms = sorted(counts.keys(), key=lambda tv: counts[tv], reverse=True)
    labels = [all_labels[tv] for tv in sorted_arms]
    heights = [counts[tv] for tv in sorted_arms]
    x_pos = np.arange(len(sorted_arms))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        x_pos,
        heights,
        color="#8AB3FFC5",
        edgecolor="#0040B7",
        linewidth=0.2,
        width=0.6,
    )
    ax.set_yscale("log")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_xlabel("treatment")
    ax.set_ylabel("count (log scale)")
    ax.yaxis.set_tick_params(labelleft=True)
    fig.savefig(os.path.join(data_dir, "treatment.png"), dpi=300)
    plt.close(fig)
