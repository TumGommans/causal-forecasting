"""LaTeX table generation helpers for experiment results."""

import os

import numpy as np


def _update_latex_table(results: dict, output_dir: str) -> None:
    """Saves a LaTeX table with RMSE results for all model/DGP combinations.

    Writes ``main_experiment.tex`` to ``<output_dir>/tables/`` using a
    ``threeparttable`` environment. Rows are grouped by learner (X-learner,
    HyX-learner) and RA formula (Acharki et al., Residual correction), with
    separate columns for the MSE and Tweedie first-stage objectives and one
    column per DGP (A, B). Values are formatted as ``mean (std)``.

    Args:
        results: Nested dict keyed by DGP identifier, then learner identifier,
            then variant key, containing ``'mean_rmse'`` and ``'std_rmse'``.
        output_dir: Base results directory containing the ``tables/``
            subdirectory.
    """
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    dgp_keys = ["dgp_a", "dgp_b"]
    dgp_display = {"dgp_a": "DGP A", "dgp_b": "DGP B"}
    learner_keys = ["x_learner", "hyx_learner"]
    learner_display = {"x_learner": "X-learner", "hyx_learner": "HyX-learner"}

    # RA groups: (ra_label, mse_key, tweedie_key)
    ra_groups = [
        ("Acharki et al. (2023)", "acharki_mse", "acharki_tweedie"),
        ("Residual correction",   "ours_mse",    "ours_tweedie"),
    ]

    def fmt(learner_key, ra_obj_key):
        vals = []
        for dgp_key in dgp_keys:
            entry = results.get(dgp_key, {}).get(learner_key, {}).get(ra_obj_key)
            if entry is None:
                vals.append("--")
            else:
                vals.append(f"{entry['mean_rmse']:.3f} ({entry['std_rmse']:.2f})")
        return " & ".join(vals)

    body_lines = []
    for l_idx, learner_key in enumerate(learner_keys):
        l_name = learner_display[learner_key]
        ra1_name, mse1_key, tweedie1_key = ra_groups[0]
        ra2_name, mse2_key, tweedie2_key = ra_groups[1]

        # Four LaTeX rows per learner, with multirow spanning.
        # Lines without \\ are continuations of the same LaTeX row.
        body_lines.append(f"            \\multirow{{4}}{{*}}{{{l_name}}}")
        body_lines.append(f"                & \\multirow{{2}}{{*}}{{{ra1_name}}}")
        body_lines.append(f"                                & MSE     & {fmt(learner_key, mse1_key)} \\\\")
        body_lines.append(f"                &               & Tweedie & {fmt(learner_key, tweedie1_key)} \\\\")
        body_lines.append(f"                & \\multirow{{2}}{{*}}{{{ra2_name}}}")
        body_lines.append(f"                                & MSE     & {fmt(learner_key, mse2_key)} \\\\")
        body_lines.append(f"                &               & Tweedie & {fmt(learner_key, tweedie2_key)} \\\\")

        if l_idx < len(learner_keys) - 1:
            body_lines.append("            \\midrule")

    dgp_header = " & ".join(dgp_display[k] for k in dgp_keys)

    table = (
        "\\begin{table}[H]\n"
        "    \\centering\n"
        "    \\caption{RMSE values for every model and DGP combination. Formatted as mean (std).}\n"
        "    \\label{tab:main_experiment}\n"
        "    \\begin{threeparttable}\n"
        "        \\begin{tabular}{l l l c c}\n"
        "            \\toprule\n"
        f"            Model & RA Formula & Objective\\textsuperscript{{*}} & {dgp_header} \\\\\n"
        "            \\midrule\n"
        + "\n".join(body_lines) + "\n"
        "            \\bottomrule\n"
        "        \\end{tabular}\n"
        "        \\begin{tablenotes}\n"
        "            \\small\n"
        "            \\item[*] Applied to the first stage outcome models.\n"
        "        \\end{tablenotes}\n"
        "    \\end{threeparttable}\n"
        "\\end{table}"
    )

    table_path = os.path.join(tables_dir, "main_experiment.tex")
    with open(table_path, "w") as f:
        f.write(table)


def _save_interference_latex_table(
    no_interf_results: dict,
    interf_results: dict,
    output_dir: str,
) -> None:
    """Saves a LaTeX table comparing all model variants with and without interference.

    Writes ``interference_test.tex`` to ``<output_dir>/tables/`` using a
    ``threeparttable`` environment. The first three columns (Model, RA Formula,
    Objective) mirror the layout of ``main_experiment.tex``, with four rows per
    learner grouped by RA formula via multirow spanning. Columns show bias and
    RMSE for the no-interference baseline (DGP B from the main experiment) and
    the interference scenario side by side.

    Args:
        no_interf_results: Dict keyed by learner identifier, then variant key,
            containing ``'mean_bias'``, ``'std_bias'``, ``'mean_rmse'``, and
            ``'std_rmse'``.  Sourced from ``performance_metrics.json`` DGP B entry.
        interf_results: Same structure as ``no_interf_results`` but for the
            DGP B + interference scenario produced by the stress test.
        output_dir: Base results directory containing the ``tables/``
            subdirectory.
    """
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    learner_keys = ["x_learner", "hyx_learner"]
    learner_display = {"x_learner": "X-learner", "hyx_learner": "HyX-learner"}
    ra_groups = [
        ("Acharki et al. (2023)", "acharki_mse", "acharki_tweedie"),
        ("Residual correction",   "ours_mse",    "ours_tweedie"),
    ]

    def fmt(entry):
        if entry is None:
            return "--"
        return f"{entry['mean_bias']:.3f} ({entry['std_bias']:.2f})"

    def fmt_rmse(entry):
        if entry is None:
            return "--"
        return f"{entry['mean_rmse']:.3f} ({entry['std_rmse']:.2f})"

    def fmt_row(learner_key, ra_obj_key):
        no_e = no_interf_results.get(learner_key, {}).get(ra_obj_key)
        in_e = interf_results.get(learner_key, {}).get(ra_obj_key)
        return (
            f" & {fmt(no_e)} & {fmt_rmse(no_e)}"
            f" & {fmt(in_e)} & {fmt_rmse(in_e)} \\\\"
        )

    body_lines = []
    for l_idx, learner_key in enumerate(learner_keys):
        l_name = learner_display[learner_key]
        ra1_name, mse1_key, tweedie1_key = ra_groups[0]
        ra2_name, mse2_key, tweedie2_key = ra_groups[1]

        body_lines.append(f"            \\multirow{{4}}{{*}}{{{l_name}}}")
        body_lines.append(f"                & \\multirow{{2}}{{*}}{{{ra1_name}}}")
        body_lines.append(f"                                & MSE    {fmt_row(learner_key, mse1_key)}")
        body_lines.append(f"                &               & Tweedie{fmt_row(learner_key, tweedie1_key)}")
        body_lines.append(f"                & \\multirow{{2}}{{*}}{{{ra2_name}}}")
        body_lines.append(f"                                & MSE    {fmt_row(learner_key, mse2_key)}")
        body_lines.append(f"                &               & Tweedie{fmt_row(learner_key, tweedie2_key)}")

        if l_idx < len(learner_keys) - 1:
            body_lines.append("            \\midrule")

    table = (
        "\\begin{table}[H]\n"
        "    \\centering\n"
        "    \\caption{Bias and RMSE values for the interference stress test. Formatted as mean (std).}\n"
        "    \\label{tab:interference_test}\n"
        "    \\begin{threeparttable}\n"
        "        \\begin{tabular}{l l l c c c c}\n"
        "            \\toprule\n"
        "            & & & \\multicolumn{2}{c}{No interference} & \\multicolumn{2}{c}{Interference} \\\\\n"
        "            \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}\n"
        "            Model & RA Formula & Objective\\textsuperscript{*} & Bias & RMSE & Bias & RMSE \\\\\n"
        "            \\midrule\n"
        + "\n".join(body_lines) + "\n"
        "            \\bottomrule\n"
        "        \\end{tabular}\n"
        "        \\begin{tablenotes}\n"
        "            \\small\n"
        "            \\item[*] Applied to the first stage outcome models.\n"
        "        \\end{tablenotes}\n"
        "    \\end{threeparttable}\n"
        "\\end{table}"
    )

    table_path = os.path.join(tables_dir, "interference_test.tex")
    with open(table_path, "w") as f:
        f.write(table)


def _save_real_life_latex_table(
    gates_results: dict,
    output_dir: str,
    treatment_labels: dict,
) -> None:
    """Saves the GATES extreme-groups table to results/tables/real_life_tests.tex.

    Rows are grouped by learner and RA formula (same structure as
    main_experiment.tex).  Columns are the treatment types present in the
    data.  Each cell shows the estimated difference gamma_G - gamma_1 with
    its HC1-robust standard error in parentheses and significance stars from
    the one-sided t-test.

    Args:
        gates_results: Nested dict keyed by learner_key -> ra_obj_key ->
            treatment_val -> {'diff', 'se', 't_stat', 'p_value'}.
        output_dir: Base results directory; table written to tables/ subdir.
        treatment_labels: Dict mapping treatment int code to display name,
            containing only arms present in the training data (ordered).
    """
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    def stars(p: float) -> str:
        if p is None or np.isnan(p):
            return ""
        if p < 0.01:
            return "***"
        if p < 0.05:
            return "**"
        if p < 0.10:
            return "*"
        return ""

    def fmt_cell(entry) -> str:
        if entry is None:
            return "--"
        diff = entry["diff"]
        se = entry["se"]
        p = entry["p_value"]
        if diff is None or np.isnan(diff) or np.isnan(se):
            return "--"
        return f"{diff:.2f}{stars(p)} ({se:.2f})"

    learner_keys = ["x_learner", "hyx_learner"]
    learner_display = {"x_learner": "X-learner", "hyx_learner": "HyX-learner"}
    ra_groups = [
        ("Acharki et al. (2023)", "acharki_mse", "acharki_tweedie"),
        ("Residual correction",   "ours_mse",    "ours_tweedie"),
    ]
    treatment_vals = sorted(treatment_labels.keys())
    treatment_label_list = [treatment_labels[v] for v in treatment_vals]
    n_treat = len(treatment_label_list)

    body_lines = []
    for l_idx, lk in enumerate(learner_keys):
        l_name = learner_display[lk]
        ra1_name, mse1_key, tweedie1_key = ra_groups[0]
        ra2_name, mse2_key, tweedie2_key = ra_groups[1]

        def row_cells(ra_obj_key):
            return " & ".join(
                fmt_cell(gates_results.get(lk, {}).get(ra_obj_key, {}).get(v))
                for v in treatment_vals
            )

        body_lines.append(f"            \\multirow{{4}}{{*}}{{{l_name}}}")
        body_lines.append(f"                & \\multirow{{2}}{{*}}{{{ra1_name}}}")
        body_lines.append(f"                                & MSE     & {row_cells(mse1_key)} \\\\")
        body_lines.append(f"                &               & Tweedie & {row_cells(tweedie1_key)} \\\\")
        body_lines.append(f"                & \\multirow{{2}}{{*}}{{{ra2_name}}}")
        body_lines.append(f"                                & MSE     & {row_cells(mse2_key)} \\\\")
        body_lines.append(f"                &               & Tweedie & {row_cells(tweedie2_key)} \\\\")

        if l_idx < len(learner_keys) - 1:
            body_lines.append("            \\midrule")

    treatment_header = " & ".join(treatment_label_list)

    table = (
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\\caption{Extreme groups test statistics: ($\hat{\gamma}_5(d_k) - \hat{\gamma}_1(d_k)$).}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        f"\\begin{{tabular}}{{lll{'c' * (n_treat)}}}\n"
        "\\toprule\n"
        f"Model & RA Formula & Objective\\textsuperscript{{*}} & {treatment_header} \\\\\n"
        "\\midrule\n"
        + "\n".join(body_lines) + "\n"
        "\\bottomrule\n"
        f"\\multicolumn{{{n_treat + 3}}}{{l}}"
        "{{\\footnotesize Heteroskedasticity-robust standard errors in parentheses.}} \\\\\n"
        f"\\multicolumn{{{n_treat + 3}}}{{l}}"
        "{{\\footnotesize *** $p < 0.01$, ** $p < 0.05$.}} \\\\\n"
        f"\\multicolumn{{{n_treat + 3}}}{{l}}"
        "{{\\footnotesize * Applied to the first stage outcome models.}}\n"
        "\\end{tabular}%\n"
        "}\n"
        "\\end{table}"
    )

    table_path = os.path.join(tables_dir, "real_life_tests.tex")
    with open(table_path, "w") as f:
        f.write(table)
