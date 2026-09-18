#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 13:55:13 2026

@author: temilolagbadamosi-adeniyi
"""

"""
CNC Manufacturing Process Quality Improvement
Improve Phase — Sensitivity Analysis
"""

#%% IMPORTS

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures


#%% PROJECT PATHS

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

PROJECT_ROOT = SCRIPT_DIR.parent

DATA_PATH = (
    PROJECT_ROOT
    / "CNC"
    / "manufacturing_quality_improvement_data.csv"
)

OUTPUT_DIR = SCRIPT_DIR / "outputs"
FIGURE_DIR = SCRIPT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


#%% CONSTANTS

ROUGHNESS_USL = 1.60
BURR_USL = 0.080

WEAR_THRESHOLDS = [
    0.60,
    0.65,
    0.70,
    0.725,
    0.75,
    0.775,
    0.80,
    0.825,
    0.85,
    0.875,
    0.90,
]


#%% HELPER FUNCTIONS

def save_figure(fig, filename):
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / filename,
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
    plt.close(fig)


def fit_linear_model(data, x_col, y_col):
    X = sm.add_constant(data[[x_col]])
    y = data[y_col]
    return sm.OLS(y, X).fit()


def compare_linear_quadratic(data, x_col, y_col):
    X = data[[x_col]].values
    y = data[y_col].values

    linear_model = LinearRegression()
    linear_model.fit(X, y)
    linear_r2 = r2_score(
        y,
        linear_model.predict(X),
    )

    polynomial = PolynomialFeatures(
        degree=2,
        include_bias=False,
    )

    X_quadratic = polynomial.fit_transform(X)

    quadratic_model = LinearRegression()
    quadratic_model.fit(X_quadratic, y)

    quadratic_r2 = r2_score(
        y,
        quadratic_model.predict(X_quadratic),
    )

    return {
        "linear_model": linear_model,
        "quadratic_model": quadratic_model,
        "polynomial": polynomial,
        "linear_r2": linear_r2,
        "quadratic_r2": quadratic_r2,
    }


#%% QUESTION 1 — TOOL WEAR SENSITIVITY

def question_1_tool_wear_sensitivity(df):

    analysis_df = df[
        [
            "tool_wear",
            "surface_roughness_um",
            "burr_height_mm",
        ]
    ].dropna().copy()

    analysis_df["roughness_defect"] = (
        analysis_df["surface_roughness_um"]
        > ROUGHNESS_USL
    ).astype(int)

    analysis_df["burr_defect"] = (
        analysis_df["burr_height_mm"]
        > BURR_USL
    ).astype(int)

    # Tool-wear bins
    wear_bins = np.arange(
        0.0,
        1.01,
        0.1,
    )

    analysis_df["wear_bin"] = pd.cut(
        analysis_df["tool_wear"],
        bins=wear_bins,
        right=False,
        include_lowest=True,
    )

    wear_summary = (
        analysis_df
        .groupby(
            "wear_bin",
            observed=False,
        )
        .agg(
            parts=("tool_wear", "size"),
            mean_tool_wear=("tool_wear", "mean"),

            mean_roughness=(
                "surface_roughness_um",
                "mean",
            ),
            median_roughness=(
                "surface_roughness_um",
                "median",
            ),
            std_roughness=(
                "surface_roughness_um",
                "std",
            ),
            max_roughness=(
                "surface_roughness_um",
                "max",
            ),
            roughness_defects=(
                "roughness_defect",
                "sum",
            ),

            mean_burr=(
                "burr_height_mm",
                "mean",
            ),
            median_burr=(
                "burr_height_mm",
                "median",
            ),
            std_burr=(
                "burr_height_mm",
                "std",
            ),
            max_burr=(
                "burr_height_mm",
                "max",
            ),
            burr_defects=(
                "burr_defect",
                "sum",
            ),
        )
        .reset_index()
    )

    wear_summary[
        "roughness_defect_rate_pct"
    ] = (
        100
        * wear_summary["roughness_defects"]
        / wear_summary["parts"]
    )

    wear_summary[
        "burr_defect_rate_pct"
    ] = (
        100
        * wear_summary["burr_defects"]
        / wear_summary["parts"]
    )

    # Consecutive-bin changes
    wear_summary[
        "roughness_mean_change"
    ] = wear_summary[
        "mean_roughness"
    ].diff()

    wear_summary[
        "roughness_defect_rate_change"
    ] = wear_summary[
        "roughness_defect_rate_pct"
    ].diff()

    wear_summary[
        "burr_mean_change"
    ] = wear_summary[
        "mean_burr"
    ].diff()

    wear_summary[
        "burr_defect_rate_change"
    ] = wear_summary[
        "burr_defect_rate_pct"
    ].diff()

    # Correlations
    roughness_corr = (
        analysis_df["tool_wear"]
        .corr(
            analysis_df[
                "surface_roughness_um"
            ]
        )
    )

    burr_corr = (
        analysis_df["tool_wear"]
        .corr(
            analysis_df["burr_height_mm"]
        )
    )

    # Linear models
    roughness_linear = fit_linear_model(
        analysis_df,
        "tool_wear",
        "surface_roughness_um",
    )

    burr_linear = fit_linear_model(
        analysis_df,
        "tool_wear",
        "burr_height_mm",
    )

    # Linear vs quadratic
    roughness_compare = (
        compare_linear_quadratic(
            analysis_df,
            "tool_wear",
            "surface_roughness_um",
        )
    )

    burr_compare = (
        compare_linear_quadratic(
            analysis_df,
            "tool_wear",
            "burr_height_mm",
        )
    )

    # Low-wear vs high-wear comparison
    cutoff_rows = []

    for cutoff in [
        0.60,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
    ]:

        low = analysis_df[
            analysis_df["tool_wear"] < cutoff
        ]

        high = analysis_df[
            analysis_df["tool_wear"] >= cutoff
        ]

        if low.empty or high.empty:
            continue

        cutoff_rows.append(
            {
                "wear_cutoff": cutoff,
                "parts_below": len(low),
                "parts_at_or_above": len(high),

                "roughness_mean_below":
                    low[
                        "surface_roughness_um"
                    ].mean(),

                "roughness_mean_high":
                    high[
                        "surface_roughness_um"
                    ].mean(),

                "roughness_defect_rate_below_pct":
                    100
                    * low[
                        "roughness_defect"
                    ].mean(),

                "roughness_defect_rate_high_pct":
                    100
                    * high[
                        "roughness_defect"
                    ].mean(),

                "burr_mean_below":
                    low[
                        "burr_height_mm"
                    ].mean(),

                "burr_mean_high":
                    high[
                        "burr_height_mm"
                    ].mean(),

                "burr_defect_rate_below_pct":
                    100
                    * low[
                        "burr_defect"
                    ].mean(),

                "burr_defect_rate_high_pct":
                    100
                    * high[
                        "burr_defect"
                    ].mean(),
            }
        )

    cutoff_comparison = pd.DataFrame(
        cutoff_rows
    )

    # Figures
    x_grid = np.linspace(
        analysis_df["tool_wear"].min(),
        analysis_df["tool_wear"].max(),
        300,
    ).reshape(-1, 1)

    # Roughness vs wear
    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.scatter(
        analysis_df["tool_wear"],
        analysis_df["surface_roughness_um"],
        alpha=0.25,
        s=18,
    )

    ax.axhline(
        ROUGHNESS_USL,
        linestyle="--",
        linewidth=2,
        label=f"USL = {ROUGHNESS_USL:.2f} µm",
    )

    x_quad = roughness_compare[
        "polynomial"
    ].transform(x_grid)

    y_grid = roughness_compare[
        "quadratic_model"
    ].predict(x_quad)

    ax.plot(
        x_grid.flatten(),
        y_grid,
        linewidth=2,
        label="Quadratic trend",
    )

    ax.set_xlabel("Tool Wear")
    ax.set_ylabel("Surface Roughness (µm)")
    ax.set_title(
        "Surface Roughness vs Tool Wear"
    )
    ax.legend()

    save_figure(
        fig,
        "q1_roughness_vs_tool_wear.png",
    )

    # Burr vs wear
    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.scatter(
        analysis_df["tool_wear"],
        analysis_df["burr_height_mm"],
        alpha=0.25,
        s=18,
    )

    ax.axhline(
        BURR_USL,
        linestyle="--",
        linewidth=2,
        label=f"USL = {BURR_USL:.3f} mm",
    )

    x_quad = burr_compare[
        "polynomial"
    ].transform(x_grid)

    y_grid = burr_compare[
        "quadratic_model"
    ].predict(x_quad)

    ax.plot(
        x_grid.flatten(),
        y_grid,
        linewidth=2,
        label="Quadratic trend",
    )

    ax.set_xlabel("Tool Wear")
    ax.set_ylabel("Burr Height (mm)")
    ax.set_title(
        "Burr Height vs Tool Wear"
    )
    ax.legend()

    save_figure(
        fig,
        "q1_burr_vs_tool_wear.png",
    )

    # Roughness defect rate
    plot_df = wear_summary.dropna(
        subset=["mean_tool_wear"]
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        plot_df["mean_tool_wear"],
        plot_df[
            "roughness_defect_rate_pct"
        ],
        marker="o",
    )

    ax.set_xlabel(
        "Mean Tool Wear Within Bin"
    )
    ax.set_ylabel(
        "Roughness Defect Rate (%)"
    )
    ax.set_title(
        "Surface Roughness Defect Rate vs Tool Wear"
    )

    save_figure(
        fig,
        "q1_roughness_defect_rate_vs_wear.png",
    )

    # Burr defect rate
    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        plot_df["mean_tool_wear"],
        plot_df[
            "burr_defect_rate_pct"
        ],
        marker="o",
    )

    ax.set_xlabel(
        "Mean Tool Wear Within Bin"
    )
    ax.set_ylabel(
        "Burr Defect Rate (%)"
    )
    ax.set_title(
        "Burr Defect Rate vs Tool Wear"
    )

    save_figure(
        fig,
        "q1_burr_defect_rate_vs_wear.png",
    )

    # Exports
    wear_summary.to_csv(
        OUTPUT_DIR
        / "sensitivity_q1_tool_wear_summary.csv",
        index=False,
    )

    cutoff_comparison.to_csv(
        OUTPUT_DIR
        / "sensitivity_q1_cutoff_comparison.csv",
        index=False,
    )

    # Results
    print("\nQUESTION 1 — TOOL WEAR SENSITIVITY")

    print(
        f"\nTool wear vs roughness correlation: "
        f"{roughness_corr:.4f}"
    )

    print(
        f"Tool wear vs burr correlation: "
        f"{burr_corr:.4f}"
    )

    print("\nROUGHNESS LINEAR MODEL")
    print(roughness_linear.summary())

    print("\nBURR LINEAR MODEL")
    print(burr_linear.summary())

    print("\nLINEAR VS QUADRATIC")

    comparison = pd.DataFrame(
        {
            "CTQ": [
                "Surface Roughness",
                "Burr Height",
            ],
            "Linear R2": [
                roughness_compare[
                    "linear_r2"
                ],
                burr_compare[
                    "linear_r2"
                ],
            ],
            "Quadratic R2": [
                roughness_compare[
                    "quadratic_r2"
                ],
                burr_compare[
                    "quadratic_r2"
                ],
            ],
        }
    )

    comparison["R2 Improvement"] = (
        comparison["Quadratic R2"]
        - comparison["Linear R2"]
    )

    print(
        comparison
        .round(6)
        .to_string(index=False)
    )

    print("\nTOOL-WEAR BIN SUMMARY")
    print(
        wear_summary
        .round(6)
        .to_string(index=False)
    )

    print("\nWEAR CUTOFF COMPARISON")
    print(
        cutoff_comparison
        .round(6)
        .to_string(index=False)
    )


#%% QUESTION 2 — PREVENTIVE TOOL-REPLACEMENT THRESHOLD

def question_2_replacement_threshold(df):

    analysis_df = df[
        [
            "tool_wear",
            "surface_roughness_um",
            "burr_height_mm",
        ]
    ].dropna().copy()

    analysis_df["roughness_defect"] = (
        analysis_df["surface_roughness_um"]
        > ROUGHNESS_USL
    ).astype(int)

    analysis_df["burr_defect"] = (
        analysis_df["burr_height_mm"]
        > BURR_USL
    ).astype(int)

    analysis_df["target_ctq_defect"] = (
        (
            analysis_df["roughness_defect"]
            == 1
        )
        |
        (
            analysis_df["burr_defect"]
            == 1
        )
    ).astype(int)

    n_total = len(analysis_df)

    baseline_combined_rate = (
        100
        * analysis_df[
            "target_ctq_defect"
        ].mean()
    )

    max_observed_wear = (
        analysis_df["tool_wear"].max()
    )

    rows = []

    for threshold in WEAR_THRESHOLDS:

        retained = analysis_df[
            analysis_df["tool_wear"]
            < threshold
        ]

        excluded = analysis_df[
            analysis_df["tool_wear"]
            >= threshold
        ]

        if retained.empty:
            continue

        n_retained = len(retained)
        n_excluded = len(excluded)

        rows.append(
            {
                "replacement_threshold":
                    threshold,

                "wear_utilization_pct":
                    100
                    * threshold
                    / max_observed_wear,

                "historical_parts_below_threshold":
                    n_retained,

                "historical_parts_at_or_above_threshold":
                    n_excluded,

                "production_region_retained_pct":
                    100
                    * n_retained
                    / n_total,

                "roughness_defect_rate_below_pct":
                    100
                    * retained[
                        "roughness_defect"
                    ].mean(),

                "burr_defect_rate_below_pct":
                    100
                    * retained[
                        "burr_defect"
                    ].mean(),

                "combined_defect_rate_below_pct":
                    100
                    * retained[
                        "target_ctq_defect"
                    ].mean(),

                "roughness_defect_rate_high_pct":
                    (
                        100
                        * excluded[
                            "roughness_defect"
                        ].mean()
                        if n_excluded
                        else np.nan
                    ),

                "burr_defect_rate_high_pct":
                    (
                        100
                        * excluded[
                            "burr_defect"
                        ].mean()
                        if n_excluded
                        else np.nan
                    ),

                "combined_defect_rate_high_pct":
                    (
                        100
                        * excluded[
                            "target_ctq_defect"
                        ].mean()
                        if n_excluded
                        else np.nan
                    ),

                "roughness_defects_in_high_wear_region":
                    excluded[
                        "roughness_defect"
                    ].sum(),

                "burr_defects_in_high_wear_region":
                    excluded[
                        "burr_defect"
                    ].sum(),

                "combined_defective_parts_in_high_wear_region":
                    excluded[
                        "target_ctq_defect"
                    ].sum(),
            }
        )

    threshold_summary = pd.DataFrame(
        rows
    )

    threshold_summary[
        "relative_quality_risk_reduction_pct"
    ] = (
        100
        * (
            baseline_combined_rate
            - threshold_summary[
                "combined_defect_rate_below_pct"
            ]
        )
        / baseline_combined_rate
    )

    threshold_summary[
        "delta_wear_utilization_pct"
    ] = threshold_summary[
        "wear_utilization_pct"
    ].diff()

    threshold_summary[
        "delta_combined_defect_rate_pct"
    ] = threshold_summary[
        "combined_defect_rate_below_pct"
    ].diff()

    # Quality risk vs threshold
    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        threshold_summary[
            "replacement_threshold"
        ],
        threshold_summary[
            "combined_defect_rate_below_pct"
        ],
        marker="o",
    )

    ax.set_xlabel(
        "Candidate Tool-Replacement Threshold"
    )
    ax.set_ylabel(
        "Observed Roughness/Burr Defect Rate Below Threshold (%)"
    )
    ax.set_title(
        "Quality Risk vs Preventive Tool-Replacement Threshold"
    )

    save_figure(
        fig,
        "q2_quality_risk_vs_threshold.png",
    )

    # Tool utilization
    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        threshold_summary[
            "replacement_threshold"
        ],
        threshold_summary[
            "wear_utilization_pct"
        ],
        marker="o",
    )

    ax.set_xlabel(
        "Candidate Tool-Replacement Threshold"
    )
    ax.set_ylabel(
        "Tool Wear Utilization (%)"
    )
    ax.set_title(
        "Tool Utilization vs Preventive Replacement Threshold"
    )

    save_figure(
        fig,
        "q2_utilization_vs_threshold.png",
    )

    # Quality-utilization tradeoff
    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        threshold_summary[
            "wear_utilization_pct"
        ],
        threshold_summary[
            "combined_defect_rate_below_pct"
        ],
        marker="o",
    )

    for _, row in (
        threshold_summary.iterrows()
    ):
        ax.annotate(
            f"{row['replacement_threshold']:.3f}",
            (
                row[
                    "wear_utilization_pct"
                ],
                row[
                    "combined_defect_rate_below_pct"
                ],
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel(
        "Tool Wear Utilization (%)"
    )
    ax.set_ylabel(
        "Observed Roughness/Burr Defect Rate Below Threshold (%)"
    )
    ax.set_title(
        "Quality–Tool Utilization Tradeoff"
    )

    save_figure(
        fig,
        "q2_quality_utilization_tradeoff.png",
    )

    # High-wear risk
    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        threshold_summary[
            "replacement_threshold"
        ],
        threshold_summary[
            "combined_defect_rate_high_pct"
        ],
        marker="o",
    )

    ax.set_xlabel(
        "Tool-Replacement Threshold"
    )
    ax.set_ylabel(
        "Observed Defect Rate at/Above Threshold (%)"
    )
    ax.set_title(
        "Quality Risk in High-Wear Region"
    )

    save_figure(
        fig,
        "q2_high_wear_quality_risk.png",
    )

    threshold_summary.to_csv(
        OUTPUT_DIR
        / "sensitivity_q2_tool_replacement_thresholds.csv",
        index=False,
    )

    print(
        "\nQUESTION 2 — PREVENTIVE TOOL-REPLACEMENT THRESHOLD"
    )

    print(
        f"\nBaseline combined Roughness/Burr defect rate: "
        f"{baseline_combined_rate:.4f}%"
    )

    print(
        "\nCANDIDATE THRESHOLD COMPARISON"
    )

    print(
        threshold_summary
        .round(4)
        .to_string(index=False)
    )


#%% QUESTION 3 — FEED-RATE SENSITIVITY FOR BURR

def question_3_feed_rate_sensitivity(df):

    columns = [
        "feed_rate_mm_rev",
        "tool_wear",
        "material_hardness_effect",
        "burr_height_mm",
    ]

    analysis_df = (
        df[columns]
        .dropna()
        .copy()
    )

    analysis_df["burr_defect"] = (
        analysis_df["burr_height_mm"]
        > BURR_USL
    ).astype(int)

    raw_corr = (
        analysis_df["feed_rate_mm_rev"]
        .corr(
            analysis_df["burr_height_mm"]
        )
    )

    analysis_df["feed_bin"] = pd.qcut(
        analysis_df["feed_rate_mm_rev"],
        q=10,
        duplicates="drop",
    )

    feed_summary = (
        analysis_df
        .groupby(
            "feed_bin",
            observed=True,
        )
        .agg(
            parts=(
                "feed_rate_mm_rev",
                "size",
            ),
            mean_feed=(
                "feed_rate_mm_rev",
                "mean",
            ),
            min_feed=(
                "feed_rate_mm_rev",
                "min",
            ),
            max_feed=(
                "feed_rate_mm_rev",
                "max",
            ),
            mean_tool_wear=(
                "tool_wear",
                "mean",
            ),
            mean_hardness=(
                "material_hardness_effect",
                "mean",
            ),
            mean_burr=(
                "burr_height_mm",
                "mean",
            ),
            median_burr=(
                "burr_height_mm",
                "median",
            ),
            std_burr=(
                "burr_height_mm",
                "std",
            ),
            max_burr=(
                "burr_height_mm",
                "max",
            ),
            burr_defects=(
                "burr_defect",
                "sum",
            ),
        )
        .reset_index()
    )

    feed_summary[
        "burr_defect_rate_pct"
    ] = (
        100
        * feed_summary["burr_defects"]
        / feed_summary["parts"]
    )

    raw_model = smf.ols(
        """
        burr_height_mm
        ~ feed_rate_mm_rev
        """,
        data=analysis_df,
    ).fit()

    adjusted_model = smf.ols(
        """
        burr_height_mm
        ~ tool_wear
        + feed_rate_mm_rev
        + material_hardness_effect
        """,
        data=analysis_df,
    ).fit()

    feed_min = (
        analysis_df[
            "feed_rate_mm_rev"
        ].min()
    )

    feed_max = (
        analysis_df[
            "feed_rate_mm_rev"
        ].max()
    )

    feed_grid = np.linspace(
        feed_min,
        feed_max,
        300,
    )

    median_wear = (
        analysis_df["tool_wear"].median()
    )

    median_hardness = (
        analysis_df[
            "material_hardness_effect"
        ].median()
    )

    prediction_df = pd.DataFrame(
        {
            "tool_wear": median_wear,
            "feed_rate_mm_rev": feed_grid,
            "material_hardness_effect":
                median_hardness,
        }
    )

    prediction = (
        adjusted_model
        .get_prediction(prediction_df)
        .summary_frame(alpha=0.05)
    )

    prediction_df[
        "predicted_burr"
    ] = prediction["mean"].values

    prediction_df[
        "ci_lower"
    ] = prediction[
        "mean_ci_lower"
    ].values

    prediction_df[
        "ci_upper"
    ] = prediction[
        "mean_ci_upper"
    ].values

    candidate_feeds = [
        value
        for value in [
            0.150,
            0.160,
            0.170,
            0.180,
            0.190,
            0.200,
            0.210,
            0.220,
        ]
        if feed_min <= value <= feed_max
    ]

    candidate_prediction_df = pd.DataFrame(
        {
            "feed_rate_mm_rev":
                candidate_feeds,
            "tool_wear":
                median_wear,
            "material_hardness_effect":
                median_hardness,
        }
    )

    candidate_predictions = (
        adjusted_model
        .get_prediction(
            candidate_prediction_df
        )
        .summary_frame(alpha=0.05)
    )

    candidate_prediction_df[
        "adjusted_predicted_burr"
    ] = candidate_predictions[
        "mean"
    ].values

    candidate_prediction_df[
        "ci_lower"
    ] = candidate_predictions[
        "mean_ci_lower"
    ].values

    candidate_prediction_df[
        "ci_upper"
    ] = candidate_predictions[
        "mean_ci_upper"
    ].values

    # High-wear predictions
    high_wear_prediction_df = (
        candidate_prediction_df[
            [
                "feed_rate_mm_rev",
                "material_hardness_effect",
            ]
        ].copy()
    )

    high_wear_prediction_df[
        "tool_wear"
    ] = 0.75

    high_wear_predictions = (
        adjusted_model
        .get_prediction(
            high_wear_prediction_df[
                [
                    "feed_rate_mm_rev",
                    "tool_wear",
                    "material_hardness_effect",
                ]
            ]
        )
        .summary_frame(alpha=0.05)
    )

    high_wear_prediction_df[
        "adjusted_predicted_burr"
    ] = high_wear_predictions[
        "mean"
    ].values

    # Raw Burr vs feed
    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.scatter(
        analysis_df["feed_rate_mm_rev"],
        analysis_df["burr_height_mm"],
        alpha=0.25,
        s=18,
    )

    ax.axhline(
        BURR_USL,
        linestyle="--",
        linewidth=2,
        label=f"USL = {BURR_USL:.3f} mm",
    )

    ax.set_xlabel(
        "Feed Rate (mm/rev)"
    )
    ax.set_ylabel(
        "Burr Height (mm)"
    )
    ax.set_title(
        "Raw Burr Height vs Feed Rate"
    )
    ax.legend()

    save_figure(
        fig,
        "q3_raw_burr_vs_feed.png",
    )

    # Mean Burr by feed bin
    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        feed_summary["mean_feed"],
        feed_summary["mean_burr"],
        marker="o",
    )

    ax.set_xlabel(
        "Mean Feed Rate Within Bin (mm/rev)"
    )
    ax.set_ylabel(
        "Mean Burr Height (mm)"
    )
    ax.set_title(
        "Mean Burr Height vs Feed Rate"
    )

    save_figure(
        fig,
        "q3_mean_burr_vs_feed.png",
    )

    # Burr defect rate
    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        feed_summary["mean_feed"],
        feed_summary[
            "burr_defect_rate_pct"
        ],
        marker="o",
    )

    ax.set_xlabel(
        "Mean Feed Rate Within Bin (mm/rev)"
    )
    ax.set_ylabel(
        "Burr Defect Rate (%)"
    )
    ax.set_title(
        "Burr Defect Rate vs Feed Rate"
    )

    save_figure(
        fig,
        "q3_burr_defect_rate_vs_feed.png",
    )

    # Adjusted feed effect
    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        prediction_df[
            "feed_rate_mm_rev"
        ],
        prediction_df[
            "predicted_burr"
        ],
        linewidth=2,
        label="Adjusted predicted Burr",
    )

    ax.fill_between(
        prediction_df[
            "feed_rate_mm_rev"
        ],
        prediction_df["ci_lower"],
        prediction_df["ci_upper"],
        alpha=0.2,
        label="95% confidence interval",
    )

    ax.axhline(
        BURR_USL,
        linestyle="--",
        linewidth=2,
        label=f"USL = {BURR_USL:.3f} mm",
    )

    ax.set_xlabel(
        "Feed Rate (mm/rev)"
    )
    ax.set_ylabel(
        "Adjusted Predicted Burr Height (mm)"
    )
    ax.set_title(
        "Adjusted Burr Sensitivity to Feed Rate"
    )
    ax.legend()

    save_figure(
        fig,
        "q3_adjusted_feed_effect.png",
    )

    feed_summary.to_csv(
        OUTPUT_DIR
        / "sensitivity_q3_feed_summary.csv",
        index=False,
    )

    candidate_prediction_df.to_csv(
        OUTPUT_DIR
        / "sensitivity_q3_adjusted_feed_predictions.csv",
        index=False,
    )

    high_wear_prediction_df.to_csv(
        OUTPUT_DIR
        / "sensitivity_q3_high_wear_feed_predictions.csv",
        index=False,
    )

    print(
        "\nQUESTION 3 — FEED-RATE SENSITIVITY FOR BURR"
    )

    print(
        f"\nRaw feed/Burr correlation: "
        f"{raw_corr:.4f}"
    )

    print("\nRAW MODEL")
    print(raw_model.summary())

    print("\nADJUSTED MODEL")
    print(adjusted_model.summary())

    print(
        "\nADJUSTED BURR AT CANDIDATE FEED SETTINGS"
    )

    print(
        candidate_prediction_df
        .round(6)
        .to_string(index=False)
    )

    print(
        "\nADJUSTED FEED SENSITIVITY AT TOOL WEAR = 0.75"
    )

    print(
        high_wear_prediction_df
        .round(6)
        .to_string(index=False)
    )


#%% QUESTION 4 — TOOL WEAR × FEED-RATE INTERACTION

def question_4_wear_feed_interaction(df):

    columns = [
        "tool_wear",
        "feed_rate_mm_rev",
        "material_hardness_effect",
        "burr_height_mm",
    ]

    analysis_df = (
        df[columns]
        .dropna()
        .copy()
    )

    base_model = smf.ols(
        """
        burr_height_mm
        ~ tool_wear
        + feed_rate_mm_rev
        + material_hardness_effect
        """,
        data=analysis_df,
    ).fit()

    interaction_model = smf.ols(
        """
        burr_height_mm
        ~ tool_wear * feed_rate_mm_rev
        + material_hardness_effect
        """,
        data=analysis_df,
    ).fit()

    interaction_term = (
        "tool_wear:feed_rate_mm_rev"
    )

    interaction_coef = (
        interaction_model.params[
            interaction_term
        ]
    )

    interaction_pvalue = (
        interaction_model.pvalues[
            interaction_term
        ]
    )

    interaction_ci = (
        interaction_model
        .conf_int()
        .loc[interaction_term]
    )

    feed_main_coef = (
        interaction_model.params[
            "feed_rate_mm_rev"
        ]
    )

    wear_levels = [
        0.20,
        0.40,
        0.60,
        0.70,
        0.75,
        0.775,
        0.80,
        0.90,
    ]

    sensitivity_rows = []

    for wear in wear_levels:

        effective_feed_slope = (
            feed_main_coef
            + interaction_coef * wear
        )

        sensitivity_rows.append(
            {
                "tool_wear": wear,
                "effective_feed_slope":
                    effective_feed_slope,
                "burr_change_per_plus_0.010_feed_mm":
                    effective_feed_slope
                    * 0.010,
            }
        )

    feed_sensitivity_by_wear = (
        pd.DataFrame(
            sensitivity_rows
        )
    )

    # Prediction grid
    candidate_wear = [
        0.25,
        0.50,
        0.70,
        0.75,
        0.775,
        0.80,
        0.90,
    ]

    candidate_feed = [
        0.170,
        0.180,
        0.185,
        0.190,
        0.195,
        0.200,
    ]

    median_hardness = (
        analysis_df[
            "material_hardness_effect"
        ].median()
    )

    prediction_rows = []

    for wear in candidate_wear:
        for feed in candidate_feed:

            prediction_rows.append(
                {
                    "tool_wear": wear,
                    "feed_rate_mm_rev":
                        feed,
                    "material_hardness_effect":
                        median_hardness,
                }
            )

    prediction_df = pd.DataFrame(
        prediction_rows
    )

    prediction_result = (
        interaction_model
        .get_prediction(prediction_df)
        .summary_frame(alpha=0.05)
    )

    prediction_df[
        "predicted_burr_mm"
    ] = prediction_result[
        "mean"
    ].values

    prediction_df[
        "ci_lower"
    ] = prediction_result[
        "mean_ci_lower"
    ].values

    prediction_df[
        "ci_upper"
    ] = prediction_result[
        "mean_ci_upper"
    ].values

    # Raw descriptive interaction
    wear_bins = [
        0.0,
        0.4,
        0.6,
        0.7,
        0.775,
        1.0,
    ]

    wear_labels = [
        "<0.40",
        "0.40-0.60",
        "0.60-0.70",
        "0.70-0.775",
        ">=0.775",
    ]

    analysis_df["wear_region"] = pd.cut(
        analysis_df["tool_wear"],
        bins=wear_bins,
        labels=wear_labels,
        right=False,
        include_lowest=True,
    )

    median_feed = (
        analysis_df[
            "feed_rate_mm_rev"
        ].median()
    )

    analysis_df["feed_region"] = (
        np.where(
            analysis_df[
                "feed_rate_mm_rev"
            ] <= median_feed,
            "Lower feed",
            "Higher feed",
        )
    )

    raw_interaction_summary = (
        analysis_df
        .groupby(
            [
                "wear_region",
                "feed_region",
            ],
            observed=True,
        )
        .agg(
            parts=(
                "burr_height_mm",
                "size",
            ),
            mean_wear=(
                "tool_wear",
                "mean",
            ),
            mean_feed=(
                "feed_rate_mm_rev",
                "mean",
            ),
            mean_burr=(
                "burr_height_mm",
                "mean",
            ),
            median_burr=(
                "burr_height_mm",
                "median",
            ),
        )
        .reset_index()
    )

    # Predicted Burr across feed
    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for wear in candidate_wear:

        subset = prediction_df[
            prediction_df["tool_wear"]
            == wear
        ]

        ax.plot(
            subset["feed_rate_mm_rev"],
            subset["predicted_burr_mm"],
            marker="o",
            label=f"Wear = {wear:.3f}",
        )

    ax.axhline(
        BURR_USL,
        linestyle="--",
        linewidth=2,
        label=f"USL = {BURR_USL:.3f} mm",
    )

    ax.set_xlabel(
        "Feed Rate (mm/rev)"
    )
    ax.set_ylabel(
        "Adjusted Predicted Burr Height (mm)"
    )
    ax.set_title(
        "Predicted Burr Height Across Feed Rates "
        "at Different Tool-Wear Levels"
    )
    ax.legend()

    save_figure(
        fig,
        "q4_wear_feed_interaction.png",
    )

    # Effective feed sensitivity
    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        feed_sensitivity_by_wear[
            "tool_wear"
        ],
        feed_sensitivity_by_wear[
            "burr_change_per_plus_0.010_feed_mm"
        ],
        marker="o",
    )

    ax.set_xlabel("Tool Wear")
    ax.set_ylabel(
        "Burr Change per +0.010 mm/rev Feed (mm)"
    )
    ax.set_title(
        "Feed Sensitivity Across Tool Wear"
    )

    save_figure(
        fig,
        "q4_feed_sensitivity_by_wear.png",
    )

    feed_sensitivity_by_wear.to_csv(
        OUTPUT_DIR
        / "sensitivity_q4_feed_sensitivity_by_wear.csv",
        index=False,
    )

    prediction_df.to_csv(
        OUTPUT_DIR
        / "sensitivity_q4_wear_feed_predictions.csv",
        index=False,
    )

    raw_interaction_summary.to_csv(
        OUTPUT_DIR
        / "sensitivity_q4_raw_interaction_summary.csv",
        index=False,
    )

    print(
        "\nQUESTION 4 — TOOL WEAR × FEED-RATE INTERACTION"
    )

    print("\nBASE MODEL")
    print(base_model.summary())

    print("\nINTERACTION MODEL")
    print(interaction_model.summary())

    print("\nMODEL COMPARISON")

    print(
        f"Base R²        : "
        f"{base_model.rsquared:.6f}"
    )

    print(
        f"Interaction R² : "
        f"{interaction_model.rsquared:.6f}"
    )

    print(
        f"R² improvement : "
        f"{interaction_model.rsquared - base_model.rsquared:.6f}"
    )

    print(
        f"Base AIC        : "
        f"{base_model.aic:.3f}"
    )

    print(
        f"Interaction AIC : "
        f"{interaction_model.aic:.3f}"
    )

    print(
        f"Base BIC        : "
        f"{base_model.bic:.3f}"
    )

    print(
        f"Interaction BIC : "
        f"{interaction_model.bic:.3f}"
    )

    print("\nINTERACTION EFFECT")

    print(
        f"Coefficient: "
        f"{interaction_coef:.6f}"
    )

    print(
        f"P-value: "
        f"{interaction_pvalue:.6e}"
    )

    print(
        f"95% CI: "
        f"[{interaction_ci.iloc[0]:.6f}, "
        f"{interaction_ci.iloc[1]:.6f}]"
    )

    print(
        "\nEFFECTIVE FEED SENSITIVITY BY WEAR"
    )

    print(
        feed_sensitivity_by_wear
        .round(6)
        .to_string(index=False)
    )

    print(
        "\nRAW WEAR × FEED SUMMARY"
    )

    print(
        raw_interaction_summary
        .round(6)
        .to_string(index=False)
    )


#%% QUESTION 5 — MATERIAL CONDITION SENSITIVITY

def question_5_material_conditions(df):

    columns = [
        "tool_wear",
        "feed_rate_mm_rev",
        "material_hardness_effect",
        "material_machinability_effect",
        "surface_roughness_um",
        "burr_height_mm",
    ]

    analysis_df = (
        df[columns]
        .dropna()
        .copy()
    )

    analysis_df["roughness_defect"] = (
        analysis_df[
            "surface_roughness_um"
        ]
        > ROUGHNESS_USL
    ).astype(int)

    analysis_df["burr_defect"] = (
        analysis_df["burr_height_mm"]
        > BURR_USL
    ).astype(int)

    analysis_df["hardness_group"] = (
        pd.qcut(
            analysis_df[
                "material_hardness_effect"
            ],
            q=3,
            labels=[
                "Low",
                "Medium",
                "High",
            ],
            duplicates="drop",
        )
    )

    analysis_df[
        "machinability_group"
    ] = pd.qcut(
        analysis_df[
            "material_machinability_effect"
        ],
        q=3,
        labels=[
            "Low",
            "Medium",
            "High",
        ],
        duplicates="drop",
    )

    # Material summaries
    hardness_summary = (
        analysis_df
        .groupby(
            "hardness_group",
            observed=True,
        )
        .agg(
            parts=("tool_wear", "size"),
            mean_hardness=(
                "material_hardness_effect",
                "mean",
            ),
            mean_tool_wear=(
                "tool_wear",
                "mean",
            ),
            mean_feed=(
                "feed_rate_mm_rev",
                "mean",
            ),
            mean_burr=(
                "burr_height_mm",
                "mean",
            ),
            burr_defects=(
                "burr_defect",
                "sum",
            ),
            mean_roughness=(
                "surface_roughness_um",
                "mean",
            ),
            roughness_defects=(
                "roughness_defect",
                "sum",
            ),
        )
        .reset_index()
    )

    hardness_summary[
        "burr_defect_rate_pct"
    ] = (
        100
        * hardness_summary[
            "burr_defects"
        ]
        / hardness_summary["parts"]
    )

    hardness_summary[
        "roughness_defect_rate_pct"
    ] = (
        100
        * hardness_summary[
            "roughness_defects"
        ]
        / hardness_summary["parts"]
    )

    mach_summary = (
        analysis_df
        .groupby(
            "machinability_group",
            observed=True,
        )
        .agg(
            parts=("tool_wear", "size"),
            mean_machinability=(
                "material_machinability_effect",
                "mean",
            ),
            mean_tool_wear=(
                "tool_wear",
                "mean",
            ),
            mean_feed=(
                "feed_rate_mm_rev",
                "mean",
            ),
            mean_roughness=(
                "surface_roughness_um",
                "mean",
            ),
            roughness_defects=(
                "roughness_defect",
                "sum",
            ),
            mean_burr=(
                "burr_height_mm",
                "mean",
            ),
            burr_defects=(
                "burr_defect",
                "sum",
            ),
        )
        .reset_index()
    )

    mach_summary[
        "roughness_defect_rate_pct"
    ] = (
        100
        * mach_summary[
            "roughness_defects"
        ]
        / mach_summary["parts"]
    )

    mach_summary[
        "burr_defect_rate_pct"
    ] = (
        100
        * mach_summary[
            "burr_defects"
        ]
        / mach_summary["parts"]
    )

    # Burr / hardness models
    burr_base = smf.ols(
        """
        burr_height_mm
        ~ tool_wear
        + feed_rate_mm_rev
        + material_hardness_effect
        """,
        data=analysis_df,
    ).fit()

    burr_interaction = smf.ols(
        """
        burr_height_mm
        ~ tool_wear
        + feed_rate_mm_rev
        + material_hardness_effect
        + tool_wear:material_hardness_effect
        """,
        data=analysis_df,
    ).fit()

    # Roughness / machinability models
    roughness_base = smf.ols(
        """
        surface_roughness_um
        ~ tool_wear
        + material_machinability_effect
        """,
        data=analysis_df,
    ).fit()

    roughness_interaction = smf.ols(
        """
        surface_roughness_um
        ~ tool_wear
        + material_machinability_effect
        + tool_wear:material_machinability_effect
        """,
        data=analysis_df,
    ).fit()

    hardness_levels = {
        "Low":
            analysis_df[
                "material_hardness_effect"
            ].quantile(0.10),

        "Median":
            analysis_df[
                "material_hardness_effect"
            ].quantile(0.50),

        "High":
            analysis_df[
                "material_hardness_effect"
            ].quantile(0.90),
    }

    machinability_levels = {
        "Low":
            analysis_df[
                "material_machinability_effect"
            ].quantile(0.10),

        "Median":
            analysis_df[
                "material_machinability_effect"
            ].quantile(0.50),

        "High":
            analysis_df[
                "material_machinability_effect"
            ].quantile(0.90),
    }

    wear_levels = [
        0.50,
        0.70,
        0.75,
        0.775,
        0.80,
        0.90,
    ]

    feed_control_level = (
        analysis_df[
            "feed_rate_mm_rev"
        ].median()
    )

    # Burr predictions
    burr_rows = []

    for wear in wear_levels:
        for name, value in (
            hardness_levels.items()
        ):

            burr_rows.append(
                {
                    "tool_wear": wear,
                    "feed_rate_mm_rev":
                        feed_control_level,
                    "material_hardness_effect":
                        value,
                    "hardness_level":
                        name,
                }
            )

    burr_prediction_df = pd.DataFrame(
        burr_rows
    )

    burr_predictions = (
        burr_interaction
        .get_prediction(
            burr_prediction_df[
                [
                    "tool_wear",
                    "feed_rate_mm_rev",
                    "material_hardness_effect",
                ]
            ]
        )
        .summary_frame(alpha=0.05)
    )

    burr_prediction_df[
        "predicted_burr_mm"
    ] = burr_predictions[
        "mean"
    ].values

    # Roughness predictions
    roughness_rows = []

    for wear in wear_levels:
        for name, value in (
            machinability_levels.items()
        ):

            roughness_rows.append(
                {
                    "tool_wear": wear,
                    "material_machinability_effect":
                        value,
                    "machinability_level":
                        name,
                }
            )

    rough_prediction_df = pd.DataFrame(
        roughness_rows
    )

    rough_predictions = (
        roughness_interaction
        .get_prediction(
            rough_prediction_df[
                [
                    "tool_wear",
                    "material_machinability_effect",
                ]
            ]
        )
        .summary_frame(alpha=0.05)
    )

    rough_prediction_df[
        "predicted_roughness_um"
    ] = rough_predictions[
        "mean"
    ].values

    # Material-specific risk below candidate thresholds
    risk_rows = []

    for threshold in [
        0.70,
        0.75,
        0.775,
        0.80,
    ]:

        below = analysis_df[
            analysis_df["tool_wear"]
            < threshold
        ]

        for group_name, group_df in (
            below.groupby(
                "hardness_group",
                observed=True,
            )
        ):

            risk_rows.append(
                {
                    "material_dimension":
                        "hardness",
                    "material_group":
                        str(group_name),
                    "wear_threshold":
                        threshold,
                    "parts":
                        len(group_df),
                    "roughness_defect_rate_pct":
                        100
                        * group_df[
                            "roughness_defect"
                        ].mean(),
                    "burr_defect_rate_pct":
                        100
                        * group_df[
                            "burr_defect"
                        ].mean(),
                }
            )

        for group_name, group_df in (
            below.groupby(
                "machinability_group",
                observed=True,
            )
        ):

            risk_rows.append(
                {
                    "material_dimension":
                        "machinability",
                    "material_group":
                        str(group_name),
                    "wear_threshold":
                        threshold,
                    "parts":
                        len(group_df),
                    "roughness_defect_rate_pct":
                        100
                        * group_df[
                            "roughness_defect"
                        ].mean(),
                    "burr_defect_rate_pct":
                        100
                        * group_df[
                            "burr_defect"
                        ].mean(),
                }
            )

    material_threshold_risk = (
        pd.DataFrame(risk_rows)
    )

    # Burr by hardness group
    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.bar(
        hardness_summary[
            "hardness_group"
        ].astype(str),
        hardness_summary[
            "mean_burr"
        ],
    )

    ax.set_xlabel(
        "Material Hardness Group"
    )
    ax.set_ylabel(
        "Mean Burr Height (mm)"
    )
    ax.set_title(
        "Mean Burr Height by Material Hardness Group"
    )

    save_figure(
        fig,
        "q5_burr_by_hardness_group.png",
    )

    # Roughness by machinability
    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.bar(
        mach_summary[
            "machinability_group"
        ].astype(str),
        mach_summary[
            "mean_roughness"
        ],
    )

    ax.set_xlabel(
        "Material Machinability Group"
    )
    ax.set_ylabel(
        "Mean Surface Roughness (µm)"
    )
    ax.set_title(
        "Mean Surface Roughness by Material Machinability Group"
    )

    save_figure(
        fig,
        "q5_roughness_by_machinability_group.png",
    )

    # Burr wear × hardness
    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for name in hardness_levels:

        subset = burr_prediction_df[
            burr_prediction_df[
                "hardness_level"
            ] == name
        ]

        ax.plot(
            subset["tool_wear"],
            subset["predicted_burr_mm"],
            marker="o",
            label=name,
        )

    ax.axhline(
        BURR_USL,
        linestyle="--",
        linewidth=2,
        label=f"USL = {BURR_USL:.3f} mm",
    )

    ax.set_xlabel("Tool Wear")
    ax.set_ylabel(
        "Predicted Burr Height (mm)"
    )
    ax.set_title(
        "Predicted Burr Across Tool Wear "
        "and Material Hardness"
    )
    ax.legend()

    save_figure(
        fig,
        "q5_burr_wear_hardness.png",
    )

    # Roughness wear × machinability
    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for name in machinability_levels:

        subset = rough_prediction_df[
            rough_prediction_df[
                "machinability_level"
            ] == name
        ]

        ax.plot(
            subset["tool_wear"],
            subset[
                "predicted_roughness_um"
            ],
            marker="o",
            label=name,
        )

    ax.axhline(
        ROUGHNESS_USL,
        linestyle="--",
        linewidth=2,
        label=f"USL = {ROUGHNESS_USL:.2f} µm",
    )

    ax.set_xlabel("Tool Wear")
    ax.set_ylabel(
        "Predicted Surface Roughness (µm)"
    )
    ax.set_title(
        "Predicted Surface Roughness Across "
        "Tool Wear and Material Machinability"
    )
    ax.legend()

    save_figure(
        fig,
        "q5_roughness_wear_machinability.png",
    )

    # Exports
    hardness_summary.to_csv(
        OUTPUT_DIR
        / "sensitivity_q5_hardness_summary.csv",
        index=False,
    )

    mach_summary.to_csv(
        OUTPUT_DIR
        / "sensitivity_q5_machinability_summary.csv",
        index=False,
    )

    burr_prediction_df.to_csv(
        OUTPUT_DIR
        / "sensitivity_q5_burr_hardness_predictions.csv",
        index=False,
    )

    rough_prediction_df.to_csv(
        OUTPUT_DIR
        / "sensitivity_q5_roughness_machinability_predictions.csv",
        index=False,
    )

    material_threshold_risk.to_csv(
        OUTPUT_DIR
        / "sensitivity_q5_material_threshold_risk.csv",
        index=False,
    )

    print(
        "\nQUESTION 5 — MATERIAL CONDITION SENSITIVITY"
    )

    print(
        "\nQUALITY BY MATERIAL HARDNESS GROUP"
    )

    print(
        hardness_summary
        .round(6)
        .to_string(index=False)
    )

    print(
        "\nQUALITY BY MATERIAL MACHINABILITY GROUP"
    )

    print(
        mach_summary
        .round(6)
        .to_string(index=False)
    )

    print(
        "\nBURR MODEL — HARDNESS MAIN EFFECT"
    )
    print(burr_base.summary())

    print(
        "\nBURR MODEL — WEAR × HARDNESS"
    )
    print(burr_interaction.summary())

    print(
        "\nROUGHNESS MODEL — MACHINABILITY MAIN EFFECT"
    )
    print(roughness_base.summary())

    print(
        "\nROUGHNESS MODEL — WEAR × MACHINABILITY"
    )
    print(roughness_interaction.summary())

    print(
        "\nMATERIAL-SPECIFIC DEFECT RISK BELOW WEAR THRESHOLDS"
    )

    print(
        material_threshold_risk
        .round(6)
        .to_string(index=False)
    )


#%% MAIN ANALYSIS

def main():

    manufacturing_data = pd.read_csv(
        DATA_PATH
    )

    question_1_tool_wear_sensitivity(
        manufacturing_data
    )

    question_2_replacement_threshold(
        manufacturing_data
    )

    question_3_feed_rate_sensitivity(
        manufacturing_data
    )

    question_4_wear_feed_interaction(
        manufacturing_data
    )

    question_5_material_conditions(
        manufacturing_data
    )


#%% RUN

if __name__ == "__main__":
    main()