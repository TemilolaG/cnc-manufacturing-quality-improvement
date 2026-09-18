#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 15:17:16 2026

@author: temilolagbadamosi-adeniyi
"""

"""
CNC Manufacturing Process Quality Improvement
Bore Diameter Root Cause Analysis
"""

#%% IMPORTS

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from statsmodels.stats.anova import anova_lm


#%% PROJECT PATHS

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

PROJECT_ROOT = SCRIPT_DIR.parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "CNC"
    / "manufacturing_quality_improvement_data.csv"
)

OUTPUT_DIR = SCRIPT_DIR / "outputs"
FIGURE_DIR = SCRIPT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


#%% BORE DATA PREPARATION

def prepare_bore_data(df):

    data = df.dropna(
        subset=[
            "bore_diameter_mm",
            "bore_lsl_mm",
            "bore_usl_mm",
        ]
    ).copy()

    data["bore_spec_midpoint_mm"] = (
        data["bore_lsl_mm"]
        + data["bore_usl_mm"]
    ) / 2

    data["bore_deviation_mm"] = (
        data["bore_diameter_mm"]
        - data["bore_spec_midpoint_mm"]
    )

    data["undersize"] = (
        data["bore_diameter_mm"]
        < data["bore_lsl_mm"]
    )

    data["oversize"] = (
        data["bore_diameter_mm"]
        > data["bore_usl_mm"]
    )

    return data


#%% BORE DIAMETER BASELINE

def summarize_bore_performance(df):

    data = prepare_bore_data(df)

    undersize = int(data["undersize"].sum())
    oversize = int(data["oversize"].sum())
    total_defects = undersize + oversize

    return pd.DataFrame(
        {
            "Parts Evaluated": [len(data)],
            "Bore LSL (mm)": [
                data["bore_lsl_mm"].iloc[0]
            ],
            "Bore USL (mm)": [
                data["bore_usl_mm"].iloc[0]
            ],
            "Specification Midpoint (mm)": [
                data["bore_spec_midpoint_mm"].iloc[0]
            ],
            "Mean Bore Diameter (mm)": [
                data["bore_diameter_mm"].mean()
            ],
            "Median Bore Diameter (mm)": [
                data["bore_diameter_mm"].median()
            ],
            "Bore SD (mm)": [
                data["bore_diameter_mm"].std()
            ],
            "Mean Deviation (mm)": [
                data["bore_deviation_mm"].mean()
            ],
            "Undersize Defects": [undersize],
            "Oversize Defects": [oversize],
            "Total Defects": [total_defects],
            "Defect Rate (%)": [
                100 * total_defects / len(data)
            ],
        }
    )


#%% BORE DISTRIBUTION FIGURE

def plot_bore_distribution(df):

    data = prepare_bore_data(df)

    target = data[
        "bore_spec_midpoint_mm"
    ].median()

    lsl = data[
        "bore_lsl_mm"
    ].median()

    usl = data[
        "bore_usl_mm"
    ].median()

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.hist(
        data["bore_diameter_mm"],
        bins=30,
        edgecolor="black",
        alpha=0.7,
    )

    ax.axvline(
        target,
        linestyle="--",
        linewidth=2,
        label=f"Target = {target:.3f}",
    )

    ax.axvline(
        lsl,
        linestyle="--",
        linewidth=2,
        label=f"LSL = {lsl:.3f}",
    )

    ax.axvline(
        usl,
        linestyle="--",
        linewidth=2,
        label=f"USL = {usl:.3f}",
    )

    ax.set_xlabel(
        "Bore Diameter (mm)"
    )

    ax.set_ylabel(
        "Number of Parts"
    )

    ax.set_title(
        "Distribution of Bore Diameter"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "bore_diameter_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)


#%% CATEGORICAL VARIANCE SCREENING

def between_group_share(
    df,
    factor,
    target="bore_deviation_mm",
):

    data = (
        df[[factor, target]]
        .dropna()
    )

    grand_mean = data[target].mean()
    total_var = data[target].var(ddof=0)

    group_stats = (
        data.groupby(factor)[target]
        .agg(["mean", "count"])
    )

    between_var = (
        (
            group_stats["count"]
            * (
                group_stats["mean"]
                - grand_mean
            ) ** 2
        ).sum()
        / len(data)
    )

    within_var = (
        total_var - between_var
    )

    return {
        "Factor": factor,
        "Groups":
            data[factor].nunique(),
        "Between Variance":
            between_var,
        "Within Variance":
            within_var,
        "Between Share (%)":
            100 * between_var / total_var,
        "Within Share (%)":
            100 * within_var / total_var,
    }


def categorical_variance_screen(df):

    data = prepare_bore_data(df)

    factors = [
        "machine_id",
        "material_lot",
        "operator_id",
        "shift_name",
    ]

    return (
        pd.DataFrame(
            [
                between_group_share(
                    data,
                    factor,
                )
                for factor in factors
            ]
        )
        .sort_values(
            "Between Share (%)",
            ascending=False,
        )
        .reset_index(drop=True)
    )


#%% BORE PERFORMANCE BY MACHINE

def summarize_bore_by_machine(df):

    data = prepare_bore_data(df)

    summary = (
        data.groupby("machine_id")
        .agg(
            Parts=(
                "bore_diameter_mm",
                "count",
            ),
            Mean_Bore_mm=(
                "bore_diameter_mm",
                "mean",
            ),
            Mean_Deviation_mm=(
                "bore_deviation_mm",
                "mean",
            ),
            Bore_SD_mm=(
                "bore_diameter_mm",
                "std",
            ),
            Undersize_Defects=(
                "undersize",
                "sum",
            ),
            Oversize_Defects=(
                "oversize",
                "sum",
            ),
        )
        .reset_index()
    )

    summary["Total_Defects"] = (
        summary["Undersize_Defects"]
        + summary["Oversize_Defects"]
    )

    summary["Defect_Rate (%)"] = (
        100
        * summary["Total_Defects"]
        / summary["Parts"]
    )

    return (
        summary
        .sort_values("Mean_Deviation_mm")
        .reset_index(drop=True)
    )


#%% BORE BY MACHINE FIGURE

def plot_bore_by_machine(df):

    data = prepare_bore_data(df)

    machines = sorted(
        data["machine_id"].unique()
    )

    plot_data = [
        data.loc[
            data["machine_id"] == machine,
            "bore_deviation_mm",
        ]
        for machine in machines
    ]

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.boxplot(
        plot_data,
        tick_labels=machines,
        showfliers=False,
    )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1.5,
        label="Specification Midpoint",
    )

    ax.set_xlabel("Machine")

    ax.set_ylabel(
        "Bore Deviation from Midpoint (mm)"
    )

    ax.set_title(
        "Bore Diameter Centering by CNC Machine"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "bore_deviation_by_machine.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)


#%% MACHINE EFFECT SIGNIFICANCE

def test_bore_machine_effect(df):

    data = prepare_bore_data(df)

    model = smf.ols(
        "bore_deviation_mm ~ C(machine_id)",
        data=data,
    ).fit()

    anova = sm.stats.anova_lm(
        model,
        typ=2,
    )

    residual_ss = (
        anova.loc["Residual", "sum_sq"]
    )

    machine_ss = (
        anova.loc[
            "C(machine_id)",
            "sum_sq",
        ]
    )

    partial_eta_sq = (
        machine_ss
        / (machine_ss + residual_ss)
    )

    results = pd.DataFrame(
        {
            "R2": [model.rsquared],
            "Partial Eta Squared": [
                partial_eta_sq
            ],
        }
    )

    return model, anova, results


#%% TOOL WEAR WITHIN MACHINE

def analyze_bore_wear_by_machine(df):

    data = prepare_bore_data(df)

    data = data.dropna(
        subset=["tool_wear"]
    )

    results = []

    for machine, group in (
        data.groupby("machine_id")
    ):

        correlation = (
            group["tool_wear"]
            .corr(
                group["bore_deviation_mm"]
            )
        )

        model = smf.ols(
            """
            bore_deviation_mm
            ~ tool_wear
            """,
            data=group,
        ).fit()

        results.append(
            {
                "Machine": machine,
                "Parts": len(group),
                "Wear Correlation":
                    correlation,
                "Wear Coefficient":
                    model.params["tool_wear"],
                "Wear p-value":
                    model.pvalues["tool_wear"],
                "R2":
                    model.rsquared,
            }
        )

    return pd.DataFrame(results)


#%% TOOL WEAR FIGURE

def plot_bore_wear_by_machine(df):

    data = (
        prepare_bore_data(df)
        .dropna(
            subset=["tool_wear"]
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for machine, group in (
        data.groupby("machine_id")
    ):

        ax.scatter(
            group["tool_wear"],
            group["bore_deviation_mm"],
            alpha=0.25,
            s=12,
        )

        coefficients = np.polyfit(
            group["tool_wear"],
            group["bore_deviation_mm"],
            1,
        )

        x_line = np.linspace(
            group["tool_wear"].min(),
            group["tool_wear"].max(),
            100,
        )

        y_line = (
            coefficients[0] * x_line
            + coefficients[1]
        )

        ax.plot(
            x_line,
            y_line,
            linewidth=2,
            label=machine,
        )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel("Tool Wear")

    ax.set_ylabel(
        "Bore Deviation from Midpoint (mm)"
    )

    ax.set_title(
        "Tool Wear vs Bore Deviation by Machine"
    )

    ax.legend(
        title="Machine"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "bore_tool_wear_by_machine.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)


#%% MACHINE × TOOL WEAR

def test_machine_wear_bore(df):

    data = (
        prepare_bore_data(df)
        .dropna(
            subset=["tool_wear"]
        )
    )

    machine_model = smf.ols(
        """
        bore_deviation_mm
        ~ C(machine_id)
        """,
        data=data,
    ).fit()

    additive_model = smf.ols(
        """
        bore_deviation_mm
        ~ C(machine_id)
        + tool_wear
        """,
        data=data,
    ).fit()

    interaction_model = smf.ols(
        """
        bore_deviation_mm
        ~ C(machine_id)
        * tool_wear
        """,
        data=data,
    ).fit()

    wear_test = anova_lm(
        machine_model,
        additive_model,
    )

    interaction_test = anova_lm(
        additive_model,
        interaction_model,
    )

    results = pd.DataFrame(
        {
            "Model": [
                "Machine Only",
                "Machine + Wear",
                "Machine × Wear",
            ],
            "R2": [
                machine_model.rsquared,
                additive_model.rsquared,
                interaction_model.rsquared,
            ],
        }
    )

    return (
        machine_model,
        additive_model,
        interaction_model,
        wear_test,
        interaction_test,
        results,
    )


#%% PROCESS VARIABLE BLOCK

PROCESS_VARIABLES = [
    "machine_deterioration",
    "spindle_speed_rpm",
    "feed_rate_mm_rev",
    "coolant_flow_l_min",
    "cutting_force_n",
    "spindle_load_pct",
    "temperature_c",
    "vibration_mm_s",
]


def test_bore_machine_process_block(df):

    data = prepare_bore_data(df)

    required = (
        ["machine_id"]
        + PROCESS_VARIABLES
    )

    data = (
        data.dropna(
            subset=required
        )
    )

    machine_model = smf.ols(
        """
        bore_deviation_mm
        ~ C(machine_id)
        """,
        data=data,
    ).fit()

    formula = (
        "bore_deviation_mm "
        "~ C(machine_id) + "
        + " + ".join(
            PROCESS_VARIABLES
        )
    )

    expanded_model = smf.ols(
        formula,
        data=data,
    ).fit()

    block_test = anova_lm(
        machine_model,
        expanded_model,
    )

    coefficient_table = pd.DataFrame(
        {
            "Coefficient":
                expanded_model.params[
                    PROCESS_VARIABLES
                ],
            "p-value":
                expanded_model.pvalues[
                    PROCESS_VARIABLES
                ],
        }
    )

    results = pd.DataFrame(
        {
            "R2 Machine Only": [
                machine_model.rsquared
            ],
            "R2 Expanded": [
                expanded_model.rsquared
            ],
            "Incremental R2": [
                expanded_model.rsquared
                - machine_model.rsquared
            ],
        }
    )

    return (
        machine_model,
        expanded_model,
        block_test,
        coefficient_table,
        results,
    )


#%% CONTINUOUS VARIABLE SCREENING

CONTINUOUS_VARIABLES = [
    "tool_wear",
    "machine_deterioration",
    "material_hardness_effect",
    "material_machinability_effect",
    "spindle_speed_rpm",
    "feed_rate_mm_rev",
    "coolant_flow_l_min",
    "cutting_force_n",
    "spindle_load_pct",
    "temperature_c",
    "vibration_mm_s",
]


def screen_bore_continuous_variables(df):

    data = prepare_bore_data(df)

    correlations = (
        data[
            CONTINUOUS_VARIABLES
            + ["bore_deviation_mm"]
        ]
        .corr()["bore_deviation_mm"]
        .drop("bore_deviation_mm")
        .sort_values(
            key=abs,
            ascending=False,
        )
    )

    return (
        correlations
        .rename(
            "Correlation with Bore Deviation"
        )
        .reset_index()
        .rename(
            columns={
                "index":
                    "Process Variable"
            }
        )
    )


#%% MACHINE DETERIORATION WITHIN MACHINE

def analyze_bore_deterioration_by_machine(df):

    data = (
        prepare_bore_data(df)
        .dropna(
            subset=[
                "machine_deterioration"
            ]
        )
    )

    results = []

    for machine, group in (
        data.groupby("machine_id")
    ):

        correlation = (
            group[
                "machine_deterioration"
            ]
            .corr(
                group["bore_deviation_mm"]
            )
        )

        model = smf.ols(
            """
            bore_deviation_mm
            ~ machine_deterioration
            """,
            data=group,
        ).fit()

        results.append(
            {
                "Machine": machine,
                "Parts": len(group),
                "Deterioration Correlation":
                    correlation,
                "Deterioration Coefficient":
                    model.params[
                        "machine_deterioration"
                    ],
                "p-value":
                    model.pvalues[
                        "machine_deterioration"
                    ],
                "R2":
                    model.rsquared,
            }
        )

    return pd.DataFrame(results)


#%% TOOL WEAR vs MACHINE DETERIORATION

def analyze_wear_deterioration_relationship(
    df,
):

    data = (
        df[
            [
                "machine_id",
                "tool_wear",
                "machine_deterioration",
            ]
        ]
        .dropna()
        .copy()
    )

    overall_correlation = (
        data["tool_wear"]
        .corr(
            data["machine_deterioration"]
        )
    )

    results = []

    for machine, group in (
        data.groupby("machine_id")
    ):

        correlation = (
            group["tool_wear"]
            .corr(
                group[
                    "machine_deterioration"
                ]
            )
        )

        results.append(
            {
                "Machine": machine,
                "Parts": len(group),
                "Wear-Deterioration Correlation":
                    correlation,
            }
        )

    return (
        overall_correlation,
        pd.DataFrame(results),
    )


#%% REDUCED BORE MODEL

def fit_bore_reduced_model(df):

    data = (
        prepare_bore_data(df)
        .dropna(
            subset=[
                "machine_id",
                "tool_wear",
                "machine_deterioration",
            ]
        )
    )

    model = smf.ols(
        """
        bore_deviation_mm ~
        C(machine_id)
        + tool_wear
        + machine_deterioration
        """,
        data=data,
    ).fit()

    return model


#%% MATERIAL LOT CONTRIBUTION

def test_bore_material_lot_increment(df):

    data = (
        prepare_bore_data(df)
        .dropna(
            subset=[
                "machine_id",
                "material_lot",
                "tool_wear",
                "machine_deterioration",
            ]
        )
    )

    base_model = smf.ols(
        """
        bore_deviation_mm ~
        C(machine_id)
        + tool_wear
        + machine_deterioration
        """,
        data=data,
    ).fit()

    lot_model = smf.ols(
        """
        bore_deviation_mm ~
        C(machine_id)
        + tool_wear
        + machine_deterioration
        + C(material_lot)
        """,
        data=data,
    ).fit()

    test = anova_lm(
        base_model,
        lot_model,
    )

    results = pd.DataFrame(
        {
            "R2 Without Material Lot": [
                base_model.rsquared
            ],
            "R2 With Material Lot": [
                lot_model.rsquared
            ],
            "Incremental R2": [
                lot_model.rsquared
                - base_model.rsquared
            ],
        }
    )

    return (
        base_model,
        lot_model,
        test,
        results,
    )


#%% OPERATOR CONTRIBUTION

def test_bore_operator_increment(df):

    data = (
        prepare_bore_data(df)
        .dropna(
            subset=[
                "machine_id",
                "material_lot",
                "operator_id",
                "tool_wear",
                "machine_deterioration",
            ]
        )
    )

    base_model = smf.ols(
        """
        bore_deviation_mm ~
        C(machine_id)
        + C(material_lot)
        + tool_wear
        + machine_deterioration
        """,
        data=data,
    ).fit()

    operator_model = smf.ols(
        """
        bore_deviation_mm ~
        C(machine_id)
        + C(material_lot)
        + C(operator_id)
        + tool_wear
        + machine_deterioration
        """,
        data=data,
    ).fit()

    test = anova_lm(
        base_model,
        operator_model,
    )

    results = pd.DataFrame(
        {
            "R2 Without Operator": [
                base_model.rsquared
            ],
            "R2 With Operator": [
                operator_model.rsquared
            ],
            "Incremental R2": [
                operator_model.rsquared
                - base_model.rsquared
            ],
        }
    )

    return (
        base_model,
        operator_model,
        test,
        results,
    )


#%% SHIFT CONTRIBUTION

def test_bore_shift_increment(df):

    data = (
        prepare_bore_data(df)
        .dropna(
            subset=[
                "machine_id",
                "material_lot",
                "operator_id",
                "shift_name",
                "tool_wear",
                "machine_deterioration",
            ]
        )
    )

    base_model = smf.ols(
        """
        bore_deviation_mm ~
        C(machine_id)
        + C(material_lot)
        + C(operator_id)
        + tool_wear
        + machine_deterioration
        """,
        data=data,
    ).fit()

    shift_model = smf.ols(
        """
        bore_deviation_mm ~
        C(machine_id)
        + C(material_lot)
        + C(operator_id)
        + C(shift_name)
        + tool_wear
        + machine_deterioration
        """,
        data=data,
    ).fit()

    test = anova_lm(
        base_model,
        shift_model,
    )

    results = pd.DataFrame(
        {
            "R2 Without Shift": [
                base_model.rsquared
            ],
            "R2 With Shift": [
                shift_model.rsquared
            ],
            "Incremental R2": [
                shift_model.rsquared
                - base_model.rsquared
            ],
        }
    )

    return (
        base_model,
        shift_model,
        test,
        results,
    )


#%% MATERIAL MECHANISM TEST

def test_bore_material_mechanism(df):

    data = (
        prepare_bore_data(df)
        .dropna(
            subset=[
                "machine_id",
                "material_lot",
                "tool_wear",
                "machine_deterioration",
                "material_hardness_effect",
                "material_machinability_effect",
            ]
        )
    )

    base_model = smf.ols(
        """
        bore_deviation_mm ~
        C(machine_id)
        + tool_wear
        + machine_deterioration
        """,
        data=data,
    ).fit()

    property_model = smf.ols(
        """
        bore_deviation_mm ~
        C(machine_id)
        + tool_wear
        + machine_deterioration
        + material_hardness_effect
        + material_machinability_effect
        """,
        data=data,
    ).fit()

    lot_model = smf.ols(
        """
        bore_deviation_mm ~
        C(machine_id)
        + tool_wear
        + machine_deterioration
        + C(material_lot)
        """,
        data=data,
    ).fit()

    full_model = smf.ols(
        """
        bore_deviation_mm ~
        C(machine_id)
        + tool_wear
        + machine_deterioration
        + material_hardness_effect
        + material_machinability_effect
        + C(material_lot)
        """,
        data=data,
    ).fit()

    property_test = anova_lm(
        base_model,
        property_model,
    )

    lot_after_properties_test = (
        anova_lm(
            property_model,
            full_model,
        )
    )

    results = pd.DataFrame(
        {
            "Model": [
                "Baseline",
                "+ Material Properties",
                "+ Material Lot",
                "+ Properties + Lot",
            ],
            "R2": [
                base_model.rsquared,
                property_model.rsquared,
                lot_model.rsquared,
                full_model.rsquared,
            ],
        }
    )

    return (
        base_model,
        property_model,
        lot_model,
        full_model,
        property_test,
        lot_after_properties_test,
        results,
    )


#%% STANDARDIZED BORE MODEL

def fit_standardized_bore_model(df):

    data = (
        prepare_bore_data(df)
        .dropna(
            subset=[
                "machine_id",
                "material_lot",
                "tool_wear",
                "machine_deterioration",
            ]
        )
        .copy()
    )

    data["tool_wear_z"] = (
        data["tool_wear"]
        - data["tool_wear"].mean()
    ) / data["tool_wear"].std()

    data["machine_deterioration_z"] = (
        data["machine_deterioration"]
        - data[
            "machine_deterioration"
        ].mean()
    ) / data[
        "machine_deterioration"
    ].std()

    model = smf.ols(
        """
        bore_deviation_mm ~
        C(machine_id)
        + C(material_lot)
        + machine_deterioration_z
        + C(machine_id) * tool_wear_z
        """,
        data=data,
    ).fit()

    return model, data


#%% MACHINE × CATEGORICAL INTERACTION

def bore_machine_interaction_test(
    df,
    factor,
):

    data = (
        prepare_bore_data(df)
        .dropna(
            subset=[
                "machine_id",
                factor,
            ]
        )
    )

    counts = pd.crosstab(
        data["machine_id"],
        data[factor],
    )

    means = pd.pivot_table(
        data,
        values="bore_deviation_mm",
        index="machine_id",
        columns=factor,
        aggfunc="mean",
    )

    additive_model = smf.ols(
        (
            "bore_deviation_mm ~ "
            f"C(machine_id) + C({factor})"
        ),
        data=data,
    ).fit()

    interaction_model = smf.ols(
        (
            "bore_deviation_mm ~ "
            f"C(machine_id) * C({factor})"
        ),
        data=data,
    ).fit()

    test = anova_lm(
        additive_model,
        interaction_model,
    )

    results = pd.DataFrame(
        {
            "R2 Additive": [
                additive_model.rsquared
            ],
            "R2 Interaction": [
                interaction_model.rsquared
            ],
            "Incremental R2": [
                interaction_model.rsquared
                - additive_model.rsquared
            ],
        }
    )

    return (
        counts,
        means,
        additive_model,
        interaction_model,
        test,
        results,
    )


#%% MACHINE × CATEGORICAL FIGURE

def plot_bore_interaction(
    means,
    factor,
    filename,
):

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for level in means.columns:

        ax.plot(
            means.index.astype(str),
            means[level],
            marker="o",
            label=str(level),
        )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel("Machine")

    ax.set_ylabel(
        "Mean Bore Deviation (mm)"
    )

    ax.set_title(
        f"Machine × {factor} — Bore Deviation"
    )

    ax.legend(
        title=factor,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)


#%% FINAL HIERARCHICAL INTERACTION MODELS

def fit_hierarchical_bore_models(df):

    data = (
        prepare_bore_data(df)
        .dropna(
            subset=[
                "machine_id",
                "material_lot",
                "operator_id",
                "tool_wear",
                "machine_deterioration",
            ]
        )
        .copy()
    )

    for column in [
        "tool_wear",
        "machine_deterioration",
    ]:

        data[f"{column}_z"] = (
            data[column]
            - data[column].mean()
        ) / data[column].std()

    m1 = smf.ols(
        """
        bore_deviation_mm ~
        C(machine_id)
        + C(material_lot)
        + C(operator_id)
        + tool_wear_z
        + machine_deterioration_z
        """,
        data=data,
    ).fit()

    m2 = smf.ols(
        """
        bore_deviation_mm ~
        C(machine_id)
        + C(material_lot)
        + C(operator_id)
        + machine_deterioration_z
        + C(machine_id) * tool_wear_z
        """,
        data=data,
    ).fit()

    m3 = smf.ols(
        """
        bore_deviation_mm ~
        C(machine_id)
        + C(material_lot)
        + C(operator_id)
        + machine_deterioration_z
        + C(machine_id) * tool_wear_z
        + C(machine_id):C(material_lot)
        """,
        data=data,
    ).fit()

    m4 = smf.ols(
        """
        bore_deviation_mm ~
        C(machine_id)
        + C(material_lot)
        + C(operator_id)
        + machine_deterioration_z
        + C(machine_id) * tool_wear_z
        + C(machine_id):C(material_lot)
        + C(machine_id):C(operator_id)
        """,
        data=data,
    ).fit()

    models = [
        ("Additive baseline", m1),
        ("+ Machine × Wear", m2),
        ("+ Machine × Material Lot", m3),
        ("+ Machine × Operator", m4),
    ]

    rows = []

    previous_r2 = None

    for name, model in models:

        delta_r2 = (
            np.nan
            if previous_r2 is None
            else model.rsquared
            - previous_r2
        )

        rows.append(
            {
                "Model": name,
                "R2": model.rsquared,
                "Adjusted R2":
                    model.rsquared_adj,
                "Incremental R2":
                    delta_r2,
                "AIC": model.aic,
                "BIC": model.bic,
                "Parameters":
                    int(model.df_model + 1),
            }
        )

        previous_r2 = model.rsquared

    fit_table = pd.DataFrame(rows)

    comparisons = [
        ("Machine × Wear", m1, m2),
        (
            "Machine × Material Lot",
            m2,
            m3,
        ),
        (
            "Machine × Operator",
            m3,
            m4,
        ),
    ]

    test_rows = []

    for label, reduced, full in comparisons:

        f_stat, p_value, df_diff = (
            full.compare_f_test(reduced)
        )

        test_rows.append(
            {
                "Interaction": label,
                "Additional df":
                    int(df_diff),
                "Partial F":
                    f_stat,
                "p-value":
                    p_value,
                "Incremental R2":
                    full.rsquared
                    - reduced.rsquared,
            }
        )

    test_table = pd.DataFrame(
        test_rows
    )

    return (
        models,
        fit_table,
        test_table,
    )


#%% MAIN ANALYSIS

def main():

    manufacturing_data = pd.read_csv(
        DATA_PATH
    )

    # Baseline
    baseline = summarize_bore_performance(
        manufacturing_data
    )

    print(
        "\nBORE DIAMETER PERFORMANCE BASELINE"
    )

    print(
        baseline
        .round(5)
        .to_string(index=False)
    )

    plot_bore_distribution(
        manufacturing_data
    )

    # Categorical screening
    categorical_screen = (
        categorical_variance_screen(
            manufacturing_data
        )
    )

    print(
        "\nCATEGORICAL VARIANCE SCREENING — BORE DIAMETER"
    )

    print(
        categorical_screen
        .round(5)
        .to_string(index=False)
    )

    # Machine performance
    machine_summary = (
        summarize_bore_by_machine(
            manufacturing_data
        )
    )

    print(
        "\nBORE DIAMETER PERFORMANCE BY MACHINE"
    )

    print(
        machine_summary
        .round(5)
        .to_string(index=False)
    )

    plot_bore_by_machine(
        manufacturing_data
    )

    # Machine significance
    (
        _,
        machine_anova,
        machine_effect,
    ) = test_bore_machine_effect(
        manufacturing_data
    )

    print(
        "\nMACHINE EFFECT — BORE DIAMETER"
    )

    print(
        machine_anova
        .round(6)
        .to_string()
    )

    print(
        machine_effect
        .round(6)
        .to_string(index=False)
    )

    # Tool wear within machine
    wear_summary = (
        analyze_bore_wear_by_machine(
            manufacturing_data
        )
    )

    print(
        "\nTOOL WEAR vs BORE DEVIATION — WITHIN MACHINE"
    )

    print(
        wear_summary
        .round(6)
        .to_string(index=False)
    )

    plot_bore_wear_by_machine(
        manufacturing_data
    )

    # Machine × wear
    (
        _,
        _,
        _,
        wear_test,
        wear_interaction_test,
        wear_model_results,
    ) = test_machine_wear_bore(
        manufacturing_data
    )

    print(
        "\nMACHINE + TOOL WEAR — BORE DIAMETER"
    )

    print(
        wear_model_results
        .round(6)
        .to_string(index=False)
    )

    print("\nPARTIAL F-TEST — TOOL WEAR")
    print(wear_test.to_string())

    print(
        "\nPARTIAL F-TEST — MACHINE × TOOL WEAR"
    )

    print(
        wear_interaction_test.to_string()
    )

    # Process-variable block
    (
        _,
        _,
        process_test,
        process_coefficients,
        process_results,
    ) = test_bore_machine_process_block(
        manufacturing_data
    )

    print(
        "\nPROCESS VARIABLE BLOCK — BORE DIAMETER"
    )

    print(
        process_results
        .round(6)
        .to_string(index=False)
    )

    print("\nPARTIAL F-TEST")
    print(process_test.to_string())

    print(
        "\nPROCESS VARIABLE COEFFICIENTS"
    )

    print(
        process_coefficients
        .round(6)
        .to_string()
    )

    # Continuous screening
    continuous_screen = (
        screen_bore_continuous_variables(
            manufacturing_data
        )
    )

    print(
        "\nCONTINUOUS VARIABLE SCREENING — BORE DIAMETER"
    )

    print(
        continuous_screen
        .round(4)
        .to_string(index=False)
    )

    # Deterioration within machine
    deterioration_summary = (
        analyze_bore_deterioration_by_machine(
            manufacturing_data
        )
    )

    print(
        "\nMACHINE DETERIORATION vs BORE DEVIATION — WITHIN MACHINE"
    )

    print(
        deterioration_summary
        .round(6)
        .to_string(index=False)
    )

    # Wear vs deterioration
    (
        overall_corr,
        wear_deterioration,
    ) = analyze_wear_deterioration_relationship(
        manufacturing_data
    )

    print(
        "\nTOOL WEAR vs MACHINE DETERIORATION"
    )

    print(
        f"Overall correlation: "
        f"{overall_corr:.4f}"
    )

    print(
        wear_deterioration
        .round(4)
        .to_string(index=False)
    )

    # Reduced model
    reduced_model = (
        fit_bore_reduced_model(
            manufacturing_data
        )
    )

    print(
        "\nBORE REDUCED MODEL — MACHINE + WEAR + DETERIORATION"
    )

    print(
        reduced_model.summary()
    )

    # Material lot
    (
        _,
        _,
        lot_test,
        lot_results,
    ) = test_bore_material_lot_increment(
        manufacturing_data
    )

    print(
        "\nMATERIAL LOT INCREMENTAL CONTRIBUTION — BORE"
    )

    print(
        lot_results
        .round(6)
        .to_string(index=False)
    )

    print(lot_test.to_string())

    # Operator
    (
        _,
        _,
        operator_test,
        operator_results,
    ) = test_bore_operator_increment(
        manufacturing_data
    )

    print(
        "\nOPERATOR INCREMENTAL CONTRIBUTION — BORE"
    )

    print(
        operator_results
        .round(6)
        .to_string(index=False)
    )

    print(operator_test.to_string())

    # Shift
    (
        _,
        _,
        shift_test,
        shift_results,
    ) = test_bore_shift_increment(
        manufacturing_data
    )

    print(
        "\nSHIFT INCREMENTAL CONTRIBUTION — BORE"
    )

    print(
        shift_results
        .round(6)
        .to_string(index=False)
    )

    print(shift_test.to_string())

    # Material mechanism
    (
        _,
        _,
        _,
        _,
        property_test,
        lot_after_properties_test,
        material_results,
    ) = test_bore_material_mechanism(
        manufacturing_data
    )

    print(
        "\nMATERIAL LOT vs MEASURED MATERIAL PROPERTIES — BORE"
    )

    print(
        material_results
        .round(6)
        .to_string(index=False)
    )

    print(
        "\nPARTIAL F-TEST — MATERIAL PROPERTIES"
    )
    print(property_test.to_string())

    print(
        "\nPARTIAL F-TEST — MATERIAL LOT AFTER PROPERTIES"
    )
    print(
        lot_after_properties_test.to_string()
    )

    # Standardized model
    (
        standardized_model,
        _,
    ) = fit_standardized_bore_model(
        manufacturing_data
    )

    print(
        "\nSTANDARDIZED BORE RCA MODEL"
    )

    print(
        standardized_model.summary()
    )

    # Machine × categorical factors
    interaction_factors = [
        (
            "material_lot",
            "Material Lot",
            "bore_machine_material_lot.png",
        ),
        (
            "operator_id",
            "Operator",
            "bore_machine_operator.png",
        ),
        (
            "shift_name",
            "Shift",
            "bore_machine_shift.png",
        ),
    ]

    for (
        factor,
        factor_name,
        filename,
    ) in interaction_factors:

        (
            counts,
            means,
            _,
            _,
            interaction_test,
            interaction_results,
        ) = bore_machine_interaction_test(
            manufacturing_data,
            factor,
        )

        print(
            f"\nMACHINE × {factor_name.upper()} — BORE"
        )

        print("\nCOUNTS")
        print(counts.to_string())

        print("\nMEAN BORE DEVIATION")
        print(
            means
            .round(5)
            .to_string()
        )

        print("\nINTERACTION MODEL")
        print(
            interaction_results
            .round(6)
            .to_string(index=False)
        )

        print("\nPARTIAL F-TEST")
        print(
            interaction_test.to_string()
        )

        plot_bore_interaction(
            means,
            factor_name,
            filename,
        )

    # Final hierarchical model
    (
        _,
        fit_table,
        hierarchical_tests,
    ) = fit_hierarchical_bore_models(
        manufacturing_data
    )

    print(
        "\nFINAL BORE MODEL — HIERARCHICAL INTERACTION TESTING"
    )

    print(
        fit_table
        .round(6)
        .to_string(index=False)
    )

    print(
        "\nPARTIAL F-TESTS"
    )

    print(
        hierarchical_tests
        .round(6)
        .to_string(index=False)
    )


#%% RUN

if __name__ == "__main__":
    main()