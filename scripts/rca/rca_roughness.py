#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 12:02:37 2026

@author: temilolagbadamosi-adeniyi
"""

"""
CNC Manufacturing Process Quality Improvement
Surface Roughness Root Cause Analysis
"""

#%% IMPORTS

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from scipy.stats import zscore
from statsmodels.formula.api import ols
from statsmodels.graphics.factorplots import interaction_plot
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.outliers_influence import variance_inflation_factor


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


#%% PASS / FAIL VALIDATION

def validate_roughness_pass(df):
    """Verify stored roughness pass/fail against the roughness USL."""

    data = df.copy()

    expected_pass = (
        data["surface_roughness_um"]
        <= data["roughness_usl_um"]
    )

    matches = (
        data["roughness_pass"].astype(bool)
        == expected_pass
    )

    return pd.DataFrame(
        {
            "Total Parts": [len(data)],
            "Matching Results": [int(matches.sum())],
            "Mismatching Results": [int((~matches).sum())],
        }
    )


#%% OVERALL ROUGHNESS PERFORMANCE

def summarize_roughness_performance(df):

    total_parts = len(df)

    defects = (
        ~df["roughness_pass"].astype(bool)
    ).sum()

    return pd.DataFrame(
        {
            "Total Parts": [total_parts],
            "Roughness Defects": [int(defects)],
            "Defect Rate (%)": [
                100 * defects / total_parts
            ],
            "Mean Roughness (um)": [
                df["surface_roughness_um"].mean()
            ],
            "Roughness Std Dev (um)": [
                df["surface_roughness_um"].std()
            ],
        }
    )


#%% CATEGORICAL VARIANCE SCREENING

def between_group_share(
    df,
    factor,
    target="surface_roughness_um",
):
    """Estimate target variance associated with a categorical factor."""

    data = df[[factor, target]].dropna()

    grand_mean = data[target].mean()
    total_var = data[target].var(ddof=0)

    group_stats = (
        data.groupby(factor)[target]
        .agg(["mean", "count"])
    )

    between_var = (
        (
            group_stats["count"]
            * (group_stats["mean"] - grand_mean) ** 2
        ).sum()
        / len(data)
    )

    within_var = total_var - between_var

    return {
        "Factor": factor,
        "Groups": data[factor].nunique(),
        "Between Variance": between_var,
        "Within Variance": within_var,
        "Between Share (%)":
            100 * between_var / total_var,
        "Within Share (%)":
            100 * within_var / total_var,
    }


def categorical_variance_screen(df):

    factors = [
        "material_lot",
        "operator_id",
        "machine_id",
        "shift_name",
    ]

    return (
        pd.DataFrame(
            [
                between_group_share(df, factor)
                for factor in factors
            ]
        )
        .sort_values(
            "Between Share (%)",
            ascending=False,
        )
        .reset_index(drop=True)
    )


#%% ROUGHNESS BY CATEGORICAL FACTOR

def analyze_roughness_by_category(
    df,
    category_column,
    category_name,
):

    summary = (
        df.groupby(category_column)
        .agg(
            Parts=("part_number", "count"),
            Roughness_Defects=(
                "roughness_pass",
                lambda x: (~x.astype(bool)).sum(),
            ),
            Mean_Roughness_um=(
                "surface_roughness_um",
                "mean",
            ),
            Median_Roughness_um=(
                "surface_roughness_um",
                "median",
            ),
            Roughness_Std_Dev_um=(
                "surface_roughness_um",
                "std",
            ),
        )
        .reset_index()
        .rename(
            columns={
                category_column: category_name
            }
        )
    )

    summary["Roughness Defect Rate (%)"] = (
        100
        * summary["Roughness_Defects"]
        / summary["Parts"]
    )

    return (
        summary
        .sort_values(
            "Roughness Defect Rate (%)",
            ascending=False,
        )
        .reset_index(drop=True)
    )


#%% MATERIAL LOT INVESTIGATION

def analyze_material_lot_properties(df):

    summary = (
        df.groupby("material_lot")
        .agg(
            Parts=("part_number", "count"),
            Mean_Roughness_um=(
                "surface_roughness_um",
                "mean",
            ),
            Roughness_Defects=(
                "roughness_pass",
                lambda x: (~x.astype(bool)).sum(),
            ),
            Mean_Hardness_Effect=(
                "material_hardness_effect",
                "mean",
            ),
            Mean_Machinability_Effect=(
                "material_machinability_effect",
                "mean",
            ),
        )
        .reset_index()
    )

    summary["Roughness Defect Rate (%)"] = (
        100
        * summary["Roughness_Defects"]
        / summary["Parts"]
    )

    return (
        summary
        .sort_values(
            "Roughness Defect Rate (%)",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def analyze_material_lot_tool_wear(df):

    summary = (
        df.groupby("material_lot")
        .agg(
            Parts=("part_number", "count"),
            Mean_Tool_Wear=("tool_wear", "mean"),
            Median_Tool_Wear=("tool_wear", "median"),
            Max_Tool_Wear=("tool_wear", "max"),
            Mean_Roughness_um=(
                "surface_roughness_um",
                "mean",
            ),
            Roughness_Defects=(
                "roughness_pass",
                lambda x: (~x.astype(bool)).sum(),
            ),
        )
        .reset_index()
    )

    summary["Roughness Defect Rate (%)"] = (
        100
        * summary["Roughness_Defects"]
        / summary["Parts"]
    )

    return (
        summary
        .sort_values(
            "Roughness Defect Rate (%)",
            ascending=False,
        )
        .reset_index(drop=True)
    )


#%% MACHINE × OPERATOR INVESTIGATION

def analyze_machine_operator(df):

    summary = (
        df.groupby(
            ["machine_id", "operator_id"]
        )["surface_roughness_um"]
        .agg(
            Mean="mean",
            Std_Dev="std",
            Parts="count",
        )
        .reset_index()
    )

    mean_matrix = summary.pivot(
        index="operator_id",
        columns="machine_id",
        values="Mean",
    )

    return summary, mean_matrix


def fit_machine_operator_interaction(df):

    model = ols(
        """
        surface_roughness_um
        ~ C(machine_id)
        + C(operator_id)
        + C(machine_id):C(operator_id)
        """,
        data=df,
    ).fit()

    anova = sm.stats.anova_lm(
        model,
        typ=2,
    )

    return model, anova


def plot_machine_operator_interaction(df):

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    interaction_plot(
        x=df["machine_id"],
        trace=df["operator_id"],
        response=df["surface_roughness_um"],
        markers=["o"] * df["operator_id"].nunique(),
        ms=6,
        ax=ax,
    )

    ax.set_xlabel("Machine")
    ax.set_ylabel("Mean Surface Roughness (um)")
    ax.set_title(
        "Machine × Operator Interaction — Surface Roughness"
    )
    ax.grid(alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "roughness_machine_operator_interaction.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)


#%% OPERATOR COMPARISONS WITHIN MACHINE

def operator_comparisons_by_machine(df):

    results = {}

    for machine in sorted(
        df["machine_id"].unique()
    ):

        subset = df[
            df["machine_id"] == machine
        ]

        results[machine] = pairwise_tukeyhsd(
            endog=subset["surface_roughness_um"],
            groups=subset["operator_id"],
            alpha=0.05,
        )

    return results


#%% OPERATOR × MACHINE PROCESS CONDITIONS

def operator_machine_process_profile(df):

    return (
        df.groupby(
            ["machine_id", "operator_id"],
            observed=True,
        )
        .agg(
            Parts=("surface_roughness_um", "size"),
            Mean_Roughness_um=(
                "surface_roughness_um",
                "mean",
            ),
            Mean_Tool_Wear=(
                "tool_wear",
                "mean",
            ),
            Mean_Feed_Rate=(
                "feed_rate_mm_rev",
                "mean",
            ),
            Mean_Coolant_Flow=(
                "coolant_flow_l_min",
                "mean",
            ),
        )
        .reset_index()
    )


#%% CONTINUOUS VARIABLE SCREENING

ROUGHNESS_PROCESS_VARIABLES = [
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


def calculate_roughness_correlations(
    df,
    process_variables,
):

    correlations = (
        df[
            process_variables
            + ["surface_roughness_um"]
        ]
        .corr(method="pearson")
        ["surface_roughness_um"]
        .drop("surface_roughness_um")
    )

    results = pd.DataFrame(
        {
            "Process Variable":
                correlations.index,
            "Correlation with Roughness":
                correlations.values,
        }
    )

    results["Absolute Correlation"] = (
        results[
            "Correlation with Roughness"
        ].abs()
    )

    return (
        results
        .sort_values(
            "Absolute Correlation",
            ascending=False,
        )
        .reset_index(drop=True)
    )


#%% PROCESS VARIABLE CORRELATION MATRIX

def process_correlation_matrix(
    df,
    process_variables,
):

    return (
        df[process_variables]
        .corr(method="pearson")
    )


#%% INITIAL MULTIVARIABLE REGRESSION

def fit_full_roughness_model(df):

    formula = """
        surface_roughness_um ~
        tool_wear
        + machine_deterioration
        + material_hardness_effect
        + material_machinability_effect
        + spindle_speed_rpm
        + feed_rate_mm_rev
        + coolant_flow_l_min
        + cutting_force_n
        + spindle_load_pct
        + temperature_c
        + vibration_mm_s
        + C(machine_id)
        + C(operator_id)
    """

    return smf.ols(
        formula=formula,
        data=df,
    ).fit()


#%% VARIANCE INFLATION FACTOR

def calculate_vif(
    df,
    process_variables,
):

    data = (
        df[process_variables]
        .dropna()
        .copy()
    )

    X = sm.add_constant(data)

    results = []

    for i, column in enumerate(X.columns):

        if column == "const":
            continue

        results.append(
            {
                "Variable": column,
                "VIF":
                    variance_inflation_factor(
                        X.values,
                        i,
                    ),
            }
        )

    return (
        pd.DataFrame(results)
        .sort_values(
            "VIF",
            ascending=False,
        )
        .reset_index(drop=True)
    )


#%% REDUCED ROUGHNESS MODEL

REDUCED_PROCESS_VARIABLES = [
    "tool_wear",
    "material_hardness_effect",
    "material_machinability_effect",
    "feed_rate_mm_rev",
    "coolant_flow_l_min",
]


def fit_reduced_roughness_model(df):

    formula = """
        surface_roughness_um ~
        tool_wear
        + material_hardness_effect
        + material_machinability_effect
        + feed_rate_mm_rev
        + coolant_flow_l_min
        + C(machine_id)
        + C(operator_id)
    """

    return smf.ols(
        formula=formula,
        data=df,
    ).fit()


#%% STANDARDIZED REGRESSION

def standardize_roughness_data(df):

    variables = [
        "surface_roughness_um",
        "tool_wear",
        "material_hardness_effect",
        "material_machinability_effect",
        "feed_rate_mm_rev",
        "coolant_flow_l_min",
    ]

    data = df.copy()

    data[variables] = (
        data[variables]
        .apply(zscore)
    )

    return data


def fit_standardized_roughness_model(df):

    formula = """
        surface_roughness_um ~
        tool_wear
        + material_hardness_effect
        + material_machinability_effect
        + feed_rate_mm_rev
        + coolant_flow_l_min
        + C(machine_id)
        + C(operator_id)
    """

    return smf.ols(
        formula=formula,
        data=df,
    ).fit()


#%% TOOL WEAR × MATERIAL INTERACTIONS

def fit_interaction_roughness_model(df):

    formula = """
        surface_roughness_um ~
        tool_wear
        + material_hardness_effect
        + material_machinability_effect
        + feed_rate_mm_rev
        + coolant_flow_l_min
        + tool_wear:material_hardness_effect
        + tool_wear:material_machinability_effect
        + C(machine_id)
        + C(operator_id)
    """

    return smf.ols(
        formula=formula,
        data=df,
    ).fit()


#%% OPERATOR INCREMENTAL CONTRIBUTION

def test_operator_contribution(df):

    without_operator = smf.ols(
        """
        surface_roughness_um ~
        tool_wear
        + material_hardness_effect
        + material_machinability_effect
        + feed_rate_mm_rev
        + coolant_flow_l_min
        + C(machine_id)
        """,
        data=df,
    ).fit()

    with_operator = smf.ols(
        """
        surface_roughness_um ~
        tool_wear
        + material_hardness_effect
        + material_machinability_effect
        + feed_rate_mm_rev
        + coolant_flow_l_min
        + C(machine_id)
        + C(operator_id)
        """,
        data=df,
    ).fit()

    incremental_r2 = (
        with_operator.rsquared
        - without_operator.rsquared
    )

    f_stat, p_value, df_diff = (
        with_operator.compare_f_test(
            without_operator
        )
    )

    results = pd.DataFrame(
        {
            "R2 Without Operator": [
                without_operator.rsquared
            ],
            "R2 With Operator": [
                with_operator.rsquared
            ],
            "Incremental R2": [
                incremental_r2
            ],
            "Partial F": [f_stat],
            "p-value": [p_value],
            "df Difference": [df_diff],
        }
    )

    return results


#%% MODEL DIAGNOSTICS

def plot_model_diagnostics(model):

    fitted = model.fittedvalues
    residuals = model.resid

    # Residuals vs fitted
    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.scatter(
        fitted,
        residuals,
        alpha=0.35,
    )

    ax.axhline(
        0,
        linestyle="--",
    )

    ax.set_xlabel(
        "Fitted Surface Roughness (um)"
    )
    ax.set_ylabel("Residual")

    ax.set_title(
        "Residuals vs Fitted — Reduced Roughness Model"
    )

    ax.grid(alpha=0.2)

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "roughness_residuals_vs_fitted.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)

    # Q-Q plot
    fig = sm.qqplot(
        residuals,
        line="45",
        fit=True,
    )

    plt.title(
        "Q-Q Plot — Reduced Roughness Model"
    )

    plt.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "roughness_qq_plot.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)


#%% COOK'S DISTANCE

def analyze_cooks_distance(model):

    influence = model.get_influence()

    cooks_d = (
        influence.cooks_distance[0]
    )

    threshold = 4 / len(cooks_d)

    results = pd.DataFrame(
        {
            "Threshold": [threshold],
            "Points Above Threshold": [
                int((cooks_d > threshold).sum())
            ],
            "Maximum Cook's Distance": [
                cooks_d.max()
            ],
        }
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.scatter(
        np.arange(len(cooks_d)),
        cooks_d,
        alpha=0.5,
    )

    ax.axhline(
        threshold,
        linestyle="--",
    )

    ax.set_xlabel("Observation")
    ax.set_ylabel("Cook's Distance")

    ax.set_title(
        "Cook's Distance — Reduced Roughness Model"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "roughness_cooks_distance.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)

    return results


#%% TOOL WEAR VALIDATION BY MACHINE

def tool_wear_correlation_by_machine(df):

    results = []

    for machine, group in df.groupby(
        "machine_id"
    ):

        correlation = (
            group["tool_wear"]
            .corr(
                group[
                    "surface_roughness_um"
                ]
            )
        )

        results.append(
            {
                "Machine": machine,
                "Parts": len(group),
                "Tool Wear vs Roughness Correlation":
                    correlation,
            }
        )

    return pd.DataFrame(results)


#%% TOOL WEAR VS SURFACE ROUGHNESS

def plot_roughness_vs_tool_wear(df):

    data = (
        df[
            [
                "tool_wear",
                "surface_roughness_um",
            ]
        ]
        .dropna()
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.scatter(
        data["tool_wear"],
        data["surface_roughness_um"],
        alpha=0.35,
        s=15,
    )

    coefficients = np.polyfit(
        data["tool_wear"],
        data["surface_roughness_um"],
        1,
    )

    x_line = np.linspace(
        data["tool_wear"].min(),
        data["tool_wear"].max(),
        200,
    )

    y_line = (
        coefficients[0] * x_line
        + coefficients[1]
    )

    ax.plot(
        x_line,
        y_line,
        linewidth=2,
        label="Linear Trend",
    )

    ax.axhline(
        1.60,
        linestyle="--",
        label="Roughness USL = 1.60 um",
    )

    ax.set_xlabel("Tool Wear")
    ax.set_ylabel(
        "Surface Roughness (um)"
    )

    ax.set_title(
        "Tool Wear vs Surface Roughness"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "roughness_vs_tool_wear.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)


#%% ROUGHNESS BY TOOL-WEAR RANGE

def analyze_roughness_by_tool_wear_bin(df):

    data = df.copy()

    bins = [
        0.0,
        0.2,
        0.4,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ]

    data["Tool Wear Range"] = pd.cut(
        data["tool_wear"],
        bins=bins,
        include_lowest=True,
        right=False,
    )

    summary = (
        data.groupby(
            "Tool Wear Range",
            observed=True,
        )
        .agg(
            Parts=("part_number", "count"),
            Mean_Roughness_um=(
                "surface_roughness_um",
                "mean",
            ),
            Roughness_Std_Dev_um=(
                "surface_roughness_um",
                "std",
            ),
            Roughness_Defects=(
                "roughness_pass",
                lambda x:
                    (~x.astype(bool)).sum(),
            ),
        )
        .reset_index()
    )

    summary["Roughness Defect Rate (%)"] = (
        100
        * summary["Roughness_Defects"]
        / summary["Parts"]
    )

    return summary


#%% MAIN ANALYSIS

def main():

    manufacturing_data = pd.read_csv(
        DATA_PATH
    )

    # Pass/fail and baseline performance
    validation = validate_roughness_pass(
        manufacturing_data
    )

    performance = summarize_roughness_performance(
        manufacturing_data
    )

    print("\nROUGHNESS PASS / FAIL VALIDATION")
    print(validation.to_string(index=False))

    print("\nOVERALL ROUGHNESS PERFORMANCE")
    print(
        performance
        .round(4)
        .to_string(index=False)
    )

    # Categorical screening
    categorical_screen = (
        categorical_variance_screen(
            manufacturing_data
        )
    )

    print("\nCATEGORICAL VARIANCE SCREENING")
    print(
        categorical_screen
        .round(4)
        .to_string(index=False)
    )

    # Performance by category
    for column, name in [
        ("machine_id", "Machine"),
        ("operator_id", "Operator"),
        ("shift_name", "Shift"),
        ("material_lot", "Material Lot"),
    ]:

        summary = analyze_roughness_by_category(
            manufacturing_data,
            column,
            name,
        )

        print(
            f"\nROUGHNESS BY {name.upper()}"
        )
        print(
            summary
            .round(4)
            .to_string(index=False)
        )

    # Material investigation
    material_properties = (
        analyze_material_lot_properties(
            manufacturing_data
        )
    )

    material_wear = (
        analyze_material_lot_tool_wear(
            manufacturing_data
        )
    )

    print(
        "\nMATERIAL LOT — ROUGHNESS AND MATERIAL PROPERTIES"
    )
    print(
        material_properties
        .round(4)
        .to_string(index=False)
    )

    print(
        "\nMATERIAL LOT × TOOL-WEAR EXPOSURE"
    )
    print(
        material_wear
        .round(4)
        .to_string(index=False)
    )

    # Machine × operator
    operator_machine, mean_matrix = (
        analyze_machine_operator(
            manufacturing_data
        )
    )

    print("\nROUGHNESS BY MACHINE × OPERATOR")
    print(
        operator_machine
        .round(4)
        .to_string(index=False)
    )

    print(
        "\nMEAN ROUGHNESS — OPERATOR × MACHINE"
    )
    print(
        mean_matrix
        .round(4)
        .to_string()
    )

    _, interaction_anova = (
        fit_machine_operator_interaction(
            manufacturing_data
        )
    )

    print("\nMACHINE × OPERATOR ANOVA")
    print(
        interaction_anova
        .round(6)
        .to_string()
    )

    plot_machine_operator_interaction(
        manufacturing_data
    )

    # Tukey comparisons
    tukey_results = (
        operator_comparisons_by_machine(
            manufacturing_data
        )
    )

    for machine, result in tukey_results.items():

        print(
            f"\nOPERATOR COMPARISONS — {machine}"
        )
        print(result)

    # Operator process profile
    process_profile = (
        operator_machine_process_profile(
            manufacturing_data
        )
    )

    print(
        "\nOPERATOR × MACHINE PROCESS-CONDITION PROFILE"
    )
    print(
        process_profile
        .round(4)
        .to_string(index=False)
    )

    # Continuous-variable screening
    correlations = (
        calculate_roughness_correlations(
            manufacturing_data,
            ROUGHNESS_PROCESS_VARIABLES,
        )
    )

    print(
        "\nCONTINUOUS VARIABLE CORRELATIONS"
    )
    print(
        correlations
        .round(4)
        .to_string(index=False)
    )

    correlation_matrix = (
        process_correlation_matrix(
            manufacturing_data,
            ROUGHNESS_PROCESS_VARIABLES,
        )
    )

    print(
        "\nPROCESS VARIABLE CORRELATION MATRIX"
    )
    print(
        correlation_matrix
        .round(3)
        .to_string()
    )

    # Full model
    full_model = fit_full_roughness_model(
        manufacturing_data
    )

    print(
        "\nINITIAL MULTIVARIABLE REGRESSION"
    )
    print(full_model.summary())

    full_vif = calculate_vif(
        manufacturing_data,
        ROUGHNESS_PROCESS_VARIABLES,
    )

    print("\nVIF — INITIAL MODEL")
    print(
        full_vif
        .round(2)
        .to_string(index=False)
    )

    # Reduced model
    reduced_model = (
        fit_reduced_roughness_model(
            manufacturing_data
        )
    )

    print(
        "\nREDUCED MULTIVARIABLE REGRESSION"
    )
    print(reduced_model.summary())

    reduced_vif = calculate_vif(
        manufacturing_data,
        REDUCED_PROCESS_VARIABLES,
    )

    print("\nVIF — REDUCED MODEL")
    print(
        reduced_vif
        .round(2)
        .to_string(index=False)
    )

    # Standardized models
    standardized_data = (
        standardize_roughness_data(
            manufacturing_data
        )
    )

    standardized_model = (
        fit_standardized_roughness_model(
            standardized_data
        )
    )

    print(
        "\nSTANDARDIZED MULTIVARIABLE REGRESSION"
    )
    print(
        standardized_model.summary()
    )

    interaction_model = (
        fit_interaction_roughness_model(
            standardized_data
        )
    )

    print(
        "\nSTANDARDIZED REGRESSION WITH INTERACTIONS"
    )
    print(
        interaction_model.summary()
    )

    # Operator incremental contribution
    operator_contribution = (
        test_operator_contribution(
            standardized_data
        )
    )

    print(
        "\nOPERATOR INCREMENTAL CONTRIBUTION"
    )
    print(
        operator_contribution
        .round(6)
        .to_string(index=False)
    )

    # Model diagnostics
    plot_model_diagnostics(
        reduced_model
    )

    cooks_summary = (
        analyze_cooks_distance(
            reduced_model
        )
    )

    print("\nCOOK'S DISTANCE")
    print(
        cooks_summary
        .round(6)
        .to_string(index=False)
    )

    # Validate tool-wear relationship
    wear_by_machine = (
        tool_wear_correlation_by_machine(
            manufacturing_data
        )
    )

    print(
        "\nTOOL WEAR vs ROUGHNESS — WITHIN MACHINE"
    )
    print(
        wear_by_machine
        .round(4)
        .to_string(index=False)
    )

    plot_roughness_vs_tool_wear(
        manufacturing_data
    )

    # Tool-wear ranges
    wear_summary = (
        analyze_roughness_by_tool_wear_bin(
            manufacturing_data
        )
    )

    print(
        "\nROUGHNESS PERFORMANCE BY TOOL-WEAR RANGE"
    )
    print(
        wear_summary
        .round(4)
        .to_string(index=False)
    )


#%% RUN

if __name__ == "__main__":
    main()