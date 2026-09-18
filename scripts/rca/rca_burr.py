#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 13:56:02 2026

@author: temilolagbadamosi-adeniyi
"""

"""
CNC Manufacturing Process Quality Improvement
Burr Height Root Cause Analysis
"""

#%% IMPORTS

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.preprocessing import StandardScaler
from statsmodels.stats.anova import anova_lm
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

def validate_burr_pass(df):
    """Verify stored burr pass/fail against the burr-height USL."""

    expected_pass = (
        df["burr_height_mm"]
        <= df["burr_usl_mm"]
    )

    matches = (
        df["burr_pass"].astype(bool)
        == expected_pass
    )

    return pd.DataFrame(
        {
            "Total Parts": [len(df)],
            "Matching Results": [
                int(matches.sum())
            ],
            "Mismatching Results": [
                int((~matches).sum())
            ],
        }
    )


#%% OVERALL BURR PERFORMANCE

def summarize_burr_performance(df):

    total_parts = len(df)

    defects = (
        ~df["burr_pass"].astype(bool)
    ).sum()

    return pd.DataFrame(
        {
            "Total Parts": [total_parts],
            "Burr Defects": [int(defects)],
            "Defect Rate (%)": [
                100 * defects / total_parts
            ],
            "Mean Burr Height (mm)": [
                df["burr_height_mm"].mean()
            ],
            "Burr Std Dev (mm)": [
                df["burr_height_mm"].std()
            ],
        }
    )


#%% CATEGORICAL VARIANCE SCREENING

def between_group_share(
    df,
    factor,
    target="burr_height_mm",
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
                between_group_share(
                    df,
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


#%% BURR PERFORMANCE BY CATEGORY

def analyze_burr_by_category(
    df,
    category_column,
    category_name,
):

    summary = (
        df.groupby(category_column)
        .agg(
            Parts=("part_number", "count"),
            Burr_Defects=(
                "burr_pass",
                lambda x:
                    (~x.astype(bool)).sum(),
            ),
            Mean_Burr_Height_mm=(
                "burr_height_mm",
                "mean",
            ),
            Median_Burr_Height_mm=(
                "burr_height_mm",
                "median",
            ),
            Burr_Std_Dev_mm=(
                "burr_height_mm",
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

    summary["Burr Defect Rate (%)"] = (
        100
        * summary["Burr_Defects"]
        / summary["Parts"]
    )

    return (
        summary
        .sort_values(
            "Burr Defect Rate (%)",
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
            Mean_Burr_Height_mm=(
                "burr_height_mm",
                "mean",
            ),
            Burr_Defects=(
                "burr_pass",
                lambda x:
                    (~x.astype(bool)).sum(),
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

    summary["Burr Defect Rate (%)"] = (
        100
        * summary["Burr_Defects"]
        / summary["Parts"]
    )

    return (
        summary
        .sort_values(
            "Burr Defect Rate (%)",
            ascending=False,
        )
        .reset_index(drop=True)
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


def analyze_burr_correlations(
    df,
    variables,
    target="burr_height_mm",
):

    correlations = (
        df[variables + [target]]
        .corr()[target]
        .drop(target)
        .sort_values(
            key=abs,
            ascending=False,
        )
    )

    return (
        correlations
        .rename("Correlation with Burr Height")
        .reset_index()
        .rename(
            columns={"index": "Process Variable"}
        )
    )


#%% INITIAL MULTIVARIABLE REGRESSION

def fit_full_burr_model(df):

    formula = """
        burr_height_mm ~
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


#%% MULTICOLLINEARITY SCREENING

def calculate_vif(
    df,
    variables,
):

    X = (
        df[variables]
        .dropna()
        .copy()
    )

    X = sm.add_constant(X)

    results = []

    for i, variable in enumerate(X.columns):

        if variable == "const":
            continue

        results.append(
            {
                "Variable": variable,
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


#%% OPERATOR INCREMENTAL CONTRIBUTION

def test_operator_incremental_contribution(df):

    base_formula = """
        burr_height_mm ~
        tool_wear
        + material_hardness_effect
        + material_machinability_effect
        + feed_rate_mm_rev
    """

    operator_formula = """
        burr_height_mm ~
        tool_wear
        + material_hardness_effect
        + material_machinability_effect
        + feed_rate_mm_rev
        + C(operator_id)
    """

    base_model = smf.ols(
        formula=base_formula,
        data=df,
    ).fit()

    operator_model = smf.ols(
        formula=operator_formula,
        data=df,
    ).fit()

    comparison = anova_lm(
        base_model,
        operator_model,
    )

    incremental_r2 = (
        operator_model.rsquared
        - base_model.rsquared
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
                incremental_r2
            ],
            "Incremental R2 (%)": [
                incremental_r2 * 100
            ],
        }
    )

    return (
        base_model,
        operator_model,
        comparison,
        results,
    )


#%% TOOL WEAR × MATERIAL INTERACTION TEST

def test_wear_material_interactions(df):

    main_formula = """
        burr_height_mm ~
        tool_wear
        + material_hardness_effect
        + material_machinability_effect
        + feed_rate_mm_rev
    """

    interaction_formula = """
        burr_height_mm ~
        tool_wear
        + material_hardness_effect
        + material_machinability_effect
        + feed_rate_mm_rev
        + tool_wear:material_hardness_effect
        + tool_wear:material_machinability_effect
    """

    main_model = smf.ols(
        formula=main_formula,
        data=df,
    ).fit()

    interaction_model = smf.ols(
        formula=interaction_formula,
        data=df,
    ).fit()

    comparison = anova_lm(
        main_model,
        interaction_model,
    )

    incremental_r2 = (
        interaction_model.rsquared
        - main_model.rsquared
    )

    results = pd.DataFrame(
        {
            "R2 Main Effects": [
                main_model.rsquared
            ],
            "R2 With Interactions": [
                interaction_model.rsquared
            ],
            "Incremental R2": [
                incremental_r2
            ],
            "Incremental R2 (%)": [
                incremental_r2 * 100
            ],
        }
    )

    return (
        main_model,
        interaction_model,
        comparison,
        results,
    )


#%% REDUCED STANDARDIZED REGRESSION

REDUCED_VARIABLES = [
    "tool_wear",
    "material_hardness_effect",
    "feed_rate_mm_rev",
]


def fit_standardized_burr_model(
    df,
    variables,
    target="burr_height_mm",
):

    model_data = (
        df[[target] + variables]
        .dropna()
        .copy()
    )

    scaler = StandardScaler()

    standardized_columns = [
        f"{variable}_z"
        for variable in variables
    ]

    model_data[standardized_columns] = (
        scaler.fit_transform(
            model_data[variables]
        )
    )

    formula = (
        f"{target} ~ "
        + " + ".join(
            standardized_columns
        )
    )

    model = smf.ols(
        formula=formula,
        data=model_data,
    ).fit()

    return model


#%% MATERIAL LOT INCREMENTAL CONTRIBUTION

def test_material_lot_incremental_contribution(df):

    base_formula = """
        burr_height_mm ~
        tool_wear
        + material_hardness_effect
        + feed_rate_mm_rev
    """

    lot_formula = """
        burr_height_mm ~
        tool_wear
        + material_hardness_effect
        + feed_rate_mm_rev
        + C(material_lot)
    """

    base_model = smf.ols(
        formula=base_formula,
        data=df,
    ).fit()

    lot_model = smf.ols(
        formula=lot_formula,
        data=df,
    ).fit()

    comparison = anova_lm(
        base_model,
        lot_model,
    )

    incremental_r2 = (
        lot_model.rsquared
        - base_model.rsquared
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
                incremental_r2
            ],
            "Incremental R2 (%)": [
                incremental_r2 * 100
            ],
        }
    )

    return (
        base_model,
        lot_model,
        comparison,
        results,
    )


#%% MODEL DIAGNOSTICS

def plot_burr_model_diagnostics(model):

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
        s=15,
    )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel(
        "Fitted Burr Height (mm)"
    )
    ax.set_ylabel(
        "Residual (mm)"
    )

    ax.set_title(
        "Residuals vs Fitted — Burr Height"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "burr_residuals_vs_fitted.png",
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
        "Q-Q Plot — Burr Height Residuals"
    )

    plt.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "burr_qq_plot.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)


#%% COOK'S DISTANCE

def analyze_burr_influence(model):

    influence = model.get_influence()

    cooks_distance = (
        influence.cooks_distance[0]
    )

    threshold = (
        4 / model.nobs
    )

    results = pd.DataFrame(
        {
            "Observations": [
                int(model.nobs)
            ],
            "4/n Threshold": [
                threshold
            ],
            "Points Above Threshold": [
                int(
                    (
                        cooks_distance
                        > threshold
                    ).sum()
                )
            ],
            "Maximum Cook's Distance": [
                cooks_distance.max()
            ],
        }
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.scatter(
        np.arange(
            len(cooks_distance)
        ),
        cooks_distance,
        alpha=0.5,
    )

    ax.axhline(
        threshold,
        linestyle="--",
    )

    ax.set_xlabel("Observation")
    ax.set_ylabel(
        "Cook's Distance"
    )

    ax.set_title(
        "Cook's Distance — Burr Height Model"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "burr_cooks_distance.png",
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
                group["burr_height_mm"]
            )
        )

        results.append(
            {
                "Machine": machine,
                "Parts": len(group),
                "Tool Wear vs Burr Correlation":
                    correlation,
            }
        )

    return pd.DataFrame(results)


#%% TOOL WEAR VS BURR HEIGHT

def plot_tool_wear_vs_burr(df):

    data = (
        df[
            [
                "tool_wear",
                "burr_height_mm",
                "burr_usl_mm",
            ]
        ]
        .dropna()
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.scatter(
        data["tool_wear"],
        data["burr_height_mm"],
        alpha=0.30,
        s=15,
    )

    coefficients = np.polyfit(
        data["tool_wear"],
        data["burr_height_mm"],
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

    burr_usl = (
        data["burr_usl_mm"]
        .median()
    )

    ax.axhline(
        burr_usl,
        linestyle="--",
        linewidth=2,
        label=(
            f"Burr USL = "
            f"{burr_usl:.3f} mm"
        ),
    )

    ax.set_xlabel("Tool Wear")
    ax.set_ylabel(
        "Burr Height (mm)"
    )

    ax.set_title(
        "Tool Wear vs Burr Height"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "burr_vs_tool_wear.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)


#%% MAIN ANALYSIS

def main():

    manufacturing_data = pd.read_csv(
        DATA_PATH
    )

    # Pass/fail validation
    validation = validate_burr_pass(
        manufacturing_data
    )

    print("\nBURR PASS / FAIL VALIDATION")
    print(
        validation.to_string(
            index=False
        )
    )

    # Overall performance
    performance = summarize_burr_performance(
        manufacturing_data
    )

    print("\nOVERALL BURR HEIGHT PERFORMANCE")
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

    print(
        "\nCATEGORICAL VARIANCE SCREENING — BURR HEIGHT"
    )

    print(
        categorical_screen
        .round(4)
        .to_string(index=False)
    )

    # Material lot and operator
    for column, name in [
        ("material_lot", "Material Lot"),
        ("operator_id", "Operator"),
    ]:

        summary = analyze_burr_by_category(
            manufacturing_data,
            column,
            name,
        )

        print(
            f"\nBURR HEIGHT BY {name.upper()}"
        )

        print(
            summary
            .round(4)
            .to_string(index=False)
        )

    # Material properties
    material_properties = (
        analyze_material_lot_properties(
            manufacturing_data
        )
    )

    print(
        "\nMATERIAL LOT — BURR HEIGHT AND MATERIAL PROPERTIES"
    )

    print(
        material_properties
        .round(4)
        .to_string(index=False)
    )

    # Continuous-variable screening
    correlations = (
        analyze_burr_correlations(
            manufacturing_data,
            CONTINUOUS_VARIABLES,
        )
    )

    print(
        "\nCONTINUOUS VARIABLE CORRELATIONS — BURR HEIGHT"
    )

    print(
        correlations
        .round(4)
        .to_string(index=False)
    )

    # Initial regression
    full_model = fit_full_burr_model(
        manufacturing_data
    )

    print(
        "\nINITIAL MULTIVARIABLE REGRESSION — BURR HEIGHT"
    )

    print(full_model.summary())

    # VIF
    burr_vif = calculate_vif(
        manufacturing_data,
        CONTINUOUS_VARIABLES,
    )

    print(
        "\nVARIANCE INFLATION FACTORS — BURR HEIGHT"
    )

    print(
        burr_vif
        .round(2)
        .to_string(index=False)
    )

    # Operator contribution
    (
        _,
        _,
        operator_test,
        operator_results,
    ) = test_operator_incremental_contribution(
        manufacturing_data
    )

    print(
        "\nOPERATOR INCREMENTAL CONTRIBUTION — BURR HEIGHT"
    )

    print(
        operator_results
        .round(6)
        .to_string(index=False)
    )

    print("\nPARTIAL F-TEST — OPERATOR")
    print(operator_test.to_string())

    # Tool wear × material interactions
    (
        _,
        interaction_model,
        interaction_test,
        interaction_results,
    ) = test_wear_material_interactions(
        manufacturing_data
    )

    print(
        "\nTOOL WEAR × MATERIAL INTERACTIONS — BURR HEIGHT"
    )

    print(
        interaction_results
        .round(6)
        .to_string(index=False)
    )

    print(
        "\nINTERACTION MODEL COEFFICIENTS"
    )

    print(
        interaction_model
        .summary()
        .tables[1]
    )

    print(
        "\nPARTIAL F-TEST — INTERACTION TERMS"
    )

    print(
        interaction_test.to_string()
    )

    # Reduced standardized model
    reduced_model = (
        fit_standardized_burr_model(
            manufacturing_data,
            REDUCED_VARIABLES,
        )
    )

    print(
        "\nREDUCED STANDARDIZED REGRESSION — BURR HEIGHT"
    )

    print(
        reduced_model.summary()
    )

    # Material lot contribution
    (
        _,
        _,
        lot_test,
        lot_results,
    ) = test_material_lot_incremental_contribution(
        manufacturing_data
    )

    print(
        "\nMATERIAL LOT INCREMENTAL CONTRIBUTION — BURR HEIGHT"
    )

    print(
        lot_results
        .round(6)
        .to_string(index=False)
    )

    print(
        "\nPARTIAL F-TEST — MATERIAL LOT"
    )

    print(
        lot_test.to_string()
    )

    # Model diagnostics
    plot_burr_model_diagnostics(
        reduced_model
    )

    influence_results = (
        analyze_burr_influence(
            reduced_model
        )
    )

    print(
        "\nCOOK'S DISTANCE — BURR HEIGHT"
    )

    print(
        influence_results
        .round(6)
        .to_string(index=False)
    )

    # Tool-wear validation
    wear_by_machine = (
        tool_wear_correlation_by_machine(
            manufacturing_data
        )
    )

    print(
        "\nTOOL WEAR vs BURR HEIGHT — WITHIN MACHINE"
    )

    print(
        wear_by_machine
        .round(4)
        .to_string(index=False)
    )

    plot_tool_wear_vs_burr(
        manufacturing_data
    )


#%% RUN

if __name__ == "__main__":
    main()