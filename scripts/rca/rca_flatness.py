#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 14:35:16 2026

@author: temilolagbadamosi-adeniyi
"""

"""
CNC Manufacturing Process Quality Improvement
Flatness Root Cause Analysis
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


#%% PASS / FAIL VALIDATION

def validate_flatness_pass_fail(df):
    """Verify stored flatness pass/fail against the flatness USL."""

    data = df.loc[
        df["flatness_mm"].notna()
        & df["flatness_usl_mm"].notna()
    ].copy()

    expected_pass = (
        data["flatness_mm"]
        <= data["flatness_usl_mm"]
    )

    stored_pass = (
        data["flatness_pass"].astype(bool)
    )

    matches = (
        expected_pass == stored_pass
    )

    results = pd.DataFrame(
        {
            "Evaluable Parts": [len(data)],
            "Matching Records": [
                int(matches.sum())
            ],
            "Mismatches": [
                int((~matches).sum())
            ],
        }
    )

    return results


#%% OVERALL FLATNESS PERFORMANCE

def summarize_flatness_performance(df):

    valid = df["flatness_mm"].notna()

    measurement_count = int(valid.sum())

    defects = int(
        (
            ~df.loc[
                valid,
                "flatness_pass",
            ].astype(bool)
        ).sum()
    )

    return pd.DataFrame(
        {
            "Total Parts": [len(df)],
            "Parts With Flatness Data": [
                measurement_count
            ],
            "Missing Measurements": [
                int((~valid).sum())
            ],
            "Flatness Defects": [
                defects
            ],
            "Defect Rate (%)": [
                100
                * defects
                / measurement_count
            ],
            "Mean Flatness (mm)": [
                df["flatness_mm"].mean()
            ],
            "Median Flatness (mm)": [
                df["flatness_mm"].median()
            ],
            "Flatness Std Dev (mm)": [
                df["flatness_mm"].std()
            ],
        }
    )


#%% CATEGORICAL VARIANCE SCREENING

def between_group_share(
    df,
    factor,
    target="flatness_mm",
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


#%% FLATNESS BY MACHINE

def summarize_flatness_by_machine(df):

    data = df.loc[
        df["flatness_mm"].notna()
    ].copy()

    summary = (
        data.groupby("machine_id")
        .agg(
            Parts=(
                "flatness_mm",
                "size",
            ),
            Flatness_Defects=(
                "flatness_pass",
                lambda x:
                    (~x.astype(bool)).sum(),
            ),
            Mean_Flatness_mm=(
                "flatness_mm",
                "mean",
            ),
            Median_Flatness_mm=(
                "flatness_mm",
                "median",
            ),
            Flatness_SD_mm=(
                "flatness_mm",
                "std",
            ),
        )
        .reset_index()
    )

    summary["Flatness_Defect_Rate (%)"] = (
        100
        * summary["Flatness_Defects"]
        / summary["Parts"]
    )

    return (
        summary
        .sort_values(
            "Mean_Flatness_mm",
            ascending=False,
        )
        .reset_index(drop=True)
    )


#%% FLATNESS BY MACHINE FIGURE

def plot_flatness_by_machine(df):

    data = df.loc[
        df["flatness_mm"].notna()
    ].copy()

    machines = sorted(
        data["machine_id"].unique()
    )

    plot_data = [
        data.loc[
            data["machine_id"] == machine,
            "flatness_mm",
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

    flatness_usl = (
        data["flatness_usl_mm"]
        .median()
    )

    ax.axhline(
        flatness_usl,
        linestyle="--",
        label=(
            f"Flatness USL = "
            f"{flatness_usl:.3f} mm"
        ),
    )

    ax.set_xlabel("Machine")
    ax.set_ylabel("Flatness (mm)")

    ax.set_title(
        "Flatness Distribution by CNC Machine"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "flatness_by_machine.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)


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


def analyze_flatness_correlations(
    df,
    variables,
):

    correlations = (
        df[
            ["flatness_mm"]
            + variables
        ]
        .corr()["flatness_mm"]
        .drop("flatness_mm")
        .sort_values(
            key=abs,
            ascending=False,
        )
    )

    return (
        correlations
        .rename(
            "Correlation with Flatness"
        )
        .reset_index()
        .rename(
            columns={
                "index":
                    "Process Variable"
            }
        )
    )


#%% INITIAL MULTIVARIABLE SCREENING

def fit_full_flatness_model(df):

    variables = (
        [
            "flatness_mm",
            "machine_id",
            "operator_id",
        ]
        + CONTINUOUS_VARIABLES
    )

    data = (
        df[variables]
        .dropna()
        .copy()
    )

    formula = """
        flatness_mm ~
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
        data=data,
    ).fit()


#%% NESTED MODEL CONTRIBUTION TEST

def nested_contribution_test(
    df,
    required_columns,
    base_formula,
    expanded_formula,
):

    data = (
        df[required_columns]
        .dropna()
        .copy()
    )

    base_model = smf.ols(
        base_formula,
        data=data,
    ).fit()

    expanded_model = smf.ols(
        expanded_formula,
        data=data,
    ).fit()

    comparison = anova_lm(
        base_model,
        expanded_model,
    )

    incremental_r2 = (
        expanded_model.rsquared
        - base_model.rsquared
    )

    results = pd.DataFrame(
        {
            "Observations": [
                len(data)
            ],
            "R2 Baseline": [
                base_model.rsquared
            ],
            "R2 Expanded": [
                expanded_model.rsquared
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
        expanded_model,
        comparison,
        results,
    )


#%% FLATNESS BY MATERIAL LOT

def summarize_flatness_by_material_lot(df):

    data = df.loc[
        df["flatness_mm"].notna()
    ].copy()

    summary = (
        data.groupby("material_lot")
        .agg(
            Parts=(
                "flatness_mm",
                "size",
            ),
            Flatness_Defects=(
                "flatness_pass",
                lambda x:
                    (~x.astype(bool)).sum(),
            ),
            Mean_Flatness_mm=(
                "flatness_mm",
                "mean",
            ),
            Median_Flatness_mm=(
                "flatness_mm",
                "median",
            ),
            Flatness_SD_mm=(
                "flatness_mm",
                "std",
            ),
        )
        .reset_index()
    )

    summary["Flatness_Defect_Rate (%)"] = (
        100
        * summary["Flatness_Defects"]
        / summary["Parts"]
    )

    return (
        summary
        .sort_values(
            "Mean_Flatness_mm",
            ascending=False,
        )
        .reset_index(drop=True)
    )


#%% PRODUCTION-SEQUENCE CHECK

def analyze_flatness_sequence_by_machine(df):

    data = (
        df[
            [
                "part_number",
                "machine_id",
                "flatness_mm",
            ]
        ]
        .dropna()
        .copy()
    )

    data["machine_sequence"] = (
        data.groupby(
            "machine_id"
        )["part_number"]
        .rank(method="first")
        .astype(int)
    )

    max_sequence = (
        data["machine_sequence"]
        .max()
    )

    edges = np.arange(
        0,
        max_sequence + 100,
        100,
    )

    data["sequence_bin"] = pd.cut(
        data["machine_sequence"],
        bins=edges,
        right=True,
        include_lowest=True,
    )

    summary = (
        data.groupby(
            [
                "machine_id",
                "sequence_bin",
            ],
            observed=True,
        )
        .agg(
            Parts=(
                "flatness_mm",
                "size",
            ),
            Mean_Flatness_mm=(
                "flatness_mm",
                "mean",
            ),
            Flatness_SD_mm=(
                "flatness_mm",
                "std",
            ),
        )
        .reset_index()
    )

    mean_table = summary.pivot(
        index="sequence_bin",
        columns="machine_id",
        values="Mean_Flatness_mm",
    )

    return (
        data,
        summary,
        mean_table,
    )


#%% PRODUCTION-SEQUENCE FIGURE

def plot_flatness_sequence(
    sequence_summary,
):

    plot_data = (
        sequence_summary.copy()
    )

    plot_data["Sequence Midpoint"] = (
        plot_data["sequence_bin"]
        .apply(
            lambda interval:
                interval.mid
        )
        .astype(float)
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for machine, group in (
        plot_data.groupby(
            "machine_id"
        )
    ):

        ax.plot(
            group["Sequence Midpoint"],
            group["Mean_Flatness_mm"],
            marker="o",
            label=machine,
        )

    ax.set_xlabel(
        "Production Sequence"
    )

    ax.set_ylabel(
        "Mean Flatness (mm)"
    )

    ax.set_title(
        "Flatness by Machine Across Production Sequence"
    )

    ax.legend(
        title="Machine"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "flatness_machine_sequence.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)


#%% CROSS-STRATIFICATION ANOVA

def cross_stratification_anova(
    df,
    factor_a,
    factor_b,
    target,
):

    data = (
        df[
            [
                target,
                factor_a,
                factor_b,
            ]
        ]
        .dropna()
        .copy()
    )

    count_table = pd.crosstab(
        data[factor_a],
        data[factor_b],
    )

    mean_table = pd.pivot_table(
        data,
        values=target,
        index=factor_a,
        columns=factor_b,
        aggfunc="mean",
    )

    formula = (
        f"{target} ~ "
        f"C({factor_a}) + "
        f"C({factor_b}) + "
        f"C({factor_a}):C({factor_b})"
    )

    model = smf.ols(
        formula=formula,
        data=data,
    ).fit()

    anova = sm.stats.anova_lm(
        model,
        typ=2,
    )

    residual_ss = (
        anova.loc[
            "Residual",
            "sum_sq",
        ]
    )

    anova["partial_eta_sq"] = np.nan

    effects = [
        f"C({factor_a})",
        f"C({factor_b})",
        (
            f"C({factor_a}):"
            f"C({factor_b})"
        ),
    ]

    for effect in effects:

        if effect in anova.index:

            effect_ss = (
                anova.loc[
                    effect,
                    "sum_sq",
                ]
            )

            anova.loc[
                effect,
                "partial_eta_sq",
            ] = (
                effect_ss
                / (
                    effect_ss
                    + residual_ss
                )
            )

    return (
        count_table,
        mean_table,
        model,
        anova,
    )


#%% CROSS-STRATIFICATION FIGURE

def plot_cross_stratification(
    mean_table,
    xlabel,
    title,
    filename,
):

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for column in mean_table.columns:

        ax.plot(
            mean_table.index.astype(str),
            mean_table[column],
            marker="o",
            label=str(column),
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(
        "Mean Flatness (mm)"
    )

    ax.set_title(title)

    ax.tick_params(
        axis="x",
        rotation=45,
    )

    ax.legend(
        title=mean_table.columns.name,
        bbox_to_anchor=(
            1.02,
            1,
        ),
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


#%% FINAL RETAINED CATEGORICAL MODEL

def fit_final_flatness_model(df):

    data = (
        df[
            [
                "flatness_mm",
                "machine_id",
                "operator_id",
                "material_lot",
            ]
        ]
        .dropna()
        .copy()
    )

    model = smf.ols(
        formula="""
            flatness_mm ~
            C(machine_id)
            + C(operator_id)
            + C(material_lot)
        """,
        data=data,
    ).fit()

    results = pd.DataFrame(
        {
            "Observations": [
                int(model.nobs)
            ],
            "R2": [
                model.rsquared
            ],
            "Adjusted R2": [
                model.rsquared_adj
            ],
        }
    )

    return model, results


#%% MAIN ANALYSIS

def main():

    manufacturing_data = pd.read_csv(
        DATA_PATH
    )

    # Pass/fail validation
    validation = (
        validate_flatness_pass_fail(
            manufacturing_data
        )
    )

    print(
        "\nFLATNESS PASS / FAIL VALIDATION"
    )

    print(
        validation.to_string(
            index=False
        )
    )

    # Overall performance
    performance = (
        summarize_flatness_performance(
            manufacturing_data
        )
    )

    print(
        "\nOVERALL FLATNESS PERFORMANCE"
    )

    print(
        performance
        .round(5)
        .to_string(index=False)
    )

    # Categorical screening
    categorical_screen = (
        categorical_variance_screen(
            manufacturing_data
        )
    )

    print(
        "\nCATEGORICAL VARIANCE SCREENING — FLATNESS"
    )

    print(
        categorical_screen
        .round(5)
        .to_string(index=False)
    )

    # Machine performance
    machine_summary = (
        summarize_flatness_by_machine(
            manufacturing_data
        )
    )

    print(
        "\nFLATNESS PERFORMANCE BY MACHINE"
    )

    print(
        machine_summary
        .round(5)
        .to_string(index=False)
    )

    plot_flatness_by_machine(
        manufacturing_data
    )

    # Continuous screening
    correlations = (
        analyze_flatness_correlations(
            manufacturing_data,
            CONTINUOUS_VARIABLES,
        )
    )

    print(
        "\nCONTINUOUS VARIABLE SCREENING — FLATNESS"
    )

    print(
        correlations
        .round(4)
        .to_string(index=False)
    )

    # Initial full model
    full_model = (
        fit_full_flatness_model(
            manufacturing_data
        )
    )

    print(
        "\nINITIAL FULL MULTIVARIABLE SCREENING — FLATNESS"
    )

    print(full_model.summary())

    # Operator contribution
    (
        _,
        _,
        operator_test,
        operator_results,
    ) = nested_contribution_test(
        df=manufacturing_data,
        required_columns=[
            "flatness_mm",
            "machine_id",
            "operator_id",
        ],
        base_formula="""
            flatness_mm ~
            C(machine_id)
        """,
        expanded_formula="""
            flatness_mm ~
            C(machine_id)
            + C(operator_id)
        """,
    )

    print(
        "\nOPERATOR INCREMENTAL CONTRIBUTION — FLATNESS"
    )

    print(
        operator_results
        .round(6)
        .to_string(index=False)
    )

    print("\nPARTIAL F-TEST")
    print(operator_test.to_string())

    # Material-lot performance
    material_summary = (
        summarize_flatness_by_material_lot(
            manufacturing_data
        )
    )

    print(
        "\nFLATNESS PERFORMANCE BY MATERIAL LOT"
    )

    print(
        material_summary
        .round(5)
        .to_string(index=False)
    )

    # Material-lot contribution
    (
        _,
        _,
        material_test,
        material_results,
    ) = nested_contribution_test(
        df=manufacturing_data,
        required_columns=[
            "flatness_mm",
            "machine_id",
            "operator_id",
            "material_lot",
        ],
        base_formula="""
            flatness_mm ~
            C(machine_id)
            + C(operator_id)
        """,
        expanded_formula="""
            flatness_mm ~
            C(machine_id)
            + C(operator_id)
            + C(material_lot)
        """,
    )

    print(
        "\nMATERIAL LOT INCREMENTAL CONTRIBUTION — FLATNESS"
    )

    print(
        material_results
        .round(6)
        .to_string(index=False)
    )

    print("\nPARTIAL F-TEST")
    print(material_test.to_string())

    # Continuous process-variable block
    process_columns = [
        "flatness_mm",
        "machine_id",
        "operator_id",
        "material_lot",
    ] + CONTINUOUS_VARIABLES

    (
        _,
        _,
        process_test,
        process_results,
    ) = nested_contribution_test(
        df=manufacturing_data,
        required_columns=process_columns,
        base_formula="""
            flatness_mm ~
            C(machine_id)
            + C(operator_id)
            + C(material_lot)
        """,
        expanded_formula="""
            flatness_mm ~
            C(machine_id)
            + C(operator_id)
            + C(material_lot)
            + tool_wear
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
        """,
    )

    print(
        "\nMEASURED PROCESS-VARIABLE CONTRIBUTION — FLATNESS"
    )

    print(
        process_results
        .round(6)
        .to_string(index=False)
    )

    print("\nPARTIAL F-TEST")
    print(process_test.to_string())

    # Shift contribution
    (
        _,
        _,
        shift_test,
        shift_results,
    ) = nested_contribution_test(
        df=manufacturing_data,
        required_columns=[
            "flatness_mm",
            "machine_id",
            "operator_id",
            "material_lot",
            "shift_name",
        ],
        base_formula="""
            flatness_mm ~
            C(machine_id)
            + C(operator_id)
            + C(material_lot)
        """,
        expanded_formula="""
            flatness_mm ~
            C(machine_id)
            + C(operator_id)
            + C(material_lot)
            + C(shift_name)
        """,
    )

    print(
        "\nSHIFT INCREMENTAL CONTRIBUTION — FLATNESS"
    )

    print(
        shift_results
        .round(6)
        .to_string(index=False)
    )

    print("\nPARTIAL F-TEST")
    print(shift_test.to_string())

    # Production-sequence check
    (
        _,
        sequence_summary,
        sequence_table,
    ) = analyze_flatness_sequence_by_machine(
        manufacturing_data
    )

    print(
        "\nMEAN FLATNESS BY MACHINE × PRODUCTION SEQUENCE"
    )

    print(
        sequence_table
        .round(5)
        .to_string()
    )

    plot_flatness_sequence(
        sequence_summary
    )

    # Machine × material lot
    (
        machine_lot_counts,
        machine_lot_means,
        machine_lot_model,
        machine_lot_anova,
    ) = cross_stratification_anova(
        manufacturing_data,
        factor_a="material_lot",
        factor_b="machine_id",
        target="flatness_mm",
    )

    print(
        "\nMACHINE × MATERIAL LOT — PART COUNTS"
    )
    print(machine_lot_counts.to_string())

    print(
        "\nMACHINE × MATERIAL LOT — MEAN FLATNESS"
    )
    print(
        machine_lot_means
        .round(5)
        .to_string()
    )

    print(
        "\nMACHINE × MATERIAL LOT — ANOVA"
    )
    print(
        machine_lot_anova
        .round(6)
        .to_string()
    )

    print(
        f"\nR²: "
        f"{machine_lot_model.rsquared:.4f}"
    )

    plot_cross_stratification(
        machine_lot_means,
        xlabel="Material Lot",
        title=(
            "Machine × Material Lot — "
            "Mean Flatness"
        ),
        filename=(
            "flatness_machine_material_lot.png"
        ),
    )

    # Machine × operator
    (
        machine_operator_counts,
        machine_operator_means,
        machine_operator_model,
        machine_operator_anova,
    ) = cross_stratification_anova(
        manufacturing_data,
        factor_a="machine_id",
        factor_b="operator_id",
        target="flatness_mm",
    )

    print(
        "\nMACHINE × OPERATOR — PART COUNTS"
    )
    print(
        machine_operator_counts.to_string()
    )

    print(
        "\nMACHINE × OPERATOR — MEAN FLATNESS"
    )
    print(
        machine_operator_means
        .round(5)
        .to_string()
    )

    print(
        "\nMACHINE × OPERATOR — ANOVA"
    )
    print(
        machine_operator_anova
        .round(6)
        .to_string()
    )

    print(
        f"\nR²: "
        f"{machine_operator_model.rsquared:.4f}"
    )

    plot_cross_stratification(
        machine_operator_means,
        xlabel="Machine",
        title=(
            "Machine × Operator — "
            "Mean Flatness"
        ),
        filename=(
            "flatness_machine_operator.png"
        ),
    )

    # Operator × material lot
    (
        operator_lot_counts,
        operator_lot_means,
        operator_lot_model,
        operator_lot_anova,
    ) = cross_stratification_anova(
        manufacturing_data,
        factor_a="material_lot",
        factor_b="operator_id",
        target="flatness_mm",
    )

    print(
        "\nOPERATOR × MATERIAL LOT — PART COUNTS"
    )
    print(
        operator_lot_counts.to_string()
    )

    print(
        "\nOPERATOR × MATERIAL LOT — MEAN FLATNESS"
    )
    print(
        operator_lot_means
        .round(5)
        .to_string()
    )

    print(
        "\nOPERATOR × MATERIAL LOT — ANOVA"
    )
    print(
        operator_lot_anova
        .round(6)
        .to_string()
    )

    print(
        f"\nR²: "
        f"{operator_lot_model.rsquared:.4f}"
    )

    plot_cross_stratification(
        operator_lot_means,
        xlabel="Material Lot",
        title=(
            "Operator × Material Lot — "
            "Mean Flatness"
        ),
        filename=(
            "flatness_operator_material_lot.png"
        ),
    )

    # Final retained model
    (
        final_model,
        final_results,
    ) = fit_final_flatness_model(
        manufacturing_data
    )

    print(
        "\nFINAL RETAINED CATEGORICAL MODEL — FLATNESS"
    )

    print(
        final_results
        .round(6)
        .to_string(index=False)
    )


#%% RUN

if __name__ == "__main__":
    main()