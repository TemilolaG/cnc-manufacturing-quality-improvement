#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 14:27:20 2026

@author: temilolagbadamosi-adeniyi
"""

"""
Manufacturing Process Quality Improvement
Performance Analysis

Evaluates overall manufacturing performance, including production
metrics, cycle time, OEE, downtime, reliability, and sources of
cycle-time variation.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from multi_machine_manufacturing_simulator import (
    SimulationConfig,
    ProductionCalendar,
)



#%% Project paths


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



#%% Data loading and inspection


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load the manufacturing dataset."""

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    dataframe = pd.read_csv(file_path)

    if dataframe.empty:
        raise ValueError(f"Dataset is empty: {file_path}")

    return dataframe


def inspect_dataset(dataframe: pd.DataFrame) -> None:
    """Display the basic structure of the manufacturing dataset."""

    print("\nDATASET OVERVIEW")
    print(f"Rows: {dataframe.shape[0]:,}")
    print(f"Columns: {dataframe.shape[1]:,}")

    print("\nColumns")
    for number, column in enumerate(dataframe.columns, start=1):
        print(f"{number:>2}. {column}")

    print("\nData types")
    print(dataframe.dtypes.to_string())

    print("\nFirst five rows")
    print(dataframe.head().to_string(index=False))



#%% Production performance


def production_summary(df: pd.DataFrame) -> pd.Series:
    """Calculate overall production counts."""

    return pd.Series(
        {
            "Total Parts Produced": len(df),
            "First Pass Good": df["first_pass_good"].sum(),
            "Good After Disposition": df["good_after_disposition"].sum(),
            "Scrapped Parts": df["scrap_unit"].sum(),
            "Reworked Parts": df["rework_unit"].sum(),
        }
    )


def calculate_performance_metrics(df: pd.DataFrame) -> pd.Series:
    """Calculate primary manufacturing quality metrics."""

    total_parts = len(df)
    first_pass_good = df["first_pass_good"].sum()
    reworked_parts = df["rework_unit"].sum()
    scrapped_parts = df["scrap_unit"].sum()

    return pd.Series(
        {
            "First Pass Yield (%)":
                first_pass_good / total_parts * 100,

            "Rework Rate (%)":
                reworked_parts / total_parts * 100,

            "Scrap Rate (%)":
                scrapped_parts / total_parts * 100,

            "Final Yield (%)":
                (first_pass_good + reworked_parts) / total_parts * 100,
        }
    )



#%% Cycle-time analysis


def cycle_time_summary(df: pd.DataFrame) -> pd.Series:
    """Calculate overall cycle-time statistics."""

    cycle = df["cycle_time_sec"]

    return pd.Series(
        {
            "Average Cycle Time (sec)": cycle.mean(),
            "Median Cycle Time (sec)": cycle.median(),
            "Minimum Cycle Time (sec)": cycle.min(),
            "Maximum Cycle Time (sec)": cycle.max(),
            "Cycle Time Std Dev (sec)": cycle.std(),
            "Ideal Cycle Time (sec)": df["ideal_cycle_time_sec"].iloc[0],
            "Average Takt Time (sec)": df["takt_time_sec"].mean(),
        }
    )


def grouped_cycle_time_summary(
    df: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    """Summarize cycle time by a selected grouping variable."""

    return (
        df.groupby(group_column)["cycle_time_sec"]
        .agg(
            Parts="count",
            Mean="mean",
            Median="median",
            Std="std",
            Minimum="min",
            Maximum="max",
        )
        .round(2)
        .sort_values("Mean")
    )



#%% Overall equipment effectiveness


def calculate_machine_oee(
    df: pd.DataFrame,
    production_calendar,
) -> pd.DataFrame:
    """Calculate OEE for each machine."""

    data = df.copy()

    data["scheduled_start_timestamp"] = pd.to_datetime(
        data["scheduled_start_timestamp"],
        errors="coerce",
    )

    data["end_timestamp"] = pd.to_datetime(
        data["end_timestamp"],
        errors="coerce",
    )

    machine_results = []

    for machine_id, group in data.groupby("machine_id"):

        first_scheduled_start = group[
            "scheduled_start_timestamp"
        ].min()

        final_end_time = group["end_timestamp"].max()

        planned_production_time_sec = (
            production_calendar.scheduled_seconds_between(
                first_scheduled_start,
                final_end_time,
            )
        )

        total_downtime_min = (
            pd.to_numeric(
                group["total_downtime_min"],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )

        total_downtime_sec = total_downtime_min * 60

        run_time_sec = max(
            planned_production_time_sec - total_downtime_sec,
            0,
        )

        total_count = len(group)

        first_pass_good_count = int(
            group["first_pass_good"]
            .fillna(False)
            .astype(bool)
            .sum()
        )

        availability = (
            run_time_sec / planned_production_time_sec
            if planned_production_time_sec > 0
            else np.nan
        )

        ideal_production_time_sec = (
            pd.to_numeric(
                group["ideal_cycle_time_sec"],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )

        performance = (
            ideal_production_time_sec / run_time_sec
            if run_time_sec > 0
            else np.nan
        )

        if pd.notna(performance):
            performance = min(performance, 1.0)

        quality = (
            first_pass_good_count / total_count
            if total_count > 0
            else np.nan
        )

        oee = availability * performance * quality

        machine_results.append(
            {
                "Machine": machine_id,
                "Planned Production Time (min)":
                    planned_production_time_sec / 60,
                "Total Downtime (min)": total_downtime_min,
                "Run Time (min)": run_time_sec / 60,
                "Total Parts": total_count,
                "First-Pass Good Parts": first_pass_good_count,
                "Availability (%)": availability * 100,
                "Performance (%)": performance * 100,
                "Quality (%)": quality * 100,
                "OEE (%)": oee * 100,
            }
        )

    return (
        pd.DataFrame(machine_results)
        .round(2)
        .sort_values("OEE (%)", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------
# Downtime and reliability
# ---------------------------------------------------------------------

def calculate_machine_downtime_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate downtime metrics for each machine."""

    summary = (
        df.groupby("machine_id")
        .agg(
            total_parts=("part_number", "count"),
            planned_maintenance_min=(
                "maintenance_duration_min",
                "sum",
            ),
            unplanned_downtime_min=(
                "unplanned_downtime_min",
                "sum",
            ),
            total_downtime_min=(
                "total_downtime_min",
                "sum",
            ),
            downtime_events=(
                "total_downtime_min",
                lambda x: (x > 0).sum(),
            ),
            average_downtime_per_part_min=(
                "total_downtime_min",
                "mean",
            ),
            maximum_downtime_event_min=(
                "total_downtime_min",
                "max",
            ),
        )
        .reset_index()
    )

    summary["average_downtime_per_event_min"] = (
        summary["total_downtime_min"]
        / summary["downtime_events"].replace(0, np.nan)
    )

    return (
        summary
        .sort_values(
            "total_downtime_min",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def calculate_machine_reliability_summary(
    df: pd.DataFrame,
    machine_oee: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate MTBF and MTTR for each machine."""

    downtime_summary = (
        df.groupby("machine_id")
        .agg(
            downtime_events=(
                "total_downtime_min",
                lambda x: (x > 0).sum(),
            ),
            total_downtime_min=(
                "total_downtime_min",
                "sum",
            ),
        )
        .reset_index()
    )

    reliability = machine_oee.merge(
        downtime_summary,
        left_on="Machine",
        right_on="machine_id",
        how="left",
    )

    reliability.drop(columns="machine_id", inplace=True)

    reliability["MTBF (min)"] = (
        reliability["Run Time (min)"]
        / reliability["downtime_events"]
    )

    reliability["MTTR (min)"] = (
        reliability["Total Downtime (min)"]
        / reliability["downtime_events"]
    )

    return reliability[
        [
            "Machine",
            "Run Time (min)",
            "Total Downtime (min)",
            "downtime_events",
            "MTBF (min)",
            "MTTR (min)",
        ]
    ].round(2)



#%% Cycle-time variation screening


def between_group_share(
    df: pd.DataFrame,
    factor: str,
    target: str = "cycle_time_sec",
) -> dict:
    """Estimate the share of target variance occurring between groups."""

    grand_mean = df[target].mean()
    total_var = df[target].var(ddof=0)

    group_stats = (
        df.groupby(factor)[target]
        .agg(["mean", "count"])
    )

    between_var = (
        (
            group_stats["count"]
            * (group_stats["mean"] - grand_mean) ** 2
        ).sum()
        / len(df)
    )

    within_var = total_var - between_var

    return {
        "factor": factor,
        "total_var": total_var,
        "between_var": between_var,
        "within_var": within_var,
        "between_share_pct": 100 * between_var / total_var,
        "within_share_pct": 100 * within_var / total_var,
        "n_groups": df[factor].nunique(),
    }


def cycle_time_variance_screen(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare between-group cycle-time variation across factors."""

    factors = [
        "machine_id",
        "operator_id",
        "shift_name",
    ]

    results = [
        between_group_share(df, factor)
        for factor in factors
    ]

    return (
        pd.DataFrame(results)
        .sort_values(
            "between_share_pct",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def operator_variation_by_machine(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize operator cycle-time variation within each machine."""

    return (
        df.groupby(
            ["machine_id", "operator_id"]
        )["cycle_time_sec"]
        .agg(
            mean="mean",
            std="std",
            count="count",
        )
        .reset_index()
    )


def levene_tests_by_machine(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Test equality of operator cycle-time variance within each machine."""

    results = []

    for machine_id, machine_data in df.groupby("machine_id"):

        groups = [
            group["cycle_time_sec"].values
            for _, group in machine_data.groupby("operator_id")
        ]

        statistic, p_value = stats.levene(*groups)

        results.append(
            {
                "machine_id": machine_id,
                "levene_statistic": statistic,
                "p_value": p_value,
            }
        )

    return pd.DataFrame(results)


def machine_cycle_time_contribution(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate each machine's contribution to between-machine variation."""

    grand_mean = df["cycle_time_sec"].mean()
    n_total = len(df)

    machine_stats = (
        df.groupby("machine_id")["cycle_time_sec"]
        .agg(["mean", "count"])
    )

    machine_stats["weighted_sq_dev"] = (
        machine_stats["count"]
        / n_total
        * (machine_stats["mean"] - grand_mean) ** 2
    )

    total_between_var = machine_stats[
        "weighted_sq_dev"
    ].sum()

    machine_stats["pct_of_total_gap"] = (
        machine_stats["weighted_sq_dev"]
        / total_between_var
        * 100
    )

    return (
        machine_stats
        .sort_values(
            "pct_of_total_gap",
            ascending=False,
        )
        .reset_index()
    )



#%% Main analysis


def main() -> None:

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manufacturing_data = load_dataset(DATA_PATH)

    inspect_dataset(manufacturing_data)

    production = production_summary(manufacturing_data)
    performance = calculate_performance_metrics(manufacturing_data)
    cycle_summary = cycle_time_summary(manufacturing_data)

    machine_cycle = grouped_cycle_time_summary(
        manufacturing_data,
        "machine_id",
    )

    shift_cycle = grouped_cycle_time_summary(
        manufacturing_data,
        "shift_name",
    )

    operator_cycle = grouped_cycle_time_summary(
        manufacturing_data,
        "operator_id",
    )

    operator_machine = pd.crosstab(
        manufacturing_data["operator_id"],
        manufacturing_data["machine_id"],
    )

    shift_machine = pd.crosstab(
        manufacturing_data["shift_name"],
        manufacturing_data["machine_id"],
    )

    operator_machine_pct = (
        pd.crosstab(
            manufacturing_data["operator_id"],
            manufacturing_data["machine_id"],
            normalize="index",
        )
        * 100
    ).round(1)

    config = SimulationConfig()
    production_calendar = ProductionCalendar(config)

    machine_oee = calculate_machine_oee(
        manufacturing_data,
        production_calendar,
    )

    machine_downtime = calculate_machine_downtime_summary(
        manufacturing_data
    )

    machine_reliability = calculate_machine_reliability_summary(
        manufacturing_data,
        machine_oee,
    )

    # Reliability over an 8-hour operating period
    operating_period_min = 480

    machine_reliability["Reliability (%)"] = (
        np.exp(
            -operating_period_min
            / machine_reliability["MTBF (min)"]
        )
        * 100
    )

    variance_screen = cycle_time_variance_screen(
        manufacturing_data
    )

    operator_machine_variation = operator_variation_by_machine(
        manufacturing_data
    )

    levene_results = levene_tests_by_machine(
        manufacturing_data
    )

    machine_contribution = machine_cycle_time_contribution(
        manufacturing_data
    )

    print("\nPRODUCTION SUMMARY")
    print(production.to_string())

    print("\nPERFORMANCE METRICS")
    print(performance.round(2).to_string())

    print("\nCYCLE TIME SUMMARY")
    print(cycle_summary.round(2).to_string())

    print("\nCYCLE TIME BY MACHINE")
    print(machine_cycle.to_string())

    print("\nCYCLE TIME BY SHIFT")
    print(shift_cycle.to_string())

    print("\nCYCLE TIME BY OPERATOR")
    print(operator_cycle.to_string())

    print("\nOPERATOR × MACHINE ASSIGNMENT")
    print(operator_machine.to_string())

    print("\nSHIFT × MACHINE ASSIGNMENT")
    print(shift_machine.to_string())

    print("\nOPERATOR × MACHINE ASSIGNMENT (%)")
    print(operator_machine_pct.to_string())

    print("\nOEE BY MACHINE")
    print(machine_oee.to_string(index=False))

    print("\nDOWNTIME BY MACHINE")
    print(machine_downtime.to_string(index=False))

    print("\nMTBF, MTTR, AND RELIABILITY BY MACHINE")
    print(machine_reliability.to_string(index=False))

    print("\nCYCLE-TIME VARIANCE SCREEN")
    print(variance_screen.to_string(index=False))

    print("\nOPERATOR VARIATION WITHIN MACHINE")
    print(operator_machine_variation.to_string(index=False))

    print("\nLEVENE TESTS BY MACHINE")
    print(levene_results.to_string(index=False))

    print("\nMACHINE CONTRIBUTION TO CYCLE-TIME DIFFERENCES")
    print(machine_contribution.to_string(index=False))

    machine_oee.to_csv(
        OUTPUT_DIR / "machine_oee_summary.csv",
        index=False,
    )

    machine_downtime.to_csv(
        OUTPUT_DIR / "machine_downtime_summary.csv",
        index=False,
    )

    machine_reliability.to_csv(
        OUTPUT_DIR / "machine_reliability_summary.csv",
        index=False,
    )


if __name__ == "__main__":
    main()