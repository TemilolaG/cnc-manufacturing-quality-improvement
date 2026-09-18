#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 06:41:47 2026

@author: temilolagbadamosi-adeniyi
"""

"""
Manufacturing Process Quality Improvement
Defect Pareto Analysis

Identifies the most frequent manufacturing defect types and their
cumulative contribution to total observed defects.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd



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
FIGURE_DIR = SCRIPT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)



#%% Pareto analysis


def calculate_defect_pareto(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate defect frequency and cumulative percentage.

    Each defect type is counted independently, so a part with
    multiple defects contributes to more than one category.
    """

    defect_columns = {
        "Bore Diameter": "bore_pass",
        "Surface Roughness": "roughness_pass",
        "Flatness": "flatness_pass",
        "Burr": "burr_pass",
    }

    defect_counts = []

    for defect_name, pass_column in defect_columns.items():

        count = (~df[pass_column].astype(bool)).sum()

        defect_counts.append(
            {
                "Defect Type": defect_name,
                "Defect Count": int(count),
            }
        )

    pareto = (
        pd.DataFrame(defect_counts)
        .sort_values("Defect Count", ascending=False)
        .reset_index(drop=True)
    )

    total_defects = pareto["Defect Count"].sum()

    pareto["Defect Percentage (%)"] = (
        pareto["Defect Count"]
        / total_defects
        * 100
    )

    pareto["Cumulative Percentage (%)"] = (
        pareto["Defect Percentage (%)"].cumsum()
    )

    return pareto



#%% Pareto chart


def plot_defect_pareto(
    pareto: pd.DataFrame,
    save_path: Path,
) -> None:
    """Create and save the manufacturing defect Pareto chart."""

    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.bar(
        pareto["Defect Type"],
        pareto["Defect Count"],
    )

    ax1.set_xlabel("Defect Type")
    ax1.set_ylabel("Defect Count")
    ax1.set_title("Manufacturing Defect Pareto Analysis")

    ax2 = ax1.twinx()

    ax2.plot(
        pareto["Defect Type"],
        pareto["Cumulative Percentage (%)"],
        marker="o",
    )

    ax2.set_ylabel("Cumulative Percentage (%)")
    ax2.set_ylim(0, 105)

    ax2.axhline(
        80,
        linestyle="--",
        linewidth=1,
    )

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()



#%% Main analysis


def main() -> None:

    manufacturing_data = pd.read_csv(DATA_PATH)

    defect_pareto = calculate_defect_pareto(
        manufacturing_data
    )

    print("\nDEFECT PARETO SUMMARY")
    print(
        defect_pareto
        .round(2)
        .to_string(index=False)
    )

    defect_pareto.to_csv(
        OUTPUT_DIR / "defect_pareto_summary.csv",
        index=False,
    )

    plot_defect_pareto(
        defect_pareto,
        FIGURE_DIR / "defect_pareto.png",
    )


if __name__ == "__main__":
    main()