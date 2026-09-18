#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 13:33:18 2026

@author: temilolagbadamosi-adeniyi
"""

"""
CNC Manufacturing Process Quality Improvement
Statistical Process Control Analysis

Evaluates process stability for Bore Diameter, Surface Roughness,
Flatness, and Burr Height using machine-level I-MR control charts
and selected SPC special-cause rules.
"""

#%% IMPORTS

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


#%% PROJECT PATHS

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()
PROJECT_ROOT = SCRIPT_DIR.parent

DATA_PATH = (PROJECT_ROOT/ "CNC"/ "manufacturing_quality_improvement_data.csv")

OUTPUT_DIR = SCRIPT_DIR / "outputs"
FIGURE_DIR = SCRIPT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


#%% LOAD DATA

manufacturing_data = pd.read_csv(DATA_PATH)


#%% I-MR STATISTICS

def calculate_imr_statistics(data, measurement_column):
    """Calculate Individuals-Moving Range control chart statistics."""

    data = data.copy()

    data["moving_range"] = (
        data[measurement_column]
        .diff()
        .abs()
    )

    mean = data[measurement_column].mean()
    std = data[measurement_column].std()

    mr_bar = data["moving_range"].mean()

    d2 = 1.128
    estimated_sigma = mr_bar / d2

    i_ucl = mean + 3 * estimated_sigma
    i_lcl = mean - 3 * estimated_sigma

    D3 = 0
    D4 = 3.267

    mr_ucl = D4 * mr_bar
    mr_lcl = D3 * mr_bar

    data["i_out_of_control"] = (
        (data[measurement_column] > i_ucl)
        | (data[measurement_column] < i_lcl)
    )

    data["mr_out_of_control"] = (
        data["moving_range"] > mr_ucl
    )

    statistics = {
        "n": len(data),
        "mean": mean,
        "std": std,
        "average_moving_range": mr_bar,
        "estimated_sigma": estimated_sigma,
        "i_lcl": i_lcl,
        "i_center": mean,
        "i_ucl": i_ucl,
        "mr_lcl": mr_lcl,
        "mr_center": mr_bar,
        "mr_ucl": mr_ucl,
        "i_out_of_control_points":
            int(data["i_out_of_control"].sum()),
        "mr_out_of_control_points":
            int(data["mr_out_of_control"].sum()),
    }

    return data, statistics


#%% SPC SPECIAL-CAUSE RULES

def detect_spc_rules(
    data,
    measurement_column,
    center_line,
    ucl,
    lcl,
):
    """
    Detect selected SPC special-cause signals.

    Rule 1: One point outside the 3-sigma control limits.
    Rule 2: Eight consecutive points on the same side of the center line.
    Rule 3: Six consecutive points continuously increasing or decreasing.
    """

    result = data.copy()

    values = (
        result[measurement_column]
        .reset_index(drop=True)
    )

    # Rule 1
    result["rule_1"] = (
        (values > ucl)
        | (values < lcl)
    )

    # Rule 2
    result["rule_2"] = False

    for i in range(7, len(values)):

        window = values.iloc[i - 7:i + 1]

        if (
            (window > center_line).all()
            or (window < center_line).all()
        ):
            result.loc[i - 7:i, "rule_2"] = True

    # Rule 3
    result["rule_3"] = False

    for i in range(5, len(values)):

        window = values.iloc[i - 5:i + 1]
        differences = window.diff().dropna()

        if (
            (differences > 0).all()
            or (differences < 0).all()
        ):
            result.loc[i - 5:i, "rule_3"] = True

    result["special_cause_signal"] = (
        result["rule_1"]
        | result["rule_2"]
        | result["rule_3"]
    )

    return result


#%% I-MR PLOT

def plot_imr_chart(
    data,
    stats,
    measurement_column,
    measurement_label,
    title,
    output_path,
    lsl=None,
    usl=None,
):
    """Create and save an Individuals-Moving Range control chart."""

    plot_data = data.copy()

    plot_data["observation"] = range(
        1,
        len(plot_data) + 1,
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    # Individuals chart
    ax1 = axes[0]

    ax1.plot(
        plot_data["observation"],
        plot_data[measurement_column],
        linewidth=0.7,
        label="Measurement",
    )

    ax1.axhline(
        stats["i_center"],
        linestyle="-",
        linewidth=1.5,
        label="Center Line",
    )

    ax1.axhline(
        stats["i_ucl"],
        linestyle="--",
        linewidth=1.5,
        label="UCL",
    )

    ax1.axhline(
        stats["i_lcl"],
        linestyle="--",
        linewidth=1.5,
        label="LCL",
    )

    if usl is not None:
        ax1.axhline(
            usl,
            linestyle=":",
            linewidth=1.5,
            label="USL",
        )

    if lsl is not None:
        ax1.axhline(
            lsl,
            linestyle=":",
            linewidth=1.5,
            label="LSL",
        )

    ooc_i = plot_data[
        plot_data["i_out_of_control"]
    ]

    ax1.scatter(
        ooc_i["observation"],
        ooc_i[measurement_column],
        marker="x",
        s=20,
        linewidths=2,
        label="Out of Control",
        zorder=5,
    )

    ax1.set_title(
        f"{title} — Individuals Chart"
    )

    ax1.set_ylabel(measurement_label)

    ax1.grid(alpha=0.25)
    ax1.legend(loc="best", ncol=3)

    # Moving Range chart
    ax2 = axes[1]

    ax2.plot(
        plot_data["observation"],
        plot_data["moving_range"],
        linewidth=0.7,
        label="Moving Range",
    )

    ax2.axhline(
        stats["mr_center"],
        linestyle="-",
        linewidth=1.5,
        label="MR Center",
    )

    ax2.axhline(
        stats["mr_ucl"],
        linestyle="--",
        linewidth=1.5,
        label="MR UCL",
    )

    ax2.axhline(
        stats["mr_lcl"],
        linestyle="--",
        linewidth=1.5,
        label="MR LCL",
    )

    ooc_mr = plot_data[
        plot_data["mr_out_of_control"]
    ]

    ax2.scatter(
        ooc_mr["observation"],
        ooc_mr["moving_range"],
        marker="x",
        s=20,
        linewidths=2,
        label="Out of Control",
        zorder=5,
    )

    ax2.set_title(
        f"{title} — Moving Range Chart"
    )

    ax2.set_xlabel("Production Observation")
    ax2.set_ylabel("Moving Range")

    ax2.grid(alpha=0.25)
    ax2.legend(loc="best", ncol=3)

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


#%% MACHINE-LEVEL SPC ANALYSIS

def analyze_ctq(
    data,
    measurement_column,
    ctq_name,
    measurement_label,
    file_name,
    lsl=None,
    usl=None,
):
    """
    Run machine-level SPC analysis for one CTQ.

    For each CNC machine:
        - Calculate I-MR statistics
        - Apply SPC special-cause rules
        - Count specification breaches
        - Plot and save the I-MR control chart
    """

    imr_results = []
    rule_results = []
    breach_results = []

    for machine_id, machine_data in data.groupby("machine_id"):

        machine_data = (
            machine_data
            .sort_values("part_number")
            .reset_index(drop=True)
        )

        # Calculate I-MR statistics
        analyzed_data, stats = calculate_imr_statistics(
            machine_data,
            measurement_column,
        )

        # Apply SPC rules
        rule_data = detect_spc_rules(
            data=analyzed_data,
            measurement_column=measurement_column,
            center_line=stats["i_center"],
            ucl=stats["i_ucl"],
            lcl=stats["i_lcl"],
        )

        # I-MR summary
        imr_results.append(
            {
                "CTQ": ctq_name,
                "Machine": machine_id,
                "Parts": stats["n"],
                "Mean": stats["mean"],
                "Std": stats["std"],
                "Average Moving Range":
                    stats["average_moving_range"],
                "Estimated Sigma":
                    stats["estimated_sigma"],
                "I-LCL": stats["i_lcl"],
                "I-UCL": stats["i_ucl"],
                "I-Chart OOC Points":
                    stats["i_out_of_control_points"],
                "MR-UCL": stats["mr_ucl"],
                "MR-Chart OOC Points":
                    stats["mr_out_of_control_points"],
            }
        )

        # SPC rule summary
        rule_results.append(
            {
                "CTQ": ctq_name,
                "Machine": machine_id,
                "Rule 1 Points":
                    int(rule_data["rule_1"].sum()),
                "Rule 2 Points":
                    int(rule_data["rule_2"].sum()),
                "Rule 3 Points":
                    int(rule_data["rule_3"].sum()),
                "Total Signal Points":
                    int(
                        rule_data[
                            "special_cause_signal"
                        ].sum()
                    ),
            }
        )

        # Specification breaches
        specification_breaches = pd.Series(
            False,
            index=machine_data.index,
        )

        if usl is not None:
            specification_breaches |= (
                machine_data[measurement_column] > usl
            )

        if lsl is not None:
            specification_breaches |= (
                machine_data[measurement_column] < lsl
            )

        breach_results.append(
            {
                "CTQ": ctq_name,
                "Machine": machine_id,
                "Specification Breaches":
                    int(specification_breaches.sum()),
            }
        )

        # Create and save control chart
        figure_path = (
            FIGURE_DIR
            / f"{machine_id.lower()}_{file_name}_imr.png"
        )

        plot_imr_chart(
            data=analyzed_data,
            stats=stats,
            measurement_column=measurement_column,
            measurement_label=measurement_label,
            title=f"{machine_id} — {ctq_name}",
            lsl=lsl,
            usl=usl,
            output_path=figure_path,
        )

    return (
        pd.DataFrame(imr_results),
        pd.DataFrame(rule_results),
        pd.DataFrame(breach_results),
    )

#%% BORE DIAMETER SPC

bore_data = (
    manufacturing_data[
        [
            "part_number",
            "machine_id",
            "production_date",
            "bore_diameter_mm",
        ]
    ]
    .sort_values(["machine_id", "part_number"])
    .reset_index(drop=True)
)

bore_imr, bore_rules, bore_breaches = analyze_ctq(
    data=bore_data,
    measurement_column="bore_diameter_mm",
    ctq_name="Bore Diameter",
    measurement_label="Bore Diameter (mm)",
    file_name="bore_diameter",
    lsl=24.950,
    usl=25.050,
)

print("\nBORE DIAMETER I-MR SUMMARY")
print(bore_imr.round(6).to_string(index=False))

print("\nBORE DIAMETER SPC RULES")
print(bore_rules.to_string(index=False))

print("\nBORE DIAMETER SPECIFICATION BREACHES")
print(bore_breaches.to_string(index=False))


#%% SURFACE ROUGHNESS SPC

roughness_data = (
    manufacturing_data[
        [
            "part_number",
            "machine_id",
            "production_date",
            "surface_roughness_um",
        ]
    ]
    .sort_values(["machine_id", "part_number"])
    .reset_index(drop=True)
)

roughness_imr, roughness_rules, roughness_breaches = analyze_ctq(
    data=roughness_data,
    measurement_column="surface_roughness_um",
    ctq_name="Surface Roughness",
    measurement_label="Surface Roughness (μm)",
    file_name="surface_roughness",
    usl=1.60,
)

print("\nSURFACE ROUGHNESS I-MR SUMMARY")
print(roughness_imr.round(6).to_string(index=False))

print("\nSURFACE ROUGHNESS SPC RULES")
print(roughness_rules.to_string(index=False))

print("\nSURFACE ROUGHNESS SPECIFICATION BREACHES")
print(roughness_breaches.to_string(index=False))


#%% FLATNESS SPC

flatness_data = (
    manufacturing_data[
        [
            "part_number",
            "machine_id",
            "production_date",
            "flatness_mm",
        ]
    ]
    .sort_values(["machine_id", "part_number"])
    .reset_index(drop=True)
)

flatness_imr, flatness_rules, flatness_breaches = analyze_ctq(
    data=flatness_data,
    measurement_column="flatness_mm",
    ctq_name="Flatness",
    measurement_label="Flatness (mm)",
    file_name="flatness",
    usl=0.040,
)

print("\nFLATNESS I-MR SUMMARY")
print(flatness_imr.round(6).to_string(index=False))

print("\nFLATNESS SPC RULES")
print(flatness_rules.to_string(index=False))

print("\nFLATNESS SPECIFICATION BREACHES")
print(flatness_breaches.to_string(index=False))


#%% BURR HEIGHT SPC

burr_data = (
    manufacturing_data[
        [
            "part_number",
            "machine_id",
            "production_date",
            "burr_height_mm",
        ]
    ]
    .sort_values(["machine_id", "part_number"])
    .reset_index(drop=True)
)

burr_imr, burr_rules, burr_breaches = analyze_ctq(
    data=burr_data,
    measurement_column="burr_height_mm",
    ctq_name="Burr Height",
    measurement_label="Burr Height (mm)",
    file_name="burr_height",
    usl=0.080,
)

print("\nBURR HEIGHT I-MR SUMMARY")
print(burr_imr.round(6).to_string(index=False))

print("\nBURR HEIGHT SPC RULES")
print(burr_rules.to_string(index=False))

print("\nBURR HEIGHT SPECIFICATION BREACHES")
print(burr_breaches.to_string(index=False))


#%% COMBINE AND SAVE SPC RESULTS

imr_summary = pd.concat(
    [
        bore_imr,
        roughness_imr,
        flatness_imr,
        burr_imr,
    ],
    ignore_index=True,
)

spc_rule_summary = pd.concat(
    [
        bore_rules,
        roughness_rules,
        flatness_rules,
        burr_rules,
    ],
    ignore_index=True,
)

specification_breach_summary = pd.concat(
    [
        bore_breaches,
        roughness_breaches,
        flatness_breaches,
        burr_breaches,
    ],
    ignore_index=True,
)

imr_summary.to_csv(
    OUTPUT_DIR / "spc_imr_summary.csv",
    index=False,
)

spc_rule_summary.to_csv(
    OUTPUT_DIR / "spc_rule_summary.csv",
    index=False,
)

specification_breach_summary.to_csv(
    OUTPUT_DIR / "spc_specification_breaches.csv",
    index=False,
)