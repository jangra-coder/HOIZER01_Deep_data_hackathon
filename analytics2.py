#!/usr/bin/env python3
"""
Usage:
    python climate_data_understanding.py --input path/to/data.csv --output outputs
Produces:
 - textual summary saved as outputs/summary.txt
 - CSV reports: outputs/missing_report.csv, outputs/basic_stats.csv
 - saved plots in outputs/ (PNG)
 - preparation artifacts in outputs/modification/
"""

import os
import argparse
import logging
from collections import OrderedDict
from difflib import get_close_matches

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 11,
    "figure.dpi": 120
})

NUMERIC_DTYPES = ["float64", "int64", "float32", "int32"]

def try_read(path):
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext in [".csv", ".txt"]:
        return pd.read_csv(path)
    elif ext in [".xls", ".xlsx"]:
        return pd.read_excel(path)
    else:
        # try csv as fallback
        return pd.read_csv(path)

def find_best_match(col_candidates, choices, cutoff=0.6):

    lowercase_map = {c.lower(): c for c in choices}
    for cand in col_candidates:
        if cand.lower() in lowercase_map:
            return lowercase_map[cand.lower()]
    for cand in col_candidates:
        matches = get_close_matches(cand.lower(), [c.lower() for c in choices], n=1, cutoff=cutoff)
        if matches:
            return lowercase_map[matches[0]]
    return None

def detect_columns(df):
    cols = list(df.columns)
    col_map = {}

    candidates = {
        "year": ["Year", "year"],
        "country": ["Country", "country", "region", "Location"],
        "avg_temp": ["Average Temperature (°C)", "Average Temperature", "Avg Temp", "Temperature", "AvgTemperature", "avg_temp"],
        "co2": ["CO2 Emissions (Tons/Capita)", "CO2 Emissions", "CO2", "CO2 Emissions (tons per capita)"],
        "sea_level": ["Sea Level Rise (mm)", "Sea Level", "SeaLevel", "Sea Level Rise", "Sea_Level"],
        "rainfall": ["Rainfall (mm)", "Rainfall", "Precipitation", "Rainfall_mm"],
        "population": ["Population", "Pop", "Population Total", "Population (Total)"],
        "renewable": ["Renewable Energy (%)", "Renewable Energy", "Renewable", "Renewable_pct", "Renewable (%)"],
        "extreme_events": ["Extreme Weather Events", "Extreme Events", "Extreme_Events", "Extreme Weather"],
        "forest_area": ["Forest Area (%)", "Forest Area", "Forest", "Forest_Area"]
    }

    for key, cand_list in candidates.items():
        found = find_best_match(cand_list, cols, cutoff=0.55)
        col_map[key] = found

    if col_map["year"] is None:
        for c in cols:
            if np.issubdtype(df[c].dtype, np.number):
                vals = df[c].dropna().unique()
                if len(vals) > 0:
                    mn, mx = vals.min(), vals.max()
                    try:
                        if 1800 <= int(mn) <= 2100 and 1800 <= int(mx) <= 2100:
                            col_map["year"] = c
                            break
                    except Exception:
                        pass

    return col_map

def summarize_structure(df, out_dir):
    """Print and save high-level structure: dtypes, non-null counts, top values for categoricals."""
    info = []
    info.append(f"DataFrame shape: {df.shape}")
    info.append("\nColumn dtypes and non-null counts:")
    dtype_counts = df.dtypes.astype(str).to_dict()
    non_null = df.notnull().sum().to_dict()
    for c in df.columns:
        info.append(f" - {c}: dtype={dtype_counts[c]}, non-null={non_null[c]}")
    info.append("\nTop values for categorical/object columns (up to 5):")
    for c in df.select_dtypes(include=['object', 'category']).columns:
        top = df[c].value_counts(dropna=True).head(5).to_dict()
        info.append(f" - {c}: {top}")

    summary_txt = "\n".join(info)
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary_txt)

    print(summary_txt)

def missing_values_report(df, out_dir):
    """Compute and save missing value report (counts & percent)."""
    miss = df.isnull().sum()
    miss_pct = (miss / len(df) * 100).round(2)
    report = pd.DataFrame({"missing_count": miss, "missing_pct": miss_pct})
    report = report.sort_values("missing_count", ascending=False)
    report.to_csv(os.path.join(out_dir, "missing_report.csv"))
    return report

def basic_numeric_stats(df, out_dir):
    """Save basic numeric statistics to CSV and return DataFrame."""
    num_df = df.select_dtypes(include=[np.number])
    stats = num_df.describe().T
    stats["missing_count"] = num_df.isnull().sum()
    stats.to_csv(os.path.join(out_dir, "basic_stats.csv"))
    return stats
def remove_duplicates(df, out_dir):
    """Find and drop duplicate rows, save report."""
    dup_mask = df.duplicated(keep='first')
    dup_count = dup_mask.sum()
    if dup_count > 0:
        duplicates = df[dup_mask].copy()
        duplicates.to_csv(os.path.join(out_dir, "duplicates_removed_sample.csv"), index=False)
    else:
        pd.DataFrame().to_csv(os.path.join(out_dir, "duplicates_removed_sample.csv"), index=False)
    df_no_dup = df.drop_duplicates(keep='first').reset_index(drop=True)
    with open(os.path.join(out_dir, "duplicates_report.txt"), "w") as f:
        f.write(f"duplicates_found={dup_count}\nrows_before={len(df)}\nrows_after={len(df_no_dup)}\n")
    return df_no_dup, dup_count

def coerce_numeric_like_columns(df, out_dir):
    """Attempt to convert object columns that look numeric into numeric dtype (remove commas)."""
    coerced = []
    for c in df.select_dtypes(include=['object']).columns:
        sample = df[c].dropna().astype(str).head(200).str.replace(",", "").str.strip()
        if sample.shape[0] > 0 and sample.str.match(r"^-?\d+(\.\d+)?$").sum() > max(3, int(0.6*len(sample))):
            try:
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "").str.strip(), errors='coerce')
                coerced.append(c)
            except Exception:
                pass
    with open(os.path.join(out_dir, "coerced_columns.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(coerced))
    return df, coerced

def normalize_percent_column(df, col, out_dir):
    if col is None or col not in df.columns:
        return df, None
    s = df[col].dropna()
    if s.empty:
        return df, None
    maxv = s.max()
    minv = s.min()
    action = None
    if maxv <= 1.01 and minv >= 0:
        df[col] = df[col].astype(float) * 100.0
        action = "scaled_0-1_to_0-100"
    if df[col].dtype == object:
        try:
            df[col] = df[col].astype(str).str.replace("%", "").str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')
            action = "stripped_percent_signs"
        except Exception:
            pass
    # final clamp (0-100)
    df[col] = pd.to_numeric(df[col], errors='coerce')
    if df[col].dropna().max() <= 1.01:
        df[col] = df[col] * 100.0
        action = (action or "") + "|ensure_scaled"
    with open(os.path.join(out_dir, "percent_normalization.txt"), "w", encoding="utf-8") as f:
        f.write(f"column={col}\naction={action}\nmin={df[col].min()}\nmax={df[col].max()}\n")
    return df, action

def clean_population_column(df, col, out_dir):
    """Remove commas and coerce population to numeric."""
    if col is None or col not in df.columns:
        return df, None
    try:
        df[col] = df[col].astype(str).str.replace(",", "").str.strip()
        df[col] = pd.to_numeric(df[col], errors='coerce')
        action = "cleaned_commas_and_coerced"
    except Exception:
        action = None
    with open(os.path.join(out_dir, "population_cleaning.txt"), "w", encoding="utf-8") as f:
        f.write(f"column={col}\naction={action}\nmissing_after={df[col].isnull().sum()}\n")
    return df, action

def detect_flag_outliers(df, numeric_cols, out_dir, iqr_multiplier=1.5):
    """
    Detect outliers via IQR per column; create boolean flags column+'_is_outlier'.
    Save outlier summary CSV with counts + example rows.
    """
    outlier_summary = []
    outlier_examples = []
    for c in numeric_cols:
        s = df[c].dropna()
        if s.shape[0] < 10:
            outlier_summary.append({"column": c, "n_outliers": 0, "pct_outliers": 0.0})
            continue
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        mask = (df[c] < lower) | (df[c] > upper)
        n_out = int(mask.sum())
        pct_out = round(100.0 * n_out / len(df), 3)
        df[c + "_is_outlier"] = mask
        outlier_summary.append({"column": c, "n_outliers": n_out, "pct_outliers": pct_out, "lower": lower, "upper": upper})
        # save up to 20 example rows flagged as outliers
        if n_out > 0:
            ex = df.loc[mask, :].head(20)
            ex_sample_path = os.path.join(out_dir, f"outliers_examples_{c}.csv")
            ex.to_csv(ex_sample_path, index=False)
            outlier_examples.append(ex_sample_path)
    out_df = pd.DataFrame(outlier_summary).sort_values("n_outliers", ascending=False)
    out_df.to_csv(os.path.join(out_dir, "outlier_summary.csv"), index=False)
    return df, out_df

def winsorize_columns(df, numeric_cols, out_dir, iqr_multiplier=1.5):
    df_w = df.copy()
    caps = []
    for c in numeric_cols:
        s = df[c].dropna()
        if s.shape[0] < 5:
            continue
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        df_w[c] = df_w[c].clip(lower=lower, upper=upper)
        caps.append({"column": c, "lower": lower, "upper": upper})
    pd.DataFrame(caps).to_csv(os.path.join(out_dir, "winsorize_caps.csv"), index=False)
    df_w.to_csv(os.path.join(out_dir, "cleaned_winsorized_data.csv"), index=False)
    return df_w, caps

def save_modification_reports(original_df, cleaned_df, modification_dir):
    report = {
        "rows_before": len(original_df),
        "rows_after": len(cleaned_df),
        "columns_before": original_df.shape[1],
        "columns_after": cleaned_df.shape[1]
    }
    with open(os.path.join(modification_dir, "modification_summary.txt"), "w", encoding="utf-8") as f:
        for k, v in report.items():
            f.write(f"{k}={v}\n")
    return report

# Plotting functions (matplotlib)

def plot_missing_bar(report, out_path):
    """Bar chart of missing counts (top 30 columns)."""
    top = report.head(30).sort_values("missing_count", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, len(top)*0.25)))
    ax.barh(top.index, top["missing_count"])
    ax.set_xlabel("Missing values (count)")
    ax.set_title("Missing Values per Column (top 30)")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def plot_histograms(df, numeric_cols, out_dir, bins=30, prefix=""):
    """Plot histograms for numeric columns in a grid. Saves multiple figures if many columns."""
    cols = numeric_cols
    per_fig = 6
    for i in range(0, len(cols), per_fig):
        subset = cols[i:i+per_fig]
        n = len(subset)
        rows = int(np.ceil(n / 2))
        fig, axes = plt.subplots(rows, 2, figsize=(12, rows*3))
        axes = axes.flatten()
        for ax, col in zip(axes, subset):
            data = df[col].dropna()
            if data.empty:
                ax.text(0.5, 0.5, "No data", ha="center")
                continue
            ax.hist(data, bins=bins, edgecolor="k", alpha=0.7)
            ax.set_title(col)
            ax.set_ylabel("Count")
        # remove unused axes
        for j in range(len(subset), len(axes)):
            fig.delaxes(axes[j])
        fig.suptitle(f"Distributions of numeric features {prefix}")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig_path = os.path.join(out_dir, f"histograms_{prefix}_{i//per_fig + 1}.png").replace("//", "/")
        fig.savefig(fig_path)
        plt.close(fig)

def plot_boxplots(df, numeric_cols, out_dir, prefix=""):
    """Boxplots for numeric columns."""
    if len(numeric_cols) == 0:
        return
    per_fig = 8
    for i in range(0, len(numeric_cols), per_fig):
        subset = numeric_cols[i:i+per_fig]
        fig, axes = plt.subplots(len(subset), 1, figsize=(10, len(subset)*1.6))
        if len(subset) == 1:
            axes = [axes]
        for ax, col in zip(axes, subset):
            data = df[col].dropna()
            ax.boxplot(data, vert=False, patch_artist=True)
            ax.set_title(col)
        plt.tight_layout()
        fig_path = os.path.join(out_dir, f"boxplots_{prefix}_{i//per_fig + 1}.png").replace("//", "/")
        fig.savefig(fig_path)
        plt.close(fig)

def plot_correlation_heatmap(df, numeric_cols, out_path, annot_threshold=0.2):
    """Correlation heatmap using matplotlib (annotated)."""
    num_df = df[numeric_cols].copy().dropna(how="all")
    if num_df.shape[1] < 2:
        return
    corr = num_df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    # annotate
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            val = corr.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.title("Correlation matrix (numeric features)")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def plot_time_trends(df, year_col, cols_to_plot, out_path):
    if year_col is None or year_col not in df.columns:
        return
    df_yr = df[[year_col] + cols_to_plot].copy()
    df_yr = df_yr.dropna(subset=[year_col])
    try:
        df_yr[year_col] = df_yr[year_col].astype(int)
    except Exception:
        pass
    grouped = df_yr.groupby(year_col).mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    for col in grouped.columns:
        ax.plot(grouped.index, grouped[col], label=col)
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean value (per-year)")
    ax.set_title("Yearly mean trends (global)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def plot_top_countries_bar(df, country_col, value_col, year_col=None, top_n=10, out_path=None):
    if country_col is None or value_col is None:
        return
    tmp = df[[country_col, value_col]].copy()
    latest_year = None
    if year_col and year_col in df.columns:
        # pick latest year available
        latest_year = df[year_col].dropna().max()
        tmp = df[df[year_col] == latest_year][[country_col, value_col]].copy()
    agg = tmp.groupby(country_col)[value_col].mean(numeric_only=True).sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(agg.index.astype(str), agg.values)
    ax.set_xticklabels(agg.index, rotation=45, ha="right")
    ax.set_title(f"Top {top_n} countries by {value_col}" + (f" (year={latest_year})" if latest_year is not None else ""))
    ax.set_ylabel(value_col)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path)
    plt.close(fig)

def scatter_pair(df, x, y, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    data = df[[x, y]].dropna()
    if data.empty:
        return
    ax.scatter(data[x], data[y], alpha=0.6, s=20)
    # linear fit
    try:
        m, b = np.polyfit(data[x].values, data[y].values, deg=1)
        xs = np.linspace(data[x].min(), data[x].max(), 100)
        ax.plot(xs, m*xs + b, linestyle="--", linewidth=1)
        ax.text(0.02, 0.95, f"y={m:.3f}x+{b:.3f}", transform=ax.transAxes, fontsize=9, va="top")
    except Exception:
        pass
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{y} vs {x}")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def prepare_data(df, col_map, modification_dir):
    os.makedirs(modification_dir, exist_ok=True)
    df0 = df.copy()
    # 1) Duplicates
    df_nd, dup_count = remove_duplicates(df0, modification_dir)

    # 2) Coerce numeric-like columns
    df_nd, coerced = coerce_numeric_like_columns(df_nd, modification_dir)

    # 3) Normalize percent columns (renewable)
    renewable_col = col_map.get("renewable")
    df_nd, percent_action = normalize_percent_column(df_nd, renewable_col, modification_dir)

    # 4) Clean population column
    pop_col = col_map.get("population")
    df_nd, pop_action = clean_population_column(df_nd, pop_col, modification_dir)

    # 5) Detect numeric columns after coercion
    numeric_cols = list(df_nd.select_dtypes(include=[np.number]).columns)
    # remove 'is_outlier' leftover if present
    numeric_cols = [c for c in numeric_cols if not c.endswith("_is_outlier")]

    # 6) Outlier detection (flag columns)
    df_flagged, outlier_summary = detect_flag_outliers(df_nd, numeric_cols, modification_dir)

    # 7) Winsorize numeric columns to produce cleaned dataset
    df_wins, caps = winsorize_columns(df_flagged, numeric_cols, modification_dir)

    # 8) Save both raw-nodup and winsorized versions
    df_nd.to_csv(os.path.join(modification_dir, "raw_no_duplicates.csv"), index=False)
    df_wins.to_csv(os.path.join(modification_dir, "cleaned_winsorized_data.csv"), index=False)

    # 9) Save before/after histograms for a few key numeric columns (if available)
    sample_cols = numeric_cols[:8]  # up to 8
    if sample_cols:
        plot_histograms(df_nd, sample_cols, modification_dir, prefix="before_prep")
        plot_histograms(df_wins, sample_cols, modification_dir, prefix="after_prep")
        plot_boxplots(df_nd, sample_cols, modification_dir, prefix="before_prep")
        plot_boxplots(df_wins, sample_cols, modification_dir, prefix="after_prep")

    # 10) Save modification summary
    save_modification_reports(df0, df_wins, modification_dir)

    # Returns winsorized df for downstream EDA
    return df_wins

def run_analysis(input_path, output_dir, top_n_countries=10):
    os.makedirs(output_dir, exist_ok=True)
    modification_dir = os.path.join(output_dir, "modification")
    os.makedirs(modification_dir, exist_ok=True)

    logging.info("Reading data...")
    df = try_read(input_path)
    logging.info(f"Data read: {df.shape}")

    # Detect important columns early (so preparation can treat them)
    col_map = detect_columns(df)
    pd.Series(col_map).to_json(os.path.join(output_dir, "detected_columns.json"), orient="index", force_ascii=False)
    logging.info("Starting Data Preparation (duplicates, coercion, normalization, outliers)...")
    df_clean = prepare_data(df, col_map, modification_dir)
    logging.info(f"Data Preparation completed. Cleaned shape: {df_clean.shape}")

    summarize_structure(df_clean, output_dir)

    # Missing value report (on cleaned data)
    missing_report = missing_values_report(df_clean, output_dir)
    plot_missing_bar(missing_report, os.path.join(output_dir, "missing_bar.png"))

    # Basic numeric stats (cleaned)
    basic_stats = basic_numeric_stats(df_clean, output_dir)

    # Re-detect columns on cleaned data (to ensure types in later steps)
    col_map = detect_columns(df_clean)
    pd.Series(col_map).to_json(os.path.join(output_dir, "detected_columns_post_prep.json"), orient="index", force_ascii=False)

    # Identify numeric columns for visualization
    numeric_cols = list(df_clean.select_dtypes(include=[np.number]).columns)
    numeric_cols = sorted(list(set([c for c in numeric_cols if not c.endswith("_is_outlier")])))
    with open(os.path.join(output_dir, "numeric_columns.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(numeric_cols))

    # Plots: distributions and boxplots
    plot_histograms(df_clean, numeric_cols, output_dir)
    plot_boxplots(df_clean, numeric_cols, output_dir)

    # Correlation heatmap
    if len(numeric_cols) >= 2:
        plot_correlation_heatmap(df_clean, numeric_cols, os.path.join(output_dir, "correlation_heatmap.png"))

    # Yearly trends: choose up to 4 interesting columns (if present)
    trend_candidates = []
    for key in ["avg_temp", "co2", "sea_level", "renewable", "extreme_events", "rainfall"]:
        if col_map.get(key):
            trend_candidates.append(col_map[key])
    # ensure unique and numeric
    trend_candidates = [c for c in trend_candidates if c in df_clean.columns and np.issubdtype(df_clean[c].dtype, np.number)]
    if len(trend_candidates) >= 1 and col_map.get("year"):
        plot_time_trends(df_clean, col_map["year"], trend_candidates, os.path.join(output_dir, "yearly_trends.png"))

    # Top countries by CO2 if column exists
    if col_map.get("country") and col_map.get("co2"):
        plot_top_countries_bar(df_clean, col_map["country"], col_map["co2"], year_col=col_map.get("year"), top_n=top_n_countries, out_path=os.path.join(output_dir, "top_countries_co2.png"))

    # Some targeted scatter pairs (if columns exist)
    scatter_pairs = [
        ("co2", "renewable"),
        ("avg_temp", "sea_level"),
        ("extreme_events", "avg_temp"),
        ("co2", "forest_area")
    ]
    for x_key, y_key in scatter_pairs:
        xcol = col_map.get(x_key)
        ycol = col_map.get(y_key)
        if xcol and ycol and xcol in df_clean.columns and ycol in df_clean.columns:
            # ensure numeric
            if np.issubdtype(df_clean[xcol].dtype, np.number) and np.issubdtype(df_clean[ycol].dtype, np.number):
                out_path = os.path.join(output_dir, f"scatter_{x_key}_vs_{y_key}.png")
                scatter_pair(df_clean, xcol, ycol, out_path)

    # Final
    summary_lines = []
    summary_lines.append("---- QUICK AUTO-SUMMARY (POST-PREP) ----")
    summary_lines.append(f"Rows: {len(df_clean)}, Columns: {df_clean.shape[1]}")
    summary_lines.append("Top 5 columns with most missing values:")
    summary_lines += [f" - {idx}: {row['missing_count']} missing ({row['missing_pct']}%)" for idx, row in missing_report.head(5).iterrows()]
    summary_lines.append("\nDetected important columns (guesses):")
    for k, v in col_map.items():
        summary_lines.append(f" - {k}: {v}")
    summary_lines.append("\nNumeric columns count: " + str(len(numeric_cols)))

    if len(numeric_cols) >= 2:
        corr = df_clean[numeric_cols].corr().abs().unstack().sort_values(kind="quicksort", ascending=False).drop_duplicates()
        strong = corr[corr < 1.0].head(10)
        summary_lines.append("\nTop absolute correlations (not self):")
        for (c1, c2), val in strong.items():
            summary_lines.append(f" - {c1} vs {c2}: {val:.3f}")
    with open(os.path.join(output_dir, "auto_summary_post_prep.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print("\n".join(summary_lines))

def parse_args():
    parser = argparse.ArgumentParser(description="Climate dataset EDA - Data Understanding (with Data Preparation)")
    parser.add_argument("--input", "-i", required=True, help="Path to input CSV / Excel file")
    parser.add_argument("--output", "-o", default="outputs", help="Output directory for summaries & plots")
    parser.add_argument("--top_n", "-n", type=int, default=10, help="Top N countries to plot for country-level charts")
    return parser.parse_args()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    run_analysis(args.input, args.output, top_n_countries=args.top_n)
