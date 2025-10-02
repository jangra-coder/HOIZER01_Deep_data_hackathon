#!/usr/bin/env python3
"""
Usage:
    python enhanced_eda_analysis.py --output outputs

Features:
- Interactive dashboard components
- Original vs Modified data comparisons
- Temporal deep-dive analysis
- Regional clustering insights
- Policy-focused visualizations
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Enhanced styling
plt.style.use('seaborn-v0_8')
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 11,
    'figure.dpi': 120,
    'axes.prop_cycle': plt.cycler('color', ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83'])
})

def load_analysis_data(output_dir):
    """Load all necessary data from the outputs directory."""
    data = {}
    
    # Load original data (before modifications)
    mod_dir = os.path.join(output_dir, "modification")
    original_path = os.path.join(mod_dir, "raw_no_duplicates.csv")
    cleaned_path = os.path.join(mod_dir, "cleaned_winsorized_data.csv")
    
    if os.path.exists(original_path):
        data['original'] = pd.read_csv(original_path)
    if os.path.exists(cleaned_path):
        data['cleaned'] = pd.read_csv(cleaned_path)
    
    # Load metadata
    col_map_path = os.path.join(output_dir, "detected_columns_post_prep.json")
    if os.path.exists(col_map_path):
        with open(col_map_path, 'r') as f:
            data['col_map'] = json.load(f)
    
    # Load basic stats
    stats_path = os.path.join(output_dir, "basic_stats.csv")
    if os.path.exists(stats_path):
        data['stats'] = pd.read_csv(stats_path, index_col=0)
    
    return data

def create_original_vs_modified_comparison(data, output_dir):
    """Generate comprehensive original vs modified data comparison."""
    if 'original' not in data or 'cleaned' not in data:
        print("Original or cleaned data not found. Skipping comparison.")
        return
    
    original_df = data['original']
    cleaned_df = data['cleaned']
    col_map = data.get('col_map', {})
    
    # Create comparison directory
    comparison_dir = os.path.join(output_dir, "original_vs_modified")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # 1. Basic statistics comparison
    create_stats_comparison(original_df, cleaned_df, comparison_dir)
    
    # 2. Distribution comparisons for key variables
    create_distribution_comparison(original_df, cleaned_df, col_map, comparison_dir)
    
    # 3. Correlation matrix comparison
    create_correlation_comparison(original_df, cleaned_df, comparison_dir)
    
    # 4. Outlier impact analysis
    create_outlier_impact_analysis(original_df, cleaned_df, col_map, comparison_dir)

def create_stats_comparison(original_df, cleaned_df, comparison_dir):
    """Compare basic statistics between original and cleaned data."""
    # Get numeric columns common to both datasets
    orig_numeric = original_df.select_dtypes(include=[np.number]).columns
    clean_numeric = cleaned_df.select_dtypes(include=[np.number]).columns
    common_numeric = list(set(orig_numeric) & set(clean_numeric))
    common_numeric = [c for c in common_numeric if not c.endswith('_is_outlier')]
    
    if not common_numeric:
        return
    
    # Calculate statistics
    orig_stats = original_df[common_numeric].describe()
    clean_stats = cleaned_df[common_numeric].describe()
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Original vs Modified Data: Statistical Comparison', fontsize=16, fontweight='bold')
    
    # Mean comparison
    means_df = pd.DataFrame({
        'Original': orig_stats.loc['mean'],
        'Modified': clean_stats.loc['mean']
    })
    means_df.plot(kind='bar', ax=axes[0,0], alpha=0.8)
    axes[0,0].set_title('Mean Values Comparison')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Standard deviation comparison
    std_df = pd.DataFrame({
        'Original': orig_stats.loc['std'],
        'Modified': clean_stats.loc['std']
    })
    std_df.plot(kind='bar', ax=axes[0,1], alpha=0.8)
    axes[0,1].set_title('Standard Deviation Comparison')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Min-Max range comparison
    range_df = pd.DataFrame({
        'Original_Range': orig_stats.loc['max'] - orig_stats.loc['min'],
        'Modified_Range': clean_stats.loc['max'] - clean_stats.loc['min']
    })
    range_df.plot(kind='bar', ax=axes[1,0], alpha=0.8)
    axes[1,0].set_title('Value Range Comparison')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Percentage change in means
    pct_change = ((clean_stats.loc['mean'] - orig_stats.loc['mean']) / orig_stats.loc['mean'] * 100).fillna(0)
    axes[1,1].bar(range(len(pct_change)), pct_change.values, alpha=0.8)
    axes[1,1].set_title('Percentage Change in Means (%)')
    axes[1,1].set_xticks(range(len(pct_change)))
    axes[1,1].set_xticklabels(pct_change.index, rotation=45)
    axes[1,1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'statistical_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed comparison table
    comparison_table = pd.concat([
        orig_stats.add_suffix('_original'),
        clean_stats.add_suffix('_modified'),
        pct_change.to_frame('pct_change_mean')
    ], axis=1)
    comparison_table.to_csv(os.path.join(comparison_dir, 'detailed_stats_comparison.csv'))

def create_distribution_comparison(original_df, cleaned_df, col_map, comparison_dir):
    """Compare distributions of key climate variables."""
    key_vars = ['avg_temp', 'co2', 'sea_level', 'renewable', 'extreme_events']
    actual_cols = [col_map.get(var) for var in key_vars if col_map.get(var)]
    actual_cols = [col for col in actual_cols if col in original_df.columns and col in cleaned_df.columns]
    
    if not actual_cols:
        return
    
    n_cols = min(len(actual_cols), 4)  # Max 4 variables per plot
    fig, axes = plt.subplots(2, n_cols, figsize=(5*n_cols, 10))
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle('Distribution Comparison: Original vs Modified Data', fontsize=16, fontweight='bold')
    
    for i, col in enumerate(actual_cols[:n_cols]):
        # Histograms
        axes[0,i].hist(original_df[col].dropna(), bins=30, alpha=0.7, label='Original', density=True)
        axes[0,i].hist(cleaned_df[col].dropna(), bins=30, alpha=0.7, label='Modified', density=True)
        axes[0,i].set_title(f'{col} - Distributions')
        axes[0,i].legend()
        axes[0,i].set_ylabel('Density')
        
        # Box plots
        box_data = [original_df[col].dropna(), cleaned_df[col].dropna()]
        axes[1,i].boxplot(box_data, labels=['Original', 'Modified'])
        axes[1,i].set_title(f'{col} - Box Plots')
        axes[1,i].set_ylabel('Values')
    
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'distribution_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_correlation_comparison(original_df, cleaned_df, comparison_dir):
    """Compare correlation matrices between original and cleaned data."""
    # Get common numeric columns
    orig_numeric = original_df.select_dtypes(include=[np.number]).columns
    clean_numeric = cleaned_df.select_dtypes(include=[np.number]).columns
    common_numeric = list(set(orig_numeric) & set(clean_numeric))
    common_numeric = [c for c in common_numeric if not c.endswith('_is_outlier')][:8]  # Limit for readability
    
    if len(common_numeric) < 3:
        return
    
    # Calculate correlations
    orig_corr = original_df[common_numeric].corr()
    clean_corr = cleaned_df[common_numeric].corr()
    
    # Create side-by-side heatmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Original correlation heatmap
    sns.heatmap(orig_corr, annot=True, cmap='RdBu_r', center=0, 
                square=True, ax=ax1, cbar_kws={'shrink': 0.8})
    ax1.set_title('Original Data Correlations', fontsize=14, fontweight='bold')
    
    # Modified correlation heatmap
    sns.heatmap(clean_corr, annot=True, cmap='RdBu_r', center=0, 
                square=True, ax=ax2, cbar_kws={'shrink': 0.8})
    ax2.set_title('Modified Data Correlations', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'correlation_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate correlation differences
    corr_diff = clean_corr - orig_corr
    
    # Plot correlation differences
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_diff, annot=True, cmap='RdBu_r', center=0, 
                square=True, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title('Correlation Changes (Modified - Original)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'correlation_changes.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_outlier_impact_analysis(original_df, cleaned_df, col_map, comparison_dir):
    """Analyze the impact of outlier treatment on key relationships."""
    key_relationships = [
        ('co2', 'renewable'),
        ('avg_temp', 'sea_level'),
        ('extreme_events', 'avg_temp')
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Outlier Treatment Impact on Key Relationships', fontsize=16, fontweight='bold')
    
    for i, (x_key, y_key) in enumerate(key_relationships):
        x_col = col_map.get(x_key)
        y_col = col_map.get(y_key)
        
        if x_col and y_col and x_col in original_df.columns and y_col in original_df.columns:
            # Original data scatter
            orig_data = original_df[[x_col, y_col]].dropna()
            clean_data = cleaned_df[[x_col, y_col]].dropna()
            
            axes[i].scatter(orig_data[x_col], orig_data[y_col], alpha=0.6, s=30, label='Original', color='red')
            axes[i].scatter(clean_data[x_col], clean_data[y_col], alpha=0.6, s=30, label='Modified', color='blue')
            
            # Add trend lines
            if len(orig_data) > 1:
                z_orig = np.polyfit(orig_data[x_col], orig_data[y_col], 1)
                p_orig = np.poly1d(z_orig)
                axes[i].plot(orig_data[x_col], p_orig(orig_data[x_col]), "r--", alpha=0.8, linewidth=2)
            
            if len(clean_data) > 1:
                z_clean = np.polyfit(clean_data[x_col], clean_data[y_col], 1)
                p_clean = np.poly1d(z_clean)
                axes[i].plot(clean_data[x_col], p_clean(clean_data[x_col]), "b--", alpha=0.8, linewidth=2)
            
            axes[i].set_xlabel(x_col)
            axes[i].set_ylabel(y_col)
            axes[i].set_title(f'{y_col} vs {x_col}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'outlier_impact_relationships.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_temporal_deep_dive(data, output_dir):
    """Enhanced temporal analysis focusing on policy-relevant periods."""
    if 'cleaned' not in data:
        return
    
    df = data['cleaned']
    col_map = data.get('col_map', {})
    year_col = col_map.get('year')
    
    if not year_col or year_col not in df.columns:
        return
    
    temporal_dir = os.path.join(output_dir, "temporal_analysis")
    os.makedirs(temporal_dir, exist_ok=True)
    
    # Key policy periods
    policy_periods = {
        'Kyoto Protocol': (1997, 2005),
        'Paris Agreement': (2015, 2020),
        'Post-Paris': (2020, 2025)
    }
    
    # Analyze trends for key variables
    key_vars = ['co2', 'renewable', 'avg_temp', 'extreme_events']
    available_vars = [col_map.get(var) for var in key_vars if col_map.get(var) and col_map.get(var) in df.columns]
    
    if not available_vars:
        return
    
    # Create temporal trends with policy period highlights
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    fig.suptitle('Temporal Trends Analysis: Climate Variables Over Time', fontsize=16, fontweight='bold')
    
    for i, var in enumerate(available_vars[:4]):
        if i < len(axes):
            # Calculate yearly averages
            yearly_data = df.groupby(year_col)[var].mean().reset_index()
            
            axes[i].plot(yearly_data[year_col], yearly_data[var], marker='o', linewidth=2, markersize=4)
            
            # Highlight policy periods
            for period_name, (start_year, end_year) in policy_periods.items():
                period_data = yearly_data[(yearly_data[year_col] >= start_year) & (yearly_data[year_col] <= end_year)]
                if not period_data.empty:
                    axes[i].axvspan(start_year, end_year, alpha=0.2, label=period_name)
            
            axes[i].set_title(f'{var} Trends Over Time')
            axes[i].set_xlabel('Year')
            axes[i].set_ylabel(var)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(temporal_dir, 'policy_period_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate trend changes by period
    create_period_comparison_analysis(df, col_map, policy_periods, temporal_dir)

def create_period_comparison_analysis(df, col_map, policy_periods, temporal_dir):
    """Compare average values and trends across different policy periods."""
    year_col = col_map.get('year')
    if not year_col:
        return
    
    key_vars = ['co2', 'renewable', 'avg_temp', 'extreme_events']
    available_vars = [col_map.get(var) for var in key_vars if col_map.get(var) and col_map.get(var) in df.columns]
    
    period_comparison = []
    
    for period_name, (start_year, end_year) in policy_periods.items():
        period_data = df[(df[year_col] >= start_year) & (df[year_col] <= end_year)]
        
        if not period_data.empty:
            period_stats = {}
            period_stats['Period'] = period_name
            period_stats['Years'] = f"{start_year}-{end_year}"
            period_stats['Sample_Size'] = len(period_data)
            
            for var in available_vars:
                period_stats[f'{var}_mean'] = period_data[var].mean()
                period_stats[f'{var}_std'] = period_data[var].std()
            
            period_comparison.append(period_stats)
    
    if period_comparison:
        comparison_df = pd.DataFrame(period_comparison)
        comparison_df.to_csv(os.path.join(temporal_dir, 'policy_period_comparison.csv'), index=False)
        
        # Visualize period comparisons
        create_period_comparison_plots(comparison_df, available_vars, temporal_dir)

def create_period_comparison_plots(comparison_df, available_vars, temporal_dir):
    """Create bar plots comparing different policy periods."""
    n_vars = len(available_vars)
    if n_vars == 0:
        return
    
    fig, axes = plt.subplots(1, min(n_vars, 3), figsize=(5*min(n_vars, 3), 6))
    if min(n_vars, 3) == 1:
        axes = [axes]
    
    fig.suptitle('Policy Period Comparison: Average Values', fontsize=14, fontweight='bold')
    
    for i, var in enumerate(available_vars[:3]):
        mean_col = f'{var}_mean'
        if mean_col in comparison_df.columns:
            bars = axes[i].bar(comparison_df['Period'], comparison_df[mean_col], alpha=0.8)
            axes[i].set_title(f'Average {var}')
            axes[i].set_ylabel(f'{var} (mean)')
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, comparison_df[mean_col]):
                if not pd.isna(value):
                    axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.01, 
                                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(temporal_dir, 'policy_period_averages.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_country_clustering_analysis(data, output_dir):
    """Perform country clustering based on climate profiles."""
    if 'cleaned' not in data:
        return
    
    df = data['cleaned']
    col_map = data.get('col_map', {})
    country_col = col_map.get('country')
    
    if not country_col or country_col not in df.columns:
        return
    
    clustering_dir = os.path.join(output_dir, "country_clustering")
    os.makedirs(clustering_dir, exist_ok=True)
    
    # Aggregate by country (latest year or average)
    climate_vars = ['avg_temp', 'co2', 'renewable', 'forest_area', 'extreme_events']
    available_vars = [col_map.get(var) for var in climate_vars if col_map.get(var) and col_map.get(var) in df.columns]
    
    if len(available_vars) < 3:
        return
    
    # Create country profiles (use most recent year or average)
    year_col = col_map.get('year')
    if year_col and year_col in df.columns:
        # Use most recent year for each country
        latest_year = df.groupby(country_col)[year_col].max().reset_index()
        country_profiles = df.merge(latest_year, on=[country_col, year_col])
    else:
        # Use country averages
        country_profiles = df.groupby(country_col)[available_vars].mean().reset_index()
    
    # Create country comparison visualizations
    create_country_profile_matrix(country_profiles, country_col, available_vars, clustering_dir)
    create_country_radar_charts(country_profiles, country_col, available_vars, clustering_dir)

def create_country_profile_matrix(df, country_col, climate_vars, clustering_dir):
    """Create a matrix showing country climate profiles."""
    # Get top 15 countries by data availability
    country_counts = df[country_col].value_counts().head(15)
    top_countries = country_counts.index.tolist()
    
    profile_data = df[df[country_col].isin(top_countries)].groupby(country_col)[climate_vars].mean()
    
    # Normalize data for better comparison (0-1 scale)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    profile_normalized = pd.DataFrame(
        scaler.fit_transform(profile_data),
        index=profile_data.index,
        columns=profile_data.columns
    )
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(profile_normalized.T, annot=True, cmap='RdYlBu_r', center=0.5, 
                square=False, cbar_kws={'shrink': 0.8})
    plt.title('Country Climate Profile Matrix (Normalized 0-1)', fontsize=14, fontweight='bold')
    plt.xlabel('Countries')
    plt.ylabel('Climate Variables')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(clustering_dir, 'country_profile_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save raw profiles
    profile_data.to_csv(os.path.join(clustering_dir, 'country_climate_profiles.csv'))

def create_country_radar_charts(df, country_col, climate_vars, clustering_dir):
    """Create radar charts for top countries."""
    from math import pi
    
    # Get top 6 countries for radar charts
    country_counts = df[country_col].value_counts().head(6)
    top_countries = country_counts.index.tolist()
    
    profile_data = df[df[country_col].isin(top_countries)].groupby(country_col)[climate_vars].mean()
    
    # Normalize for radar chart
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    profile_normalized = pd.DataFrame(
        scaler.fit_transform(profile_data),
        index=profile_data.index,
        columns=profile_data.columns
    )
    
    # Create radar chart
    angles = [n / len(climate_vars) * 2 * pi for n in range(len(climate_vars))]
    angles += angles[:1]  # Complete the circle
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), subplot_kw=dict(projection='polar'))
    axes = axes.flatten()
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    for i, country in enumerate(top_countries):
        values = profile_normalized.loc[country].tolist()
        values += values[:1]  # Complete the circle
        
        axes[i].plot(angles, values, 'o-', linewidth=2, label=country, color=colors[i])
        axes[i].fill(angles, values, alpha=0.25, color=colors[i])
        axes[i].set_xticks(angles[:-1])
        axes[i].set_xticklabels(climate_vars)
        axes[i].set_ylim(0, 1)
        axes[i].set_title(country, fontweight='bold')
        axes[i].grid(True)
    
    plt.suptitle('Country Climate Profiles - Radar Charts', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(clustering_dir, 'country_radar_charts.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_policy_dashboard_summary(data, output_dir):
    """Create a comprehensive policy dashboard summary."""
    dashboard_dir = os.path.join(output_dir, "policy_dashboard")
    os.makedirs(dashboard_dir, exist_ok=True)
    
    # Create executive summary report
    create_executive_summary(data, dashboard_dir)
    
    # Create key insights visualization
    create_key_insights_visual(data, dashboard_dir)

def create_executive_summary(data, dashboard_dir):
    """Generate executive summary for policymakers."""
    if 'cleaned' not in data or 'stats' not in data:
        return
    
    df = data['cleaned']
    col_map = data.get('col_map', {})
    
    summary = []
    summary.append("=" * 80)
    summary.append("EXECUTIVE SUMMARY: CLIMATE DATA ANALYSIS FOR POLICY MAKERS")
    summary.append("=" * 80)
    summary.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append(f"Dataset Size: {len(df):,} observations")
    summary.append(f"Time Period: {df[col_map.get('year')].min():.0f}-{df[col_map.get('year')].max():.0f}" if col_map.get('year') else "Time period unknown")
    summary.append("")
    
    # Key findings
    summary.append("KEY FINDINGS:")
    summary.append("-" * 40)
    
    # Temperature insights
    if col_map.get('avg_temp'):
        temp_col = col_map['avg_temp']
        avg_temp = df[temp_col].mean()
        temp_std = df[temp_col].std()
        summary.append(f"• Average Global Temperature: {avg_temp:.1f}°C (±{temp_std:.1f}°C)")
        high_temp_countries = len(df[df[temp_col] > avg_temp + temp_std])
        summary.append(f"• Countries with extreme temperatures: {high_temp_countries} ({high_temp_countries/len(df)*100:.1f}%)")
    
    # CO2 insights
    if col_map.get('co2'):
        co2_col = col_map['co2']
        avg_co2 = df[co2_col].mean()
        co2_std = df[co2_col].std()
        summary.append(f"• Average CO2 Emissions: {avg_co2:.2f} tons/capita")
        high_emitters = len(df[df[co2_col] > avg_co2 + co2_std])
        summary.append(f"• High-emission observations: {high_emitters} ({high_emitters/len(df)*100:.1f}%)")
    
    # Renewable energy insights
    if col_map.get('renewable'):
        renewable_col = col_map['renewable']
        avg_renewable = df[renewable_col].mean()
        summary.append(f"• Average Renewable Energy Adoption: {avg_renewable:.1f}%")
        low_renewable = len(df[df[renewable_col] < avg_renewable * 0.5])
        summary.append(f"• Countries with low renewable adoption (<{avg_renewable*0.5:.1f}%): {low_renewable}")
    
    summary.append("")
    summary.append("POLICY PRIORITIES:")
    summary.append("-" * 40)
    summary.append("1. IMMEDIATE: Target climate adaptation for extreme temperature countries")
    summary.append("2. SHORT-term: Accelerate renewable energy in low-adoption regions")
    summary.append("3. MEDIUM-term: Implement comprehensive decarbonization strategies")
    summary.append("4. LONG-term: Develop climate-resilient infrastructure globally")
    
    # Save summary
    with open(os.path.join(dashboard_dir, 'executive_summary.txt'), 'w') as f:
        f.write('\n'.join(summary))
    
    print('\n'.join(summary))

def create_key_insights_visual(data, dashboard_dir):
    """Create a comprehensive key insights visualization."""
    if 'cleaned' not in data:
        return
    
    df = data['cleaned']
    col_map = data.get('col_map', {})
    
    fig = plt.figure(figsize=(20, 12))
    
    # Create a 3x3 grid of insights
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Temperature distribution
    if col_map.get('avg_temp'):
        ax1 = fig.add_subplot(gs[0, 0])
        temp_col = col_map['avg_temp']
        ax1.hist(df[temp_col].dropna(), bins=30, alpha=0.7, color='red')
        ax1.set_title('Global Temperature Distribution')
        ax1.set_xlabel('Temperature (°C)')
        ax1.set_ylabel('Frequency')
    
    # 2. CO2 vs Renewable scatter
    if col_map.get('co2') and col_map.get('renewable'):
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.scatter(df[col_map['co2']], df[col_map['renewable']], alpha=0.6, s=20)
        ax2.set_xlabel('CO2 Emissions (tons/capita)')
        ax2.set_ylabel('Renewable Energy (%)')
        ax2.set_title('CO2 vs Renewable Energy')
    
    # 3. Time trend of key variable
    if col_map.get('year') and col_map.get('co2'):
        ax3 = fig.add_subplot(gs[0, 2])
        yearly_co2 = df.groupby(col_map['year'])[col_map['co2']].mean()
        ax3.plot(yearly_co2.index, yearly_co2.values, marker='o')
        ax3.set_title('CO2 Emissions Trend')
        ax3.set_xlabel('Year')
        ax3.set_ylabel('Average CO2 (tons/capita)')
    
    # Add more insights panels as needed...
    
    plt.suptitle('Climate Policy Dashboard - Key Insights', fontsize=20, fontweight='bold', y=0.98)
    plt.savefig(os.path.join(dashboard_dir, 'key_insights_dashboard.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main analysis pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced EDA Analysis - Post Data Preparation")
    parser.add_argument("--output", "-o", default="outputs", help="Output directory containing analysis results")
    args = parser.parse_args()
    
    output_dir = args.output
    
    print("Loading analysis data...")
    data = load_analysis_data(output_dir)
    
    print("Creating original vs modified comparison...")
    create_original_vs_modified_comparison(data, output_dir)
    
    print("Performing temporal deep-dive analysis...")
    create_temporal_deep_dive(data, output_dir)
    
    print("Creating country clustering analysis...")
    create_country_clustering_analysis(data, output_dir)
    
    print("Generating policy dashboard summary...")
    create_policy_dashboard_summary(data, output_dir)
    
    print("Enhanced EDA analysis completed!")
    print(f"Results saved in: {output_dir}")
    print("\nGenerated analysis folders:")
    print("- original_vs_modified/: Comparison between original and cleaned data")
    print("- temporal_analysis/: Policy period analysis and trends")
    print("- country_clustering/: Country climate profiles and clustering")
    print("- policy_dashboard/: Executive summary and key insights")

if __name__ == "__main__":
    main()