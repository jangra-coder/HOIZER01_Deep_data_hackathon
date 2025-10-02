#!/usr/bin/env python3
"""
Simple runner script for enhanced EDA analysis.
Executes the enhanced analysis on your existing outputs.

Usage:
    python run_enhanced_analysis.py
"""

import subprocess
import sys
import os

def run_enhanced_analysis():
    """Run the enhanced EDA analysis."""
    try:
        # Check if outputs directory exists
        if not os.path.exists("outputs"):
            print("Error: 'outputs' directory not found!")
            print("Please run analytics2.py first to generate the base analysis.")
            return False
        
        # Check for required files
        required_files = [
            "outputs/modification/raw_no_duplicates.csv",
            "outputs/modification/cleaned_winsorized_data.csv",
            "outputs/detected_columns_post_prep.json"
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            print("Error: Missing required files:")
            for f in missing_files:
                print(f"  - {f}")
            print("\nPlease run analytics2.py first to generate the base analysis.")
            return False
        
        print("Starting Enhanced EDA Analysis...")
        print("=" * 50)
        
        # Run the enhanced analysis
        result = subprocess.run([
            sys.executable, 
            "enhanced_eda_analysis.py", 
            "--output", "outputs"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Enhanced analysis completed successfully!")
            print("\n📊 Generated Analysis Reports:")
            print("=" * 50)
            
            # List generated directories and key files
            analysis_dirs = [
                ("original_vs_modified", "Original vs Modified Data Comparison"),
                ("temporal_analysis", "Temporal Trends & Policy Period Analysis"),
                ("country_clustering", "Country Climate Profiles & Clustering"),
                ("policy_dashboard", "Executive Summary & Policy Insights")
            ]
            
            for dir_name, description in analysis_dirs:
                dir_path = os.path.join("outputs", dir_name)
                if os.path.exists(dir_path):
                    print(f"\n📁 {description}")
                    print(f"   Location: outputs/{dir_name}/")
                    
                    # List key files in each directory
                    files = os.listdir(dir_path)
                    for file in sorted(files):
                        if file.endswith(('.png', '.csv', '.txt')):
                            print(f"   - {file}")
            
            print("\n🎯 Next Steps:")
            print("=" * 20)
            print("1. Review executive summary: outputs/policy_dashboard/executive_summary.txt")
            print("2. Examine key visualizations in each analysis folder")
            print("3. Use insights to develop policy recommendations")
            print("4. Create presentation materials from generated charts")
            
        else:
            print("❌ Enhanced analysis failed!")
            print("\nError Output:")
            print(result.stderr)
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error running enhanced analysis: {e}")
        return False

if __name__ == "__main__":
    success = run_enhanced_analysis()
    sys.exit(0 if success else 1)