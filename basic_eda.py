
import pandas as pd
import numpy as np

def perform_eda(df: pd.DataFrame):
    """
    Performs basic Exploratory Data Analysis (EDA) on a given pandas DataFrame.
    
    Parameters:
    df (pd.DataFrame): The dataset to analyze.
    
    Returns:
    dict: A dictionary containing the summary metrics for programmatic access.
    """
    print("=" * 60)
    print(" 📊 EXPLORATORY DATA ANALYSIS (EDA) REPORT")
    print("=" * 60)
    
    # 1. Basic Dataset Shape
    print(f"\n[1] DATASET DIMENSIONS")
    print(f"Total Rows    : {df.shape[0]:,}")
    print(f"Total Columns : {df.shape[1]:,}")
    
    # 2. Duplicate Rows
    duplicates = df.duplicated().sum()
    print(f"\n[2] DUPLICATE ROWS")
    print(f"Duplicate Count : {duplicates:,} ({(duplicates / df.shape[0]) * 100:.2f}% of dataset)")
    
    # 3. Column Information & Data Types
    print(f"\n[3] COLUMN DATA TYPES")
    dtype_df = pd.DataFrame({
        'Column Name': df.columns,
        'Non-Null Count': df.notnull().sum(),
        'Data Type': df.dtypes.values
    }).reset_index(drop=True)
    print(dtype_df.to_string(index=False))
    
    # 4. Missing Values Analysis
    print(f"\n[4] MISSING VALUES ANALYSIS")
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing_count,
        'Missing Percentage (%)': missing_pct
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False)
    
    if missing_df.empty:
        print("🎉 Great news! There are no missing values in this dataset.")
    else:
        print(missing_df.to_string())
        
    # 5. Unique Values Count
    print(f"\n[5] UNIQUE VALUES PER COLUMN")
    unique_df = pd.DataFrame({
        'Unique Values': df.nunique(),
        'Cardinality Type': ['High' if x > 0.5 * len(df) else 'Low' for x in df.nunique()]
    })
    print(unique_df.to_string())
    
    # 6. Statistical Summary (Numerical Columns)
    print(f"\n[6] STATISTICAL SUMMARY (Numerical Columns)")
    num_summary = df.describe().T
    if not num_summary.empty:
        # Adding Median (50%) explicitly since describe() has it, but let's make it neat
        print(num_summary[['count', 'mean', 'std', 'min', '50%', 'max']].to_string())
    else:
        print("No numerical columns available for statistical summary.")
        
    # 7. Statistical Summary (Categorical Columns)
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        print(f"\n[7] STATISTICAL SUMMARY (Categorical Columns)")
        print(df[cat_cols].describe().T.to_string())
    
    print("\n" + "=" * 60)
    print(" END OF EDA REPORT")
    print("=" * 60)
    
    # Return useful metrics as a dictionary for further scripting if needed
    return {
        "shape": df.shape,
        "duplicates": duplicates,
        "missing_values": missing_count.to_dict(),
        "numerical_summary": num_summary if not num_summary.empty else None
    }
