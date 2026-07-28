
# ==========================================
# ADVANCED EDA PIPELINE (Single Function Implementation)
# ==========================================
# Note: This script assumes that 'df' is already loaded in your environment.
# All code and analysis steps are encapsulated entirely inside the `eda_by_ai()` function.


def eda_by_ai(df):
    import warnings
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from statsmodels.tsa.seasonal import seasonal_decompose

    warnings.filterwarnings("ignore")

    # Styling setup
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)

    # Make a copy of the dataframe to avoid altering the original
    working_df = df.copy()

    # Identify numerical and categorical columns
    num_cols = working_df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = working_df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Automatically try to detect a date column if one exists
    date_col = None
    for col in working_df.columns:
        if (
            "date" in col.lower()
            or "time" in col.lower()
            or pd.api.types.is_datetime64_any_dtype(working_df[col])
        ):
            date_col = col
            break

    print("==========================================")
    print("PHASE 1: ENVIRONMENT SETUP & DATA PROFILING")
    print("==========================================")
    print(f"Dataset Shape: {working_df.shape}")
    print("\nMissing Values:")
    missing = working_df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values found.")

    print("\n--- Numerical Descriptives ---")
    if num_cols:
        display(working_df[num_cols].describe().T)

    print("\n--- Categorical Descriptives ---")
    if cat_cols:
        display(working_df[cat_cols].describe().T)

    print("\n==========================================")
    print("PHASE 2: CORRELATION ANALYSIS")
    print("==========================================")
    if len(num_cols) > 1:
        plt.figure(figsize=(8, 6))
        corr = working_df[num_cols].corr(method="pearson")
        sns.heatmap(
            corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5
        )
        plt.title("Pearson Correlation Matrix (Numerical Features)")
        plt.show()
    else:
        print(
            "Skipping correlation analysis: Insufficient numerical columns available."
        )

    print("\n==========================================")
    print("PHASE 3: UNIVARIATE ANALYSIS")
    print("==========================================")
    print("\n--- Numerical Distributions & Outliers ---")
    for col in num_cols:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(working_df[col], kde=True, ax=axes[0], color="skyblue")
        axes[0].set_title(f"Distribution of {col}")

        sns.boxplot(x=working_df[col], ax=axes[1], color="lightgreen")
        axes[1].set_title(f"Boxplot of {col}")
        plt.tight_layout()
        plt.show()

    print("\n--- Categorical Distributions ---")
    for col in cat_cols:
        plt.figure(figsize=(8, 4))
        sns.countplot(
            data=working_df,
            x=col,
            order=working_df[col].value_counts().index[:10],
            palette="viridis",
        )
        plt.title(f"Top Categories in {col}")
        plt.xticks(rotation=45)
        plt.show()

    print("\n==========================================")
    print("PHASE 4 & 5: BIVARIATE & MULTIVARIATE ANALYSIS")
    print("==========================================")
    # Automatically pick candidate columns for multivariate plotting if available
    if len(num_cols) > 0 and len(cat_cols) >= 2:
        metric = num_cols[0]
        cat1 = cat_cols[0]
        cat2 = cat_cols[1]

        print(
            f"Generating grouped bar plot for Metric: '{metric}' across Group: '{cat1}' broken down by Hue: '{cat2}'"
        )
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=working_df,
            x=cat1,
            y=metric,
            hue=cat2,
            estimator=np.mean,
            palette="Set2",
            errorbar=None,
        )
        plt.title(f"Multivariate Analysis: Mean {metric} by {cat1} & {cat2}")
        plt.xticks(rotation=45)
        plt.legend(title=cat2, bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.show()
    else:
        print(
            "Skipping multivariate grouped bar plot: Requires at least 1 numerical and 2 categorical columns."
        )

    print("\n==========================================")
    print("PHASE 6: TIME SERIES ANALYSIS")
    print("==========================================")
    if date_col and len(num_cols) > 0:
        print(
            f"Date column detected: '{date_col}'. Performing Time Series Analysis on '{num_cols[0]}'."
        )
        temp_ts = working_df.copy()
        temp_ts[date_col] = pd.to_datetime(temp_ts[date_col])
        temp_ts = temp_ts.sort_values(by=date_col)
        temp_ts.set_index(date_col, inplace=True)

        # Resample monthly sum of the first numerical column
        ts_data = temp_ts[num_cols[0]].resample("ME").sum()

        plt.figure(figsize=(14, 5))
        plt.plot(
            ts_data.index, ts_data.values, marker="o", linestyle="-", color="b"
        )
        plt.title(f"Monthly Trend of {num_cols[0]}")
        plt.xlabel("Date")
        plt.ylabel(f"Total {num_cols[0]}")
        plt.grid(True)
        plt.show()

        # Decomposition if sufficient data points exist (at least 24 months/periods)
        if len(ts_data) >= 24:
            decomposition = seasonal_decompose(
                ts_data, model="additive", period=12
            )
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10))
            decomposition.observed.plot(ax=ax1)
            ax1.set_title("Observed")
            decomposition.trend.plot(ax=ax2)
            ax2.set_title("Trend")
            decomposition.seasonal.plot(ax=ax3)
            ax3.set_title("Seasonal")
            decomposition.resid.plot(ax=ax4)
            ax4.set_title("Residuals")
            plt.tight_layout()
            plt.show()
        else:
            print(
                "Time series too short for seasonal decomposition (requires >= 24 periods)."
            )
    else:
        print("No date column detected. Skipping Time Series Analysis.")

    print("\n=== EDA PIPELINE COMPLETED SUCCESSFULLY ===")
