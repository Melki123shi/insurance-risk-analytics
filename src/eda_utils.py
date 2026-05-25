import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

sns.set_style("whitegrid")

# ====================== DATA SUMMARIZATION ======================
date_cols = ["TransactionMonth", "RegistrationYear", "VehicleIntroDate"]


def summarize_data(df: pd.DataFrame) -> None:
    """Print comprehensive data summary with correct column handling."""
    print("=" * 70)
    print("DATA SUMMARIZATION")
    print("=" * 70)

    print(f"Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n")

    print("DATA TYPES:")
    print(df.dtypes.value_counts())
    print("\n")

    # Numerical columns
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    print(f"NUMERICAL COLUMNS ({len(num_cols)}):")
    print(num_cols)
    print("\nDescriptive Statistics:")
    print(df[num_cols].describe().round(2))

    # Categorical columns
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    print(f"\nCATEGORICAL COLUMNS ({len(cat_cols)}):")
    print(cat_cols)


# ====================== COLUMN TYPE CASTING ======================


def cast_column_types(
    df: pd.DataFrame,
    categorical_cols: List[str] = None,
    date_cols: List[str] = None,
    numerical_cols: List[str] = None,
) -> pd.DataFrame:
    """
    Cast DataFrame columns to correct types: categorical, date, and numerical.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    categorical_cols : List[str]
        Columns to convert to 'category' dtype
    date_cols : List[str]
        Columns to convert to 'datetime64[ns]' dtype
    numerical_cols : List[str]
        Columns to convert to 'float64' dtype

    Returns:
    --------
    pd.DataFrame
        DataFrame with correctly typed columns
    """
    df = df.copy()

    print("=" * 80)
    print("COLUMN TYPE CASTING")
    print("=" * 80)

    # Cast categorical columns
    if categorical_cols:
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype("category")
                print(f"✓ {col:30} → category")

    # Cast date columns
    if date_cols:
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                print(f"✓ {col:30} → datetime64[ns]")

    # Cast numerical columns
    if numerical_cols:
        for col in numerical_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                print(f"✓ {col:30} → float64")

    print("=" * 80)
    return df


def verify_column_types(
    df: pd.DataFrame,
    categorical_cols: List[str] = None,
    date_cols: List[str] = None,
    numerical_cols: List[str] = None,
) -> bool:
    """
    Verify that all columns are correctly typed after casting.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to verify
    categorical_cols : List[str]
        Columns that should be 'category' type
    date_cols : List[str]
        Columns that should be 'datetime64[ns]' type
    numerical_cols : List[str]
        Columns that should be 'float64' type

    Returns:
    --------
    bool
        True if all columns are correctly typed, False otherwise
    """
    print("=" * 80)
    print("COLUMN TYPE VERIFICATION")
    print("=" * 80)

    all_correct = True

    # Verify categorical columns
    if categorical_cols:
        print("\n📋 Categorical Columns:")
        for col in categorical_cols:
            if col in df.columns:
                is_correct = df[col].dtype.name == "category"
                status = "✅" if is_correct else "❌"
                print(f"  {status} {col:30} → {df[col].dtype}")
                if not is_correct:
                    all_correct = False

    # Verify date columns
    if date_cols:
        print("\n📅 Date Columns:")
        for col in date_cols:
            if col in df.columns:
                is_dt = pd.api.types.is_datetime64_any_dtype(df[col])
                status = "✅" if is_dt else "❌"
                print(f"  {status} {col:30} → {df[col].dtype}")
                if not is_dt:
                    all_correct = False

    # Verify numerical columns
    if numerical_cols:
        print("\n🔢 Numerical Columns:")
        for col in numerical_cols:
            if col in df.columns:
                is_correct = pd.api.types.is_numeric_dtype(df[col])
                status = "✅" if is_correct else "❌"
                print(f"  {status} {col:30} → {df[col].dtype}")
                if not is_correct:
                    all_correct = False

    print("\n" + "=" * 80)
    if all_correct:
        print("✅ All column types are CORRECT!")
    else:
        print("⚠️  Some column types are INCORRECT. Please review above.")
    print("=" * 80)

    return all_correct


# ====================== DATA QUALITY ASSESSMENT ======================


def check_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Assess data quality - missing values, empty strings, duplicates."""
    print("=" * 70)
    print("DATA QUALITY ASSESSMENT")
    print("=" * 70)

    pd.to_datetime(df["TransactionMonth"], errors="raise")
    df["TransactionMonth"] = pd.to_datetime(
        df["TransactionMonth"], errors="coerce"
    )
    print(f"Parsed {df['TransactionMonth']} as datetime.")

    # Replace empty strings with NaN for better analysis
    df = df.replace(r"^\s*$", np.nan, regex=True)

    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame(
        {
            "Missing Count": missing,
            "Missing %": missing_pct.round(2),
        }
    ).sort_values("Missing %", ascending=False)

    missing_df = missing_df[missing_df["Missing Count"] > 0]

    if missing_df.empty:
        print("✅ No missing values found!")
    else:
        print("Top 15 Columns with Missing Values:")
        print(missing_df.head(15))

    # Duplicates
    duplicates = df.duplicated().sum()
    dup_pct = (duplicates / len(df) * 100)
    print(f"\nDuplicate Rows: {duplicates:,} ({dup_pct:.2f}%)")


def handle_missing_values(
    df: pd.DataFrame,
    drop_col_threshold: float = 0.60,
    drop_row_threshold: float = 0.02,
) -> pd.DataFrame:
    """
    Automatic missing value handler.

    Parameters
    ----------
    drop_col_threshold : float
        Drop column if missing ratio > threshold.
        Example: 0.60 = 60%

    drop_row_threshold : float
        Drop rows if column missing ratio <= threshold.
        Example: 0.02 = 2%
    """

    df = df.copy()

    print("=" * 80)
    print("AUTOMATIC MISSING VALUE HANDLING")
    print("=" * 80)

    # =========================================================
    # 0. CLEAN EMPTY STRINGS
    # =========================================================
    df = df.replace(r"^\s*$", np.nan, regex=True)

    # =========================================================
    # 1. CALCULATE MISSING RATIO (0 → 1)
    # =========================================================
    missing_ratio = df.isnull().mean()

    # =========================================================
    # 2. DROP HIGH-MISSING COLUMNS
    # =========================================================
    cols_to_drop = (
        missing_ratio[missing_ratio > drop_col_threshold].index.tolist()
    )

    pct = drop_col_threshold * 100
    print(
        f"\n🗑️ Dropping {len(cols_to_drop)} columns (> {pct:.0f}% missing)"
    )

    if cols_to_drop:
        print(cols_to_drop)

    df = df.drop(columns=cols_to_drop, errors="ignore")

    # =========================================================
    # 3. DROP ROWS FOR LOW-MISSING COLUMNS
    # =========================================================
    pct_row = drop_row_threshold * 100
    print(
        f"\n📉 Dropping rows for columns with "
        f"≤ {pct_row:.0f}% missing..."
    )

    for col in df.columns:
        miss_ratio = df[col].isnull().mean()

        if 0 < miss_ratio <= drop_row_threshold:
            rows_before = len(df)

            df = df.dropna(subset=[col])

            rows_dropped = rows_before - len(df)

            miss_pct = miss_ratio * 100
            msg = (
                f"   → {col:30} Dropped {rows_dropped:,} rows "
                f"({miss_pct:.2f}%)"
            )
            print(msg)

    # =========================================================
    # 4. HANDLE DERIVED METRICS FIRST
    # =========================================================
    if {"TotalPremium", "TotalClaims"}.issubset(df.columns):
        print("\n🧮 Recalculating LossRatio and Margin...")

        # Ensure numeric
        df["TotalPremium"] = pd.to_numeric(df["TotalPremium"], errors="coerce")

        df["TotalClaims"] = pd.to_numeric(df["TotalClaims"], errors="coerce")

        # Margin
        df["Margin"] = df["TotalPremium"] - df["TotalClaims"]

        # Safe denominator
        premium = df["TotalPremium"].replace(0, np.nan)

        # Recalculate LossRatio safely
        df["LossRatio"] = df["TotalClaims"] / premium

        # Replace inf values
        df["LossRatio"] = df["LossRatio"].replace(
            [np.inf, -np.inf],
            np.nan,
        )

        # Optional: cap unrealistic ratios
        df["LossRatio"] = df["LossRatio"].clip(lower=0)

        # Fill missing LossRatio
        lossratio_nan = df["LossRatio"].isnull().sum()

        if lossratio_nan > 0:
            median_lr = df["LossRatio"].median()

            if pd.isna(median_lr):
                median_lr = 0.0

            df["LossRatio"] = df["LossRatio"].fillna(median_lr)

            print(
                f"   → Filled {lossratio_nan:,} LossRatio NaNs "
                f"with median ({median_lr:.4f})"
            )

    # =========================================================
    # 5. FILL REMAINING MISSING VALUES
    # =========================================================
    print("\n🔧 Filling remaining missing values...")

    for col in df.columns:
        missing_count = df[col].isnull().sum()

        if missing_count == 0:
            continue

        miss_pct = df[col].isnull().mean() * 100

        # Object / categorical columns
        if df[col].dtype == "object" or str(df[col].dtype) == "category":
            mode_series = df[col].mode()

            if not mode_series.empty:
                fill_val = mode_series.iloc[0]
            else:
                fill_val = "Unknown"

            df[col] = df[col].fillna(fill_val)

            msg = f"   {col:30} → Mode: '{fill_val}' ({miss_pct:.2f}%)"
            print(msg)

        # Numeric columns
        else:
            median_val = df[col].median()

            if pd.isna(median_val):
                median_val = 0

            df[col] = df[col].fillna(median_val)

            print(f"   {col:30} → Median: {median_val:.4f} ({miss_pct:.2f}%)")

    # =========================================================
    # 6. FINAL VALIDATION
    # =========================================================
    print("\n🔍 Final validation...")

    total_missing = df.isnull().sum().sum()

    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()

    print(f"   Remaining missing values : {total_missing:,}")
    print(f"   Remaining infinite values: {inf_count:,}")

    print("\n✅ Missing value handling completed!")
    print(f"Final dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    print("=" * 80)

    return df


def recalculate_loss_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recalculate LossRatio safely.
    """

    df = df.copy()

    # Ensure numeric
    df["TotalPremium"] = pd.to_numeric(df["TotalPremium"], errors="coerce")
    df["TotalClaims"] = pd.to_numeric(df["TotalClaims"], errors="coerce")

    # Safe calculation
    df["LossRatio"] = (
        df["TotalClaims"] /
        df["TotalPremium"].replace(0, np.nan)
    )

    # Replace inf values
    df["LossRatio"] = df["LossRatio"].replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Fill missing values
    median_lr = df["LossRatio"].median()

    if pd.isna(median_lr):
        median_lr = 0

    df["LossRatio"] = df["LossRatio"].fillna(median_lr)

    return df


def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows from the dataframe."""

    df = df.copy()
    before_count = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed_count = before_count - len(df)

    print(f"Removed {removed_count:,} duplicate rows")

    return df


def check_missing_values(df: pd.DataFrame):
    """Check for missing values and print summary."""
    print("=" * 70)
    print("MISSING VALUE CHECK")
    print("=" * 70)

    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame(
        {"Missing Count": missing, "Missing %": missing_pct.round(2)}
    ).sort_values("Missing %", ascending=False)

    if missing_df.empty:
        print("✅ No missing values found!")
    else:
        print("Top 5 Columns with Missing Values:")
        print(missing_df.head())

    return


# ====================== UNIVARIATE ANALYSIS ======================


def get_numerical_columns(df: pd.DataFrame) -> List[str]:
    """Return a list of numerical columns in the dataframe."""
    cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    numerical_cols = []
    for col in cols:
        if col in date_cols:
            continue
        numerical_cols.append(col)
    print(f"Numerical columns: {cols}")
    return numerical_cols


def plot_numerical_distributions(
    df: pd.DataFrame, columns: List[str], figsize=(26, 20)
):
    """Plot histograms for key numerical features."""
    n_cols = len(columns)
    # Use 3 columns for odd number of features, 2 for even
    n_subplot_cols = 3 if n_cols % 2 == 1 else 2
    # ceiling division
    n_subplot_rows = (n_cols + n_subplot_cols - 1) // n_subplot_cols

    plt.figure(figsize=figsize)
    for i, col in enumerate(columns, 1):
        plt.subplot(n_subplot_rows, n_subplot_cols, i)
        sns.histplot(
            df[col].dropna(), kde=True, bins=30, color="skyblue"
        )
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
    plt.tight_layout()
    plt.show()


def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    """Return a list of categorical columns in the dataframe."""
    cols = df.select_dtypes(include=["object"]).columns.tolist()
    categorical_cols = []
    for col in cols:
        if col in date_cols:
            continue
        categorical_cols.append(col)
    print(f"Categorical columns: {categorical_cols}")
    return categorical_cols


def plot_categorical_distributions(
    df: pd.DataFrame,
    columns: List[str],
    top_n: int = 10,
    figsize: tuple = (30, 24),
):
    """Plot bar charts for important categorical columns."""
    plt.figure(figsize=figsize)
    n_cols = len(columns)
    # Use 3 columns for odd number of features, 2 for even
    n_subplot_cols = 3 if n_cols % 2 == 1 else 2
    n_subplot_rows = (n_cols + n_subplot_cols - 1) // n_subplot_cols
    for i, col in enumerate(columns, 1):
        plt.subplot(n_subplot_rows, n_subplot_cols, i)
        top_cats = df[col].value_counts().head(top_n)
        sns.barplot(
            x=top_cats.values, y=top_cats.index, palette="viridis"
        )
        plt.title(f"Top {top_n} {col}")
        plt.xlabel("Count")
    plt.tight_layout()
    plt.show()


# ====================== BIVARIATE / MULTIVARIATE ======================


def plot_premium_vs_claims(df: pd.DataFrame, hue: str = None):
    """Scatter plot: TotalPremium vs TotalClaims"""
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df,
        x="TotalPremium",
        y="TotalClaims",
        hue=hue,
        alpha=0.6,
        palette="coolwarm",
    )
    plt.title("Total Premium vs Total Claims")
    plt.xlabel("Total Premium (Rand)")
    plt.ylabel("Total Claims (Rand)")
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame):
    """Correlation heatmap for numerical features."""
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    plt.figure(figsize=(14, 12))
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
    )
    plt.title("Correlation Matrix - Numerical Features")
    plt.show()


# ====================== GEOGRAPHIC TRENDS ======================


def analyze_geographic_trends(df: pd.DataFrame):
    """Analyze key metrics by Province."""
    print("\n=== GEOGRAPHIC TRENDS ===")

    province_summary = (
        df.groupby("Province")
        .agg({
            "TotalPremium": "sum",
            "TotalClaims": "sum",
            "PolicyID": "nunique",
        })
        .assign(
            LossRatio=lambda x: (
                x["TotalClaims"] / x["TotalPremium"] * 100
            ).round(2),
            AvgPremium=lambda x: (
                x["TotalPremium"] / x["PolicyID"]
            ).round(2),
        )
    )

    print(province_summary.sort_values("LossRatio", ascending=False))

    # Visualization
    plt.figure(figsize=(14, 6))
    sns.barplot(
        data=province_summary.reset_index(),
        x="Province",
        y="LossRatio",
        palette="Reds_d",
    )
    plt.title("Loss Ratio by Province (%)")
    plt.xticks(rotation=45)
    plt.ylabel("Loss Ratio (%)")
    plt.show()

    return province_summary


def analyze_vehicle_makes(df: pd.DataFrame, top_n=10):
    """Top vehicle makes by claims and premium."""
    make_summary = (df.groupby("make")
                    .agg({
                        "TotalClaims": "sum",
                        "TotalPremium": "sum"
                    })
                    .assign(LossRatio=lambda x: (
                        x["TotalClaims"] / x["TotalPremium"] * 100
                    ).round(2))
                    .sort_values("TotalClaims", ascending=False))

    print(f"\nTop {top_n} Vehicle Makes by Total Claims:")
    print(make_summary.head(top_n))

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=make_summary.head(top_n).reset_index(),
        x="TotalClaims",
        y="make",
    )
    plt.title(f"Top {top_n} Vehicle Makes by Total Claims")
    plt.show()


# ====================== OUTLIER DETECTION ======================


def detect_outliers(df: pd.DataFrame, columns: List[str]):
    """Box plots and IQR outlier summary."""
    plt.figure(figsize=(26, 20))
    n_cols = len(columns)
    n_subplot_cols = 3 if n_cols % 2 == 1 else 2
    n_subplot_rows = (n_cols + n_subplot_cols - 1) // n_subplot_cols
    for i, col in enumerate(columns, 1):
        plt.subplot(n_subplot_rows, n_subplot_cols, i)
        sns.boxplot(y=df[col])
        plt.title(f"Outliers: {col}")
    plt.tight_layout()
    plt.show()

    print("\nOutlier Summary (IQR Method):")
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        pct = outliers / len(df) * 100
        print(f"{col:25}: {outliers:,} outliers ({pct:.2f}%)")


# ====================== HELPER FUNCTIONS ======================


def create_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add important derived metrics."""
    df = df.copy()
    df["LossRatio"] = df["TotalClaims"] / df["TotalPremium"]
    df["Margin"] = df["TotalPremium"] - df["TotalClaims"]
    return df


def save_cleaned_data(
    df: pd.DataFrame,
    filename: str = "../data/processed/insurance_data.csv",
):
    """Save the cleaned dataset to a CSV file."""
    df.to_csv(filename, index=False)
    print(f"Cleaned data saved to {filename}")


def portfolio_loss_ratio_analysis(df: pd.DataFrame) -> dict:
    """
    Answers: Overall Loss Ratio + variation by Province, VehicleType.
    """
    print("=" * 70)
    print("PORTFOLIO LOSS RATIO ANALYSIS")
    print("=" * 70)

    # Overall
    total_premium = df['TotalPremium'].sum()
    total_claims = df['TotalClaims'].sum()
    if total_premium != 0:
        overall_lr = (total_claims / total_premium * 100)
    else:
        overall_lr = 0

    print(f"Overall Portfolio Loss Ratio: {overall_lr:.2f}%")

    # By Province
    province_lr = (df.groupby('Province').agg(
        TotalPremium=('TotalPremium', 'sum'),
        TotalClaims=('TotalClaims', 'sum')
    ).assign(LossRatio=lambda x: (
        x['TotalClaims'] / x['TotalPremium'] * 100
    ).round(2)).sort_values('LossRatio', ascending=False))

    print("\nTop 5 Highest Loss Ratio Provinces:")
    print(province_lr.head())

    # By VehicleType
    veh_lr = (df.groupby('VehicleType').agg(
        TotalPremium=('TotalPremium', 'sum'),
        TotalClaims=('TotalClaims', 'sum')
    ).assign(LossRatio=lambda x: (
        x['TotalClaims'] / x['TotalPremium'] * 100
    ).round(2)).sort_values('LossRatio', ascending=False))

    print("\nTop 5 Highest Loss Ratio Vehicle Types:")
    print(veh_lr.head())

    # By Gender
    gender_lr = (df.groupby('Gender').agg(
        TotalPremium=('TotalPremium', 'sum'),
        TotalClaims=('TotalClaims', 'sum')
    ).assign(LossRatio=lambda x: (
        x['TotalClaims'] / x['TotalPremium'] * 100
    ).round(2)))

    print("\nLoss Ratio by Gender:")
    print(gender_lr)

    return {
        'overall': overall_lr,
        'province': province_lr,
        'vehicle_type': veh_lr,
        'gender': gender_lr
    }


def plot_financial_distributions_and_outliers(df: pd.DataFrame):
    """
    Answers: Distributions of key financial variables + Outliers.
    Creates 2 insightful plots.
    """
    key_cols = [
        'TotalPremium', 'TotalClaims', 'LossRatio', 'Margin',
        'SumInsured', 'TotalClaims'
    ]

    # Plot 1: Distributions with Log Scale (handles skewness)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    for i, col in enumerate(key_cols):
        sns.histplot(
            df[col].dropna(),
            kde=True,
            bins=50,
            ax=axes[i],
            color='skyblue',
            log_scale=(True, False)
        )
        axes[i].set_title(f'Distribution of {col} (Log Scale)')
        axes[i].set_xlabel(col)

    plt.tight_layout()
    suptitle = 'Key Financial Variables Distributions (Log Scale)'
    plt.suptitle(suptitle, y=1.02, fontsize=16)
    plt.show()

    # Plot 2: Box plots for Outlier Detection
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=df[key_cols], orient='h', palette='Set2')
    title = 'Outlier Detection in Key Financial Variables (Box Plots)'
    plt.title(title)
    plt.xlabel('Value')
    plt.tight_layout()
    plt.show()

    print("Note: Significant outliers detected in TotalClaims.")


def analyze_temporal_trends(df: pd.DataFrame):
    """
    Answers: Are there temporal trends in claim frequency and severity?
    """
    df = df.copy()
    df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'])
    df['YearMonth'] = df['TransactionMonth'].dt.to_period('M')

    monthly = df.groupby('YearMonth').agg(
        TotalPremium=('TotalPremium', 'sum'),
        TotalClaims=('TotalClaims', 'sum'),
        PolicyCount=('PolicyID', 'nunique'),
        AvgClaim=('TotalClaims', 'mean')
    ).assign(
        LossRatio=lambda x: (
            x['TotalClaims'] / x['TotalPremium'] * 100
        ).round(2),
        ClaimFrequency=lambda x: x['TotalClaims'] / x['PolicyCount']
    )

    print("Monthly Trends Summary:")
    print(monthly.tail(10))

    # Plot 3: Temporal Trends
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    monthly['LossRatio'].plot(
        ax=axes[0, 0], marker='o', color='red',
        title='Monthly Loss Ratio Trend'
    )
    monthly['ClaimFrequency'].plot(
        ax=axes[0, 1], marker='o', color='orange',
        title='Claim Frequency Trend'
    )
    monthly['AvgClaim'].plot(
        ax=axes[1, 0], marker='o', color='blue',
        title='Average Claim Severity Trend'
    )
    monthly['PolicyCount'].plot(
        ax=axes[1, 1], marker='o', color='green',
        title='Policy Volume Trend'
    )

    plt.tight_layout()
    suptitle = 'Temporal Trends in Claims (Feb 2014 - Aug 2015)'
    plt.suptitle(suptitle, y=1.02, fontsize=16)
    plt.show()

    return monthly


def analyze_vehicle_risk(df: pd.DataFrame, top_n: int = 10):
    """
    Answers: Which vehicle makes have highest and lowest claim amounts?
    """
    make_summary = df.groupby('make').agg(
        TotalClaims=('TotalClaims', 'sum'),
        TotalPremium=('TotalPremium', 'sum'),
        PolicyCount=('PolicyID', 'nunique')
    ).assign(
        AvgClaim=lambda x: x['TotalClaims'] / x['PolicyCount'],
        LossRatio=lambda x: (
            x['TotalClaims'] / x['TotalPremium'] * 100
        ).round(2)
    ).sort_values('TotalClaims', ascending=False)

    print(f"\nTop {top_n} Highest Claim Vehicle Makes:")
    cols = ['TotalClaims', 'AvgClaim', 'LossRatio']
    print(make_summary.head(top_n)[cols])

    print(f"\nTop {top_n} Lowest Claim Vehicle Makes:")
    print(make_summary.tail(top_n)[cols])

    # Plot 4: Top Makes by Claims vs Premium
    top_makes = make_summary.head(top_n)

    fig, ax = plt.subplots(figsize=(14, 8))
    x = range(len(top_makes))
    width = 0.35

    ax.bar(
        [i - width/2 for i in x],
        top_makes['TotalClaims'],
        width,
        label='Total Claims',
        color='salmon',
        alpha=0.8
    )
    ax.bar(
        [i + width/2 for i in x],
        top_makes['TotalPremium'],
        width,
        label='Total Premium',
        color='lightblue',
        alpha=0.8
    )

    ax.set_xlabel('Vehicle Make')
    ax.set_ylabel('Amount (Rand)')
    ax.set_title(f'Top {top_n} Vehicle Makes: Claims vs Premium')
    ax.set_xticks(x)
    ax.set_xticklabels(top_makes.index, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.show()

    return make_summary
