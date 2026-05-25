import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional

sns.set_style("whitegrid")

# ====================== DATA SUMMARIZATION ======================

def advanced_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform advanced data cleaning steps on the insurance dataset.
    This includes handling outliers, encoding categorical variables, and feature engineering.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The raw insurance dataset
    
    Returns:
    --------
    pd.DataFrame
        The cleaned and preprocessed dataset ready for analysis
    """
    
    # change TransactionMonth to datetime
    df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'])
    
    #check for missing values
    missing_values = df.isnull().sum()
    print("Missing values in each column:")
    print(missing_values)
    
    #remove missing values
    print(f"Total records before cleaning: {df.shape[0]}")
    df = df.dropna()
    print(f"Total records after cleaning: {df.shape[0]}")
    
    # Example: Handle outliers in TotalPremium using IQR method
    Q1 = df['TotalPremium'].quantile(0.25)
    Q3 = df['TotalPremium'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df['TotalPremium'] >= lower_bound) & (df['TotalPremium'] <= upper_bound)]
    
    # Example: Encode categorical variables (if any)
    df = pd.get_dummies(df, columns=['AccountType', 'PolicyType'], drop_first=True)
    
    return df

def summarize_data(df: pd.DataFrame) -> None:
    """Print comprehensive data summary with correct column handling."""
    print("="*70)
    print("DATA SUMMARIZATION")
    print("="*70)
    
    print(f"Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n")
    
    print("DATA TYPES:")
    print(df.dtypes.value_counts())
    print("\n")
    
    # Numerical columns
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    print(f"NUMERICAL COLUMNS ({len(num_cols)}):")
    print(num_cols)
    print("\nDescriptive Statistics:")
    print(df[num_cols].describe().round(2))
    
    # Categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"\nCATEGORICAL COLUMNS ({len(cat_cols)}):")
    print(cat_cols)
    
# ====================== DATA QUALITY ASSESSMENT ======================

def check_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Assess data quality - missing values, empty strings, duplicates."""
    print("="*70)
    print("DATA QUALITY ASSESSMENT")
    print("="*70)
    
    
    pd.to_datetime(df['TransactionMonth'], errors='raise')
    df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
    print(f"Parsed {df['TransactionMonth']} as datetime.")
    
    # Replace empty strings with NaN for better analysis
    df = df.replace(r'^\s*$', np.nan, regex=True)
    
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing %': missing_pct.round(2)
    }).sort_values('Missing %', ascending=False)
    
    missing_df = missing_df[missing_df['Missing Count'] > 0]
    
    if missing_df.empty:
        print("✅ No missing values found!")
    else:
        print("Top 15 Columns with Missing Values:")
        print(missing_df.head(15))
    
    # Duplicates
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate Rows: {duplicates:,} ({(duplicates/len(df)*100):.2f}%)")
    
    return missing_df

def handle_missing_values(df: pd.DataFrame, 
                         drop_col_threshold: float = 0.60, 
                         drop_row_threshold: float = 0.02) -> pd.DataFrame:
    """
    Automatic missing value handler with row dropping logic.
    
    Parameters:
    -----------
    drop_col_threshold : float → Drop column if missing > 60%
    drop_row_threshold : float → Drop rows if missing in column ≤ 2%
    """
    df = df.copy()
    
    print("="*80)
    print("AUTOMATIC MISSING VALUE HANDLING")
    print("="*80)
    
    df = df.replace(r'^\s*$', np.nan, regex=True)
    
    # Calculate missing percentage
    missing_pct = df.isnull().mean() * 100
    
    # === 1. DROP COLUMNS ===
    cols_to_drop = missing_pct[missing_pct > drop_col_threshold].index.tolist()
    print(f"🗑️ Dropping {len(cols_to_drop)} columns (> {drop_col_threshold*100}% missing)")
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    # === 2. DROP ROWS (for low missing columns) ===
    print(f"\n📉 Checking columns for row dropping (threshold ≤ {drop_row_threshold*100}%)...")
    
    for col in df.columns:
        miss_pct = df[col].isnull().mean()
        
        if miss_pct <= drop_row_threshold and miss_pct > 0:
            rows_before = len(df)
            df = df.dropna(subset=[col])
            rows_dropped = rows_before - len(df)
            print(f"   → Dropped {rows_dropped:,} rows from '{col}' ({miss_pct*100:.2f}%)")
    
    # === 3. FILL REMAINING MISSING VALUES ===
    print(f"\n🔧 Filling remaining missing values...")
    
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
            
        miss_pct = df[col].isnull().mean() * 100
        
        if df[col].dtype == 'object':
            fill_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
            df[col] = df[col].fillna(fill_val)
            print(f"   {col:30} → Mode: '{fill_val}' ({miss_pct:.2f}%)")
        else:
            fill_val = df[col].median()
            df[col] = df[col].fillna(fill_val)
            print(f"   {col:30} → Median: {fill_val:.2f} ({miss_pct:.2f}%)")
    
    print(f"\n✅ Final Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print("="*80)
    
    # ====================== RECALCULATE DERIVED METRICS ======================
    if {'TotalPremium', 'TotalClaims'}.issubset(df.columns):
        # Drop old LossRatio if it exists (to avoid division by old values)
        if 'LossRatio' in df.columns:
            df = df.drop(columns=['LossRatio'])
        
        df['LossRatio'] = df['TotalClaims'] / df['TotalPremium'].replace(0, np.nan)
        df['Margin'] = df['TotalPremium'] - df['TotalClaims']
        print("   LossRatio & Margin → Recalculated")
    
    print(f"\n✅ Missing values handling completed!")
    print(f"Final dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print("="*70)
    
    return df 

def check_missing_values(df: pd.DataFrame):
    """Check for missing values and print summary."""
    print("="*70)
    print("MISSING VALUE CHECK")
    print("="*70)
    
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing %': missing_pct.round(2)
    }).sort_values('Missing %', ascending=False)
    
    if missing_df.empty:
        print("✅ No missing values found!")
    else:
        print("Top 15 Columns with Missing Values:")
        print(missing_df.head(15))
    
    return missing_df
    
# ====================== UNIVARIATE ANALYSIS ======================

def get_numerical_columns(df: pd.DataFrame) -> List[str]:
    """Return a list of numerical columns in the dataframe."""
    cols =  df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    print(f"Numerical columns: {cols}")
    return cols


def plot_numerical_distributions(df: pd.DataFrame, columns: List[str], figsize=(16, 10)):
    """Plot histograms for key numerical features."""
    plt.figure(figsize=figsize)
    for i, col in enumerate(columns, 1):
        plt.subplot(2, 3, i)
        sns.histplot(df[col].dropna(), kde=True, bins=30, color='skyblue')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
    plt.tight_layout()
    plt.show()

def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    """Return a list of categorical columns in the dataframe."""
    cols = df.select_dtypes(include=['object']).columns.tolist()
    print(cols)
    return cols

def plot_categorical_distributions(df: pd.DataFrame, columns: List[str], top_n=10, figsize=(15, 12)):
    """Plot bar charts for important categorical columns."""
    plt.figure(figsize=figsize)
    for i, col in enumerate(columns, 1):
        plt.subplot(len(columns)//2 + 1, 2, i)
        top_cats = df[col].value_counts().head(top_n)
        sns.barplot(x=top_cats.values, y=top_cats.index, palette='viridis')
        plt.title(f'Top {top_n} {col}')
        plt.xlabel('Count')
    plt.tight_layout()
    plt.show()


# ====================== BIVARIATE / MULTIVARIATE ======================

def plot_premium_vs_claims(df: pd.DataFrame, hue: str = None):
    """Scatter plot: TotalPremium vs TotalClaims"""
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df, 
        x='TotalPremium', 
        y='TotalClaims', 
        hue=hue, 
        alpha=0.6,
        palette='coolwarm'
    )
    plt.title('Total Premium vs Total Claims')
    plt.xlabel('Total Premium (Rand)')
    plt.ylabel('Total Claims (Rand)')
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame):
    """Correlation heatmap for numerical features."""
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    plt.figure(figsize=(14, 12))
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix - Numerical Features')
    plt.show()


# ====================== GEOGRAPHIC TRENDS ======================

def analyze_geographic_trends(df: pd.DataFrame):
    """Analyze key metrics by Province."""
    print("\n=== GEOGRAPHIC TRENDS ===")
    
    province_summary = df.groupby('Province').agg({
        'TotalPremium': 'sum',
        'TotalClaims': 'sum',
        'PolicyID': 'nunique'
    }).assign(
        LossRatio=lambda x: (x['TotalClaims'] / x['TotalPremium'] * 100).round(2),
        AvgPremium=lambda x: (x['TotalPremium'] / x['PolicyID']).round(2)
    )
    
    print(province_summary.sort_values('LossRatio', ascending=False))
    
    # Visualization
    plt.figure(figsize=(14, 6))
    sns.barplot(data=province_summary.reset_index(), x='Province', y='LossRatio', palette='Reds_d')
    plt.title('Loss Ratio by Province (%)')
    plt.xticks(rotation=45)
    plt.ylabel('Loss Ratio (%)')
    plt.show()
    
    return province_summary


def analyze_vehicle_makes(df: pd.DataFrame, top_n=10):
    """Top vehicle makes by claims and premium."""
    make_summary = df.groupby('make').agg({
        'TotalClaims': 'sum',
        'TotalPremium': 'sum'
    }).assign(
        LossRatio=lambda x: (x['TotalClaims'] / x['TotalPremium'] * 100).round(2)
    ).sort_values('TotalClaims', ascending=False)
    
    print(f"\nTop {top_n} Vehicle Makes by Total Claims:")
    print(make_summary.head(top_n))
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=make_summary.head(top_n).reset_index(), 
                x='TotalClaims', y='make')
    plt.title(f'Top {top_n} Vehicle Makes by Total Claims')
    plt.show()


# ====================== OUTLIER DETECTION ======================

def detect_outliers(df: pd.DataFrame, columns: List[str]):
    """Box plots and IQR outlier summary."""
    plt.figure(figsize=(15, 8))
    for i, col in enumerate(columns, 1):
        plt.subplot(1, len(columns), i)
        sns.boxplot(y=df[col])
        plt.title(f'Outliers: {col}')
    plt.tight_layout()
    plt.show()
    
    print("\nOutlier Summary (IQR Method):")
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
        print(f"{col:25}: {outliers:,} outliers ({outliers/len(df)*100:.2f}%)")


# ====================== HELPER FUNCTIONS ======================

def create_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add important derived metrics."""
    df = df.copy()
    df['LossRatio'] = df['TotalClaims'] / df['TotalPremium']
    df['Margin'] = df['TotalPremium'] - df['TotalClaims']
    return df

def save_cleaned_data(df: pd.DataFrame, filename: str = '../data/processed/cleaned_insurance_data.csv'):
    """Save the cleaned dataset to a CSV file."""
    df.to_csv(filename, index=False)
    print(f"Cleaned data saved to {filename}")