import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import numpy as np
from scipy.stats import norm

import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind, f_oneway


sns.set_theme(style="whitegrid")


def plot_statistical_distribution(df, column, title_suffix=""):
    """
    Creates a three-panel visualization optimized for insurance risk analysis:
    1. Raw Distribution: Shows skewness and central tendency (Slide 6/7).
    2. Log-Transformation: Normalizes skewed data for statistical testing (Slide 4).
    3. Box Plot: Explicitly identifies high-risk outliers (Slide 3).
    """
    # 1. Calculate Core Statistics
    mu = df[column].mean()
    median = df[column].median()
    std = df[column].std()
    skew = df[column].skew()
    mode_val = df[column].mode()[0] if not df[column].mode().empty else mu

    # 2. Outlier Calculation (IQR Method)
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    outlier_threshold = q3 + 1.5 * iqr
    outliers = df[df[column] > outlier_threshold]

    # 3. Log Transformation (using log1p to handle zeros safely)
    log_values = np.log1p(df[column])

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    sns.set_theme(style="whitegrid")

    # --- Panel 1: Raw Distribution (The "Reality" - Slide 6) ---
    ax1 = axes[0]
    sns.histplot(
        df[column],
        bins=50,
        color="coral",
        edgecolor="white",
        alpha=0.7,
        ax=ax1,
        stat="density",
    )
    ax1.axvline(mu, color="navy", linestyle="--", linewidth=2, label=f"Mean: {mu:.1f}")
    ax1.axvline(
        median, color="green", linestyle="-", linewidth=2, label=f"Median: {median:.1f}"
    )
    ax1.set_title(f"{column} — Raw (Right-Skewed)", fontsize=13, fontweight="bold")
    ax1.legend()

    # --- Panel 2: Log-Transformed (The "Blueprint" - Slide 4) ---
    ax2 = axes[1]
    sns.histplot(
        log_values,
        bins=40,
        color="mediumseagreen",
        edgecolor="white",
        alpha=0.7,
        ax=ax2,
        stat="density",
    )
    # Overlay Normal Curve on Log data to check fit
    l_mu, l_std = log_values.mean(), log_values.std()
    l_median = log_values.median()
    x = np.linspace(log_values.min(), log_values.max(), 100)
    ax2.plot(x, norm.pdf(x, l_mu, l_std), color="navy", linewidth=2, label="Normal Fit")

    # Add Mean and Median lines for log-transformed data
    ax2.axvline(
        l_mu, color="red", linestyle="--", linewidth=2, label=f"Mean: {l_mu:.2f}"
    )
    ax2.axvline(
        l_median,
        color="green",
        linestyle="-",
        linewidth=2,
        label=f"Median: {l_median:.2f}",
    )

    ax2.set_title(
        f"Log({column}) — Normalized for Testing", fontsize=13, fontweight="bold"
    )
    ax2.legend()

    # --- Panel 3: Box Plot (Outlier Detection - Slide 3) ---
    ax3 = axes[2]
    sns.boxplot(
        x=df[column],
        color="coral",
        ax=ax3,
        flierprops=dict(markerfacecolor="red", alpha=0.3),
    )
    ax3.set_title(f"{column} — Outlier Detection", fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.show()

    # 4. Print Summary Statistics (Slide 14 - "What?")
    print(f"=== Analytical Summary for {column} ===")
    print(
        f"Mean:     {mu:.2f}  ← {'Pulled RIGHT' if skew > 0 else 'Pulled LEFT'} (Slide 6)"
    )
    print(f'Median:   {median:.2f}  ← More representative of "Typical" risk')
    print(
        f"Skewness: {skew:.3f}  ({'High Skew' if abs(skew) > 1 else 'Moderate Skew'})"
    )
    print(
        f"Outliers: {len(outliers)} events detected above {outlier_threshold:.2f} (IQR Method)"
    )
    print(f"Max Risk: {df[column].max():.2f} (Highest single value)")

    if abs(skew) > 1:
        print(
            "\nADVICE: Use Log-transformed data for T-tests/ANOVA to satisfy Normality assumptions."
        )


def plot_poisson_distribution(df, column, group_col=None, title_suffix=""):
    """
    Three-panel Poisson analysis for Claim Frequency (Task 3):
    1. Observed vs. Theoretical: Checks if the data follows Poisson logic (Slide 8).
    2. Log-Scale Tail: Visualizes rare multi-claim events (Slide 9).
    3. Group Comparison: Visualizes frequency differences between segments (A/B testing).
    """
    # 1. Overall Stats
    lam_total = df[column].mean()
    variance = df[column].var()
    n = len(df)

    # 2. Setup Subplots
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    sns.set_theme(style="whitegrid")

    # --- Panel 1: Observed vs. Theoretical Poisson (Slide 8/9) ---
    ax1 = axes[0]
    max_k = int(df[column].max())
    k_values = np.arange(0, max_k + 1)

    # Observed Proportions
    obs_probs = df[column].value_counts(normalize=True).sort_index()
    obs_data = [obs_probs.get(k, 0) for k in k_values]

    # Theoretical Poisson
    theory_probs = stats.poisson.pmf(k_values, mu=lam_total)

    width = 0.35
    ax1.bar(
        k_values - width / 2,
        obs_data,
        width=width,
        color="steelblue",
        alpha=0.7,
        label="Observed Risk",
    )
    ax1.bar(
        k_values + width / 2,
        theory_probs,
        width=width,
        color="coral",
        alpha=0.6,
        label=f"Poisson(λ={lam_total:.4f})",
    )

    ax1.set_title("Observed vs. Theoretical Poisson", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Number of Claims")
    ax1.set_ylabel("Proportion of Policies")
    ax1.set_xticks(k_values)
    ax1.legend()

    # --- Panel 2: Tail Behavior [Log Scale] (Slide 9) ---
    ax2 = axes[1]
    ax2.bar(k_values, obs_data, color="steelblue", alpha=0.7, label="Observed")
    ax2.plot(
        k_values, theory_probs, "o-", color="navy", markersize=8, label="Poisson Fit"
    )

    ax2.set_yscale("log")  # Essential for visualizing rare 1+ claim events
    ax2.set_title("Tail Behavior (Log Scale)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Probability [Log Scale]")
    ax2.set_xlabel("Number of Claims")
    ax2.legend()

    # --- Panel 3: Segment Comparison (A/B Test Visual) ---
    ax3 = axes[2]
    if group_col and df[group_col].nunique() >= 2:
        # Compare first two groups found in the segment
        groups = df[group_col].unique()[:2]
        colors = ["coral", "steelblue"]

        for i, g in enumerate(groups):
            group_data = df[df[group_col] == g][column]
            g_lam = group_data.mean()
            g_probs = [sum(group_data == k) / len(group_data) for k in k_values]

            ax3.bar(
                k_values + (i * 0.4 - 0.2),
                g_probs,
                width=0.4,
                color=colors[i],
                alpha=0.7,
                label=f"{g} (λ={g_lam:.4f})",
            )

        ax3.set_title(f"Frequency by {group_col}", fontsize=13, fontweight="bold")
    else:
        # Default: Theoretical Curves (Reference Logic)
        k_range = np.arange(0, max_k + 5)
        for l, c in [
            (0.5 * lam_total, "green"),
            (lam_total, "navy"),
            (2 * lam_total, "red"),
        ]:
            ax3.plot(
                k_range,
                stats.poisson.pmf(k_range, mu=l),
                "o-",
                color=c,
                alpha=0.6,
                label=f"λ={l:.2f}",
            )
        ax3.set_title("Theoretical λ Comparison", fontsize=13, fontweight="bold")

    ax3.set_xlabel("Number of Claims")
    ax3.legend()

    plt.tight_layout()
    plt.show()

    # 3. Poisson Statistics (Business Insights - Slide 14)
    print(f"=== Poisson Analysis Summary for {column} ===")
    print(f"Overall λ (Claim Frequency): {lam_total:.5f}")
    print(f"Overall Variance:           {variance:.5f}")
    print(
        f"Ratio (Var/Mean):           {variance / lam_total:.3f} (Ideally ≈ 1 for Poisson)"
    )

    if group_col:
        for g in df[group_col].unique()[:2]:
            print(
                f"Group {g} Frequency (λ): {df[df[group_col] == g][column].mean():.5f}"
            )

    print(
        "\nBusiness Insight: Higher λ indicates a segment with higher service costs and risk frequency (Slide 8)."
    )

# ===============================================================
#                      HYPOTHESIS TESTS FUNCTIONS
# ===============================================================

# ====================== HELPER FUNCTIONS ======================

def print_test_header(test_name: str):
    print('\n' + '=' * 70)
    print(f'          {test_name.upper().center(60)}')
    print('=' * 70)


def interpret_decision(p_value: float, alpha: float = 0.05) -> bool:
    """Returns True if H0 is rejected"""
    if p_value < alpha:
        print(f'DECISION: p = {p_value:.6f} < α ({alpha}) → REJECT H₀')
        return True
    else:
        print(f'DECISION: p = {p_value:.4f} ≥ α ({alpha}) → FAIL TO REJECT H₀')
        return False


def plot_chi_squared(contingency_table: pd.DataFrame, p_value: float, title: str):
    plt.figure(figsize=(10, 6))
    sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{title}\nChi-Squared Test (p = {p_value:.4f})', fontsize=14, fontweight='bold')
    plt.ylabel('Group')
    plt.xlabel('Claim Status')
    plt.tight_layout()
    plt.show()


def plot_numerical_comparison(data_a, data_b, label_a: str, label_b: str, p_value: float, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Histogram
    axes[0].hist(data_a, bins=50, alpha=0.7, color='coral', density=True, label=label_a)
    axes[0].hist(data_b, bins=50, alpha=0.7, color='steelblue', density=True, label=label_b)
    axes[0].axvline(data_a.mean(), color='darkred', linestyle='--', linewidth=2, label=f'{label_a} Mean')
    axes[0].axvline(data_b.mean(), color='navy', linestyle='--', linewidth=2, label=f'{label_b} Mean')
    axes[0].set_title('Distribution Comparison')
    axes[0].legend()
    
    # Boxplot
    axes[1].boxplot([data_a, data_b], labels=[label_a, label_b], patch_artist=True)
    axes[1].set_title(f'Boxplot Comparison (p = {p_value:.4f})')
    axes[1].set_ylabel(title)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def select_postalcodes_with_claims(
    df: pd.DataFrame,
    province: str = "Gauteng",
    min_policies: int = 50,
    min_claims: int = 10,
    cover_type: str = None,
    vehicle_type: str = None,
    gender: str = None,
    top_n: int = 10,
):
    """
    Select statistically comparable postal codes for hypothesis testing.
    """

    subset = df.copy()

    # ---------------------------------------
    # APPLY FILTERS
    # ---------------------------------------
    subset = subset[subset["Province"] == province]

    if cover_type and "CoverType" in subset.columns:
        subset = subset[subset["CoverType"] == cover_type]

    if vehicle_type and "VehicleType" in subset.columns:
        subset = subset[subset["VehicleType"] == vehicle_type]

    if gender and "Gender" in subset.columns:
        subset = subset[subset["Gender"] == gender]

    print(f"\nRemaining rows after filtering: {len(subset)}")

    if len(subset) == 0:
        print("❌ No data left after filtering.")
        return None, None

    # ---------------------------------------
    # CLAIM INDICATOR
    # ---------------------------------------
    subset = subset.copy()
    subset["HasClaim"] = subset["ClaimSeverity"].notna()

    # ---------------------------------------
    # AGGREGATE STATS
    # ---------------------------------------
    stats = subset.groupby("PostalCode").agg(
        total_policies=("PostalCode", "count"),
        num_claims=("HasClaim", "sum"),
        claim_frequency=("HasClaim", "mean"),
        mean_severity=("ClaimSeverity", "mean"),
        std_severity=("ClaimSeverity", "std"),
    )

    print(f"Unique postal codes: {len(stats)}")

    # ---------------------------------------
    # FILTER VALID GROUPS
    # ---------------------------------------
    valid = stats[
        (stats["total_policies"] >= min_policies)
        & (stats["num_claims"] >= min_claims)
    ].dropna()

    print(f"Valid postal codes: {len(valid)}")

    # ---------------------------------------
    # AUTO-RELAX THRESHOLDS
    # ---------------------------------------
    if len(valid) < 2:

        print("\n⚠ Relaxing thresholds...")

        valid = stats[
            (stats["total_policies"] >= min_policies // 2)
            & (stats["num_claims"] >= max(5, min_claims // 2))
        ].dropna()

        print(f"Valid postal codes after relaxation: {len(valid)}")

    if len(valid) < 2:
        print("❌ Still not enough postal codes.")
        return None, None

    # ---------------------------------------
    # SELECT TOP RELIABLE POSTAL CODES
    # ---------------------------------------
    candidates = valid.nlargest(top_n, "total_policies")

    # ---------------------------------------
    # FIND MOST COMPARABLE PAIR
    # ---------------------------------------
    best_pair = None
    best_score = np.inf

    postal_codes = candidates.index.tolist()

    for i in range(len(postal_codes)):
        for j in range(i + 1, len(postal_codes)):

            pc1 = postal_codes[i]
            pc2 = postal_codes[j]

            policies_diff = abs(
                candidates.loc[pc1, "total_policies"]
                - candidates.loc[pc2, "total_policies"]
            )

            freq_diff = abs(
                candidates.loc[pc1, "claim_frequency"]
                - candidates.loc[pc2, "claim_frequency"]
            )

            score = policies_diff + (freq_diff * 1000)

            if score < best_score:
                best_score = score
                best_pair = (pc1, pc2)

    zip1, zip2 = best_pair

    # ---------------------------------------
    # REPORT
    # ---------------------------------------
    print("\n✅ Selected postal codes")
    print()

    for z in [zip1, zip2]:
        print(f"Postal Code      : {z}")
        print(f"Policies         : {valid.loc[z, 'total_policies']}")
        print(f"Claims           : {valid.loc[z, 'num_claims']}")
        print(f"Claim Frequency  : {valid.loc[z, 'claim_frequency']:.4f}")
        print(f"Mean Severity    : {valid.loc[z, 'mean_severity']:.2f}")
        print(f"Std Severity     : {valid.loc[z, 'std_severity']:.2f}")
        print()

    return zip1, zip2


# ====================== TEST FUNCTIONS ======================

def chi_squared_claim_frequency_test(df: pd.DataFrame, 
                                    group_col: str, 
                                    claim_col: str = 'HasClaim', 
                                    title: str = "",
                                    drop_categories: list = None):
    """Chi-Squared Test with optional category filtering (e.g. drop 'Unspecified')"""
    print_test_header("Chi-Squared Test - Claim Frequency")
    
    data = df.copy()
    
    # Drop unwanted categories (especially useful for Gender)
    if drop_categories:
        data = data[~data[group_col].isin(drop_categories)]
        print(f"→ Filtered out categories: {drop_categories}")
    
    contingency = pd.crosstab(data[group_col], data[claim_col])
    
    if contingency.shape[0] < 2:
        print("❌ Not enough groups for Chi-Squared test after filtering.")
        return None, False, None
    
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    
    print(f'H₀: No risk difference in Claim Frequency across {group_col}')
    print('H₁: There is a significant difference in Claim Frequency')
    print(f'Chi² Statistic : {chi2:.4f}')
    print(f'p-value        : {p_value:.6f}')
    
    rejected = interpret_decision(p_value)
    
    plot_chi_squared(contingency, p_value, title or f"Claim Frequency by {group_col}")
    
    return p_value, rejected, contingency


def t_test(df: pd.DataFrame, group_col: str, value_col: str, 
           group_a, group_b, title: str = "", min_sample: int = 30):
    """Improved T-Test with sample size warning"""
    print_test_header("Independent Samples T-Test")
    
    data_a = df[df[group_col] == group_a][value_col].dropna()
    data_b = df[df[group_col] == group_b][value_col].dropna()
    
    print(f"Sample Size - {group_a}: {len(data_a)} | {group_b}: {len(data_b)}")
    
    if len(data_a) < min_sample or len(data_b) < min_sample:
        print(f"⚠️ Warning: One or both groups have small sample size (< {min_sample})")
    
    if len(data_a) == 0 or len(data_b) == 0:
        print("❌ One group has no valid data. Test cannot be performed.")
        return None, False
    
    t_stat, p_value = ttest_ind(data_a, data_b, equal_var=False)
    
    print(f'H₀: Mean {value_col} is the same for {group_a} and {group_b}')
    print(f'H₁: Mean {value_col} differs')
    print(f'T-statistic : {t_stat:.4f}')
    print(f'p-value     : {p_value:.6f}')
    
    rejected = interpret_decision(p_value)
    
    if rejected and not pd.isna(p_value):
        diff_pct = ((data_b.mean() - data_a.mean()) / data_a.mean() * 100) if data_a.mean() != 0 else np.nan
        print(f"→ {group_b} has {diff_pct:+.1f}% {'higher' if diff_pct > 0 else 'lower'} average {value_col}")
    
    plot_numerical_comparison(data_a, data_b, str(group_a), str(group_b), p_value, 
                             title or f"{value_col} by {group_col}")
    
    return p_value, rejected


def anova_test(df: pd.DataFrame, group_col: str, value_col: str, 
               title: str = "", max_groups: int = 10, min_count: int = 30):
    """Improved ANOVA - limits number of groups + better visualization"""
    print_test_header("One-Way ANOVA")
    
    # Filter to groups with sufficient sample size
    group_sizes = df.groupby(group_col)[value_col].count()
    valid_groups = group_sizes[group_sizes >= min_count].nlargest(max_groups).index.tolist()
    
    filtered_df = df[df[group_col].isin(valid_groups)].copy()
    
    print(f"Testing {len(valid_groups)} largest groups (min {min_count} samples each)")
    
    groups = [group_data[value_col].dropna() for _, group_data in filtered_df.groupby(group_col)]
    
    if len(groups) < 2:
        print("❌ Not enough valid groups for ANOVA.")
        return None, False
    
    f_stat, p_value = f_oneway(*groups)
    
    print(f'H₀: Mean {value_col} is the same across {group_col}')
    print('H₁: At least one group differs')
    print(f'F-statistic : {f_stat:.4f}')
    print(f'p-value     : {p_value:.6f}')
    
    rejected = interpret_decision(p_value)
    
    # Better Plot
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=group_col, y=value_col, data=filtered_df, order=valid_groups)
    plt.title(f'{title or f"{value_col} by {group_col}"}\nANOVA p = {p_value:.4f}', 
              fontweight='bold')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return p_value, rejected
# ====================== RESULTS TABLE BUILDER ======================

def create_results_table(results: list) -> pd.DataFrame:
    """Build summary table for deliverables"""
    return pd.DataFrame(results, columns=[
        'Hypothesis', 'KPI', 'Test Used', 'p-value', 'Decision', 'Business Interpretation'
    ])