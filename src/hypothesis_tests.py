import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import numpy as np
from scipy.stats import norm, poisson

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
