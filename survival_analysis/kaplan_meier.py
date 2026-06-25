"""
This module is designed to contain the scripts necessary for the following purposes:
1. Create stratified Kaplan-Meier plots based on specific features.
2. Create overall Kaplan-Meier plots for the entire cohort.
"""
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts
from lifelines.utils import median_survival_times
import matplotlib.pyplot as plt
import numpy as np

def plot_stratified_km(df, duration_col, event_col, feature_name, title_suffix=""):
    """
    Create a stratified Kaplan-Meier plot using a median split on one feature.

    Why this approach:
    - The cohort is divided into "high" and "low" groups based on the median
      of the selected feature, which is a common simple way to compare survival
      between two subgroups.
    - Kaplan-Meier curves are used because they handle censored follow-up times.
    - The plot includes:
        1. survival curves,
        2. censor tick marks,
        3. confidence intervals,
        4. median survival reference lines,
        5. a log-rank p-value,
        6. and a number-at-risk table.
    """

    # Split the cohort into two groups using the median value of the feature.
    # Patients above the median go into the "High" group, and those at or below
    # the median go into the "Low" group.
    cutoff = df[feature_name].median()
    masks = {
        f'High {feature_name} (>{cutoff:.2f})': df[feature_name] > cutoff,
        f'Low {feature_name} (≤{cutoff:.2f})': df[feature_name] <= cutoff
    }

    # Create the plotting canvas.
    # A slightly larger figure is useful because we will add an at-risk table below.
    plt.figure(figsize=(10, 7))
    ax = plt.subplot(111)

    # Fixed color mapping makes the plot consistent across runs and across features.
    # This helps with interpretation when multiple KM plots are compared in a report.
    color_map = {
        'High': 'tab:orange',
        'Low': 'tab:blue'
    }

    # Store fitted KM objects so they can later be passed to add_at_risk_counts().
    fitted_models = []

    # Loop over the two strata (High / Low).
    for label, mask in masks.items():
        T_sub = df.loc[mask, duration_col]
        E_sub = df.loc[mask, event_col]

        # Skip empty groups to avoid fitting errors.
        if len(T_sub) == 0:
            continue

        # Create a NEW KaplanMeierFitter for each group.
        # This is important: if the same object is reused, the second fit overwrites
        # the first one, and both plotted groups / risk tables may become incorrect.
        kmf = KaplanMeierFitter()

        # Keep colors consistent by matching group name.
        current_color = color_map['High'] if 'High' in label else color_map['Low']

        # Fit the KM estimator:
        # - T_sub contains time to event or censoring
        # - E_sub indicates whether the event happened (1) or was censored (0)
        kmf.fit(T_sub, event_observed=E_sub, label=label)

        # Plot the survival curve.
        # show_censors=True adds tick marks at the last observed time for censored patients.
        # ci_show=True adds confidence intervals around the survival estimate.
        kmf.plot_survival_function(
            ax=ax,
            color=current_color,
            show_censors=True,
            ci_show=True
        )

        # Save the fitted object for the risk table below the plot.
        fitted_models.append(kmf)

        # Median survival time is the time where survival probability crosses 0.5.
        median_time = kmf.median_survival_time_

        # Confidence intervals for the median survival time can be derived from the
        # KM confidence interval object.
        ci_df = median_survival_times(kmf.confidence_interval_)

        # Only plot median reference lines if the median is actually reached.
        # If the curve never drops to 0.5, the median survival is not estimable.
        if np.isfinite(median_time):
            # Horizontal helper line at survival = 0.5
            plt.plot(
                [0, median_time], [0.5, 0.5],
                color=current_color, linestyle='--', linewidth=1, alpha=0.6
            )

            # Vertical helper line from the median time down to the x-axis
            plt.plot(
                [median_time, median_time], [0, 0.5],
                color=current_color, linestyle='--', linewidth=1, alpha=0.6
            )

            # Mark the median point itself
            plt.scatter(
                [median_time], [0.5],
                color=current_color, s=80, zorder=5, edgecolors='white',
                label=f'Median {label}: {median_time:.0f}d'
            )

        # Print summary statistics to the console.
        # This is useful for reporting survival medians and CIs in text or tables.
        lower_ci = ci_df.iloc[0, 0]
        upper_ci = ci_df.iloc[0, 1]
        print(f"\n{label} Median {title_suffix}: {median_time} days")
        print(f"95% CI: {lower_ci} - {upper_ci} days")

    # Perform a log-rank test to compare the two survival distributions.
    # This tests the null hypothesis that the survival curves are the same.
    group_keys = list(masks.keys())
    results = logrank_test(
        df.loc[masks[group_keys[0]], duration_col],
        df.loc[masks[group_keys[1]], duration_col],
        event_observed_A=df.loc[masks[group_keys[0]], event_col],
        event_observed_B=df.loc[masks[group_keys[1]], event_col]
    )

    # Add the number-at-risk table below the KM plot.
    # This shows how many patients are still under observation and event-free
    # just before each displayed time point.
    add_at_risk_counts(*fitted_models, ax=ax)

    # Axis labels and visual formatting.
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Survival Probability")
    ax.set_ylim([0, 1.05])
    ax.legend(loc='best', fontsize='small')

    # A faint reference line at 50% survival makes the median easier to read.
    plt.axhline(0.5, color='black', lw=0.5, alpha=0.3)

    # Include the log-rank p-value in the title so the statistical comparison is
    # visible directly on the figure.
    plt.title(
        f"KM Stratified by {feature_name} ({title_suffix})\n"
        f"Log-Rank p-value: {results.p_value:.4f}"
    )

    plt.grid(True, linestyle=':', alpha=0.4)

    # Tight layout helps prevent overlap between the main plot and the at-risk table.
    plt.tight_layout()
    plt.show()



def plot_overall_km(df, duration_col, event_col, title_suffix=""):
    """
    Creates an overall Kaplan-Meier plot for the full cohort and marks the median.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing duration and event columns.
    duration_col : str
        Name of the column representing duration until event or censoring.
    event_col : str
        Name of the column representing event occurrence (1 if event occurred, 0 if censored
    title_suffix : str, optional
        Suffix to add to the plot title for context.

    Returns:
    --------
    KaplanMeierFitter
        The fitted KaplanMeierFitter object for further analysis if needed.
    """
    # 1. Setup Plot
    plt.figure(figsize=(9, 6))
    ax = plt.subplot(111)
    kmf = KaplanMeierFitter()
    
    # 2. Fit and Plot
    kmf.fit(df[duration_col], event_observed=df[event_col], label=f'Overall {title_suffix}')
    line = kmf.plot_survival_function(ax=ax, color='teal') # Using a neutral color for overall
    color = 'teal'
    
    # 3. Calculate Median Statistics
    median_time = kmf.median_survival_time_
    ci_df = median_survival_times(kmf.confidence_interval_)
    
    # 4. Visualizing the Median Point
    if np.isfinite(median_time):
        # Horizontal line to 50%
        plt.plot([0, median_time], [0.5, 0.5], color=color, linestyle='--', linewidth=1, alpha=0.6)
        # Vertical line down to X-axis
        plt.plot([median_time, median_time], [0, 0.5], color=color, linestyle='--', linewidth=1, alpha=0.6)
        # The Median Point
        plt.scatter([median_time], [0.5], color=color, s=60, zorder=5, 
                    label=f'Median: {median_time} days')
    
    # 5. Formatting for the report
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Survival Probability")
    ax.set_ylim([0, 1.05])
    plt.axhline(0.5, color='black', lw=0.5, alpha=0.3) # 50% reference line
    plt.title(f"Overall {title_suffix} (Full Cohort, n={len(df)})")
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.legend(loc='best')
    
    # 6. Printing Numerical Results for the Results Section
    lower_ci = ci_df.iloc[0, 0]
    upper_ci = ci_df.iloc[0, 1]
    print(f"--- {title_suffix} Summary ---")
    print(f"Median Survival: {median_time} days")
    print(f"95% CI: {lower_ci} - {upper_ci} days\n")
    
    plt.show()
    
    return kmf