#!/usr/bin/env python3
"""
A/B Test Sample Size Calculator for Email Campaigns
====================================================
This script calculates the required sample size and test duration for
A/B testing email campaigns on repo-stage customers.
"""

import math
from scipy import stats


class ABTestCalculator:
    """Calculator for A/B test sample sizes and duration."""

    def __init__(self,
                 baseline_rate,
                 minimum_detectable_effect,
                 alpha=0.05,
                 power=0.80,
                 daily_volume=400,
                 treatment_split=0.50):
        """
        Initialize the A/B test calculator.

        Parameters:
        -----------
        baseline_rate : float
            Historical conversion/success rate (e.g., 0.20 for 20%)
        minimum_detectable_effect : float
            Absolute minimum effect to detect (e.g., 0.05 for 5 percentage points)
        alpha : float, default=0.05
            Significance level (Type I error rate)
        power : float, default=0.80
            Statistical power (1 - Type II error rate)
        daily_volume : int, default=400
            Number of new eligible accounts per day
        treatment_split : float, default=0.50
            Proportion allocated to treatment group (0.50 = 50/50 split)
        """
        self.p1 = baseline_rate
        self.mde = minimum_detectable_effect
        self.p2 = baseline_rate + minimum_detectable_effect
        self.alpha = alpha
        self.power = power
        self.daily_volume = daily_volume
        self.treatment_split = treatment_split

        # Validate inputs
        self._validate_inputs()

    def _validate_inputs(self):
        """Validate input parameters."""
        if not 0 < self.p1 < 1:
            raise ValueError("baseline_rate must be between 0 and 1")
        if not 0 < self.p2 <= 1:
            raise ValueError("baseline_rate + MDE must be between 0 and 1")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        if not 0 < self.power < 1:
            raise ValueError("power must be between 0 and 1")
        if self.daily_volume <= 0:
            raise ValueError("daily_volume must be positive")
        if not 0 < self.treatment_split < 1:
            raise ValueError("treatment_split must be between 0 and 1")

    def calculate_sample_size(self):
        """
        Calculate required sample size per group using two-proportion formula.

        Returns:
        --------
        dict : Dictionary containing all calculation results
        """
        # Calculate z-scores
        z_alpha = stats.norm.ppf(1 - self.alpha / 2)  # Two-sided test
        z_beta = stats.norm.ppf(self.power)

        # Calculate pooled proportion
        p_bar = (self.p1 + self.p2) / 2

        # Calculate variance components
        variance = p_bar * (1 - p_bar)

        # Sample size per group formula
        numerator = 2 * variance * (z_alpha + z_beta) ** 2
        denominator = (self.p2 - self.p1) ** 2

        n_per_group = numerator / denominator

        # Total sample size
        total_n = 2 * n_per_group

        # Calculate test duration
        control_per_day = self.daily_volume * (1 - self.treatment_split)
        treatment_per_day = self.daily_volume * self.treatment_split

        days_needed_treatment = n_per_group / treatment_per_day
        days_needed_control = n_per_group / control_per_day
        days_needed = max(days_needed_treatment, days_needed_control)

        # Calculate relative lift
        relative_lift = (self.p2 - self.p1) / self.p1 * 100

        return {
            'baseline_rate_pct': self.p1 * 100,
            'target_rate_pct': self.p2 * 100,
            'absolute_lift_pp': self.mde * 100,
            'relative_lift_pct': relative_lift,
            'alpha': self.alpha,
            'power': self.power,
            'z_alpha': z_alpha,
            'z_beta': z_beta,
            'pooled_proportion': p_bar,
            'n_per_group': math.ceil(n_per_group),
            'total_n': math.ceil(total_n),
            'daily_volume': self.daily_volume,
            'treatment_split': self.treatment_split,
            'control_per_day': control_per_day,
            'treatment_per_day': treatment_per_day,
            'days_needed': math.ceil(days_needed)
        }

    def print_results(self):
        """Print formatted results."""
        results = self.calculate_sample_size()

        print("\n" + "="*70)
        print("A/B TEST SAMPLE SIZE CALCULATION RESULTS")
        print("="*70)

        print("\n📊 TEST PARAMETERS:")
        print(f"  Baseline Rate (Control):        {results['baseline_rate_pct']:.1f}%")
        print(f"  Target Rate (Treatment):        {results['target_rate_pct']:.1f}%")
        print(f"  Minimum Detectable Effect:      {results['absolute_lift_pp']:.1f} percentage points")
        print(f"  Relative Lift:                  {results['relative_lift_pct']:.1f}%")
        print(f"  Significance Level (α):         {results['alpha']}")
        print(f"  Statistical Power (1-β):        {results['power']}")

        print("\n📈 STATISTICAL VALUES:")
        print(f"  Z-score (α/2):                  {results['z_alpha']:.3f}")
        print(f"  Z-score (β):                    {results['z_beta']:.3f}")
        print(f"  Pooled Proportion (p̄):          {results['pooled_proportion']:.3f}")

        print("\n👥 SAMPLE SIZE REQUIREMENTS:")
        print(f"  Accounts per Group:             {results['n_per_group']:,}")
        print(f"  Total Accounts Needed:          {results['total_n']:,}")

        print("\n⏱️  TEST DURATION:")
        print(f"  Daily Volume:                   {results['daily_volume']:,} accounts/day")
        print(f"  Treatment Split:                {results['treatment_split']*100:.0f}% / {(1-results['treatment_split'])*100:.0f}%")
        print(f"  Control Group:                  {results['control_per_day']:.0f} accounts/day")
        print(f"  Treatment Group:                {results['treatment_per_day']:.0f} accounts/day")
        print(f"  Days Needed:                    {results['days_needed']} days")

        print("\n💡 RECOMMENDATIONS:")
        if results['days_needed'] <= 7:
            print(f"  ✓ Test duration of {results['days_needed']} days is feasible")
            print(f"    A 1-week test should provide sufficient data")
        elif results['days_needed'] <= 14:
            print(f"  ✓ Test duration of {results['days_needed']} days is reasonable")
            print(f"    A 2-week test window is recommended")
        else:
            print(f"  ⚠ Test duration of {results['days_needed']} days is lengthy")
            print(f"    Consider:")
            print(f"    • Increasing MDE (detecting larger effects)")
            print(f"    • Reducing power to 0.70")
            print(f"    • Extending test window if acceptable")

        print("\n" + "="*70 + "\n")

        return results


def calculate_baseline_rate(total_accounts, successful_accounts):
    """
    Calculate baseline rate from historical data.

    Parameters:
    -----------
    total_accounts : int
        Total number of accounts in historical period
    successful_accounts : int
        Number of accounts that achieved the KPI

    Returns:
    --------
    float : Baseline rate (proportion)
    """
    if total_accounts <= 0:
        raise ValueError("total_accounts must be positive")
    if successful_accounts < 0 or successful_accounts > total_accounts:
        raise ValueError("successful_accounts must be between 0 and total_accounts")

    baseline_rate = successful_accounts / total_accounts
    print(f"\n📋 Historical Baseline Calculation:")
    print(f"   Total Accounts: {total_accounts:,}")
    print(f"   Successful Accounts: {successful_accounts:,}")
    print(f"   Baseline Rate: {baseline_rate:.2%} ({baseline_rate:.4f})")

    return baseline_rate


def sensitivity_analysis(baseline_rate, daily_volume=400, alpha=0.05, power=0.80):
    """
    Run sensitivity analysis for different MDE values.

    Parameters:
    -----------
    baseline_rate : float
        Historical conversion rate
    daily_volume : int
        Daily account volume
    alpha : float
        Significance level
    power : float
        Statistical power
    """
    print("\n" + "="*70)
    print("SENSITIVITY ANALYSIS - Varying Minimum Detectable Effect")
    print("="*70)

    mde_values = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]

    print(f"\n{'MDE (pp)':<12} {'Target %':<12} {'Rel. Lift':<12} {'n/group':<12} {'Total n':<12} {'Days':<8}")
    print("-" * 70)

    for mde in mde_values:
        if baseline_rate + mde > 1.0:
            continue

        calc = ABTestCalculator(
            baseline_rate=baseline_rate,
            minimum_detectable_effect=mde,
            alpha=alpha,
            power=power,
            daily_volume=daily_volume
        )
        results = calc.calculate_sample_size()

        print(f"{mde*100:<12.1f} {results['target_rate_pct']:<12.1f} "
              f"{results['relative_lift_pct']:<12.1f} {results['n_per_group']:<12,} "
              f"{results['total_n']:<12,} {results['days_needed']:<8}")

    print("="*70 + "\n")


# Example usage and main execution
if __name__ == "__main__":
    print("\n" + "="*70)
    print("EMAIL A/B TEST CALCULATOR - REPO-STAGE CUSTOMERS")
    print("="*70)

    # ============================================================================
    # CONFIGURABLE VARIABLES - MODIFY THESE FOR YOUR EXPERIMENT
    # ============================================================================

    # Historical baseline calculation (optional - if you have raw data)
    # Uncomment and modify if needed:
    # baseline_rate = calculate_baseline_rate(
    #     total_accounts=1500,
    #     successful_accounts=300
    # )

    # Or set baseline rate directly:
    baseline_rate = 0.20  # 20% cure rate

    # Test parameters
    minimum_detectable_effect = 0.05  # 5 percentage points (absolute)
    significance_level = 0.05  # α = 0.05 (5% false positive rate)
    statistical_power = 0.80  # 80% power

    # Volume and split
    daily_account_volume = 400  # accounts entering repo per day
    treatment_allocation = 0.50  # 50% to treatment, 50% to control

    # ============================================================================
    # RUN CALCULATION
    # ============================================================================

    calculator = ABTestCalculator(
        baseline_rate=baseline_rate,
        minimum_detectable_effect=minimum_detectable_effect,
        alpha=significance_level,
        power=statistical_power,
        daily_volume=daily_account_volume,
        treatment_split=treatment_allocation
    )

    results = calculator.print_results()

    # ============================================================================
    # SENSITIVITY ANALYSIS (optional)
    # ============================================================================

    # Run sensitivity analysis to see how different MDE values affect sample size
    sensitivity_analysis(
        baseline_rate=baseline_rate,
        daily_volume=daily_account_volume,
        alpha=significance_level,
        power=statistical_power
    )

    # ============================================================================
    # CUSTOM SCENARIOS (optional)
    # ============================================================================

    # Example: What if we only have 300 accounts per day?
    print("\n" + "="*70)
    print("SCENARIO: Lower Daily Volume (300 accounts/day)")
    print("="*70)

    calculator_low_volume = ABTestCalculator(
        baseline_rate=baseline_rate,
        minimum_detectable_effect=minimum_detectable_effect,
        alpha=significance_level,
        power=statistical_power,
        daily_volume=300,
        treatment_split=treatment_allocation
    )

    calculator_low_volume.print_results()

    # Example: What if we accept 70% power?
    print("\n" + "="*70)
    print("SCENARIO: Lower Power (70% instead of 80%)")
    print("="*70)

    calculator_low_power = ABTestCalculator(
        baseline_rate=baseline_rate,
        minimum_detectable_effect=minimum_detectable_effect,
        alpha=significance_level,
        power=0.70,
        daily_volume=daily_account_volume,
        treatment_split=treatment_allocation
    )

    calculator_low_power.print_results()
