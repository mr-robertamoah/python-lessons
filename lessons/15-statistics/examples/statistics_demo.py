"""
Lesson 13 Example 1: Statistics Demo
Demonstrates descriptive statistics and probability distributions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, binom, poisson

print("=== Statistics Demo ===\n")

# Set random seed for reproducibility
np.random.seed(42)

# 1. DESCRIPTIVE STATISTICS
print("1. DESCRIPTIVE STATISTICS")
print("=" * 40)

# Sample dataset: Employee salaries
salaries = np.array([45000, 52000, 48000, 65000, 58000, 72000, 51000, 
                    49000, 67000, 55000, 120000, 46000, 53000, 61000])

print("Employee Salaries Analysis:")
print(f"Data: {salaries}")
print(f"Mean: ${np.mean(salaries):,.0f}")
print(f"Median: ${np.median(salaries):,.0f}")
print(f"Mode: ${stats.mode(salaries, keepdims=True).mode[0]:,.0f}")
print(f"Standard Deviation: ${np.std(salaries, ddof=1):,.0f}")
print(f"Range: ${np.max(salaries) - np.min(salaries):,.0f}")

# Percentiles
percentiles = [25, 50, 75, 90, 95]
salary_percentiles = np.percentile(salaries, percentiles)
print("\nPercentiles:")
for p, value in zip(percentiles, salary_percentiles):
    print(f"  {p}th percentile: ${value:,.0f}")

# 2. PROBABILITY DISTRIBUTIONS
print("\n2. PROBABILITY DISTRIBUTIONS")
print("=" * 40)

# Normal Distribution Example
mu, sigma = 100, 15  # IQ scores
x = np.linspace(50, 150, 100)
y = norm.pdf(x, mu, sigma)

plt.figure(figsize=(12, 8))

# Plot 1: Normal Distribution
plt.subplot(2, 2, 1)
plt.plot(x, y, 'b-', linewidth=2, label='Normal Distribution')
plt.axvline(mu, color='r', linestyle='--', label=f'Mean = {mu}')
plt.axvline(mu - sigma, color='orange', linestyle='--', alpha=0.7, label='±1σ')
plt.axvline(mu + sigma, color='orange', linestyle='--', alpha=0.7)
plt.fill_between(x, y, alpha=0.3)
plt.title('Normal Distribution (IQ Scores)')
plt.xlabel('IQ Score')
plt.ylabel('Probability Density')
plt.legend()

# Calculate probabilities
prob_between_85_115 = norm.cdf(115, mu, sigma) - norm.cdf(85, mu, sigma)
print(f"P(85 < IQ < 115) = {prob_between_85_115:.3f} ({prob_between_85_115*100:.1f}%)")

# Plot 2: Binomial Distribution
plt.subplot(2, 2, 2)
n, p = 10, 0.3  # 10 trials, 30% success rate
x_binom = np.arange(0, 11)
y_binom = binom.pmf(x_binom, n, p)
plt.bar(x_binom, y_binom, alpha=0.7, color='green')
plt.title(f'Binomial Distribution (n={n}, p={p})')
plt.xlabel('Number of Successes')
plt.ylabel('Probability')

# Plot 3: Poisson Distribution
plt.subplot(2, 2, 3)
lam = 3  # Average events per interval
x_poisson = np.arange(0, 15)
y_poisson = poisson.pmf(x_poisson, lam)
plt.bar(x_poisson, y_poisson, alpha=0.7, color='orange')
plt.title(f'Poisson Distribution (λ={lam})')
plt.xlabel('Number of Events')
plt.ylabel('Probability')

# Plot 4: Sample vs Population
plt.subplot(2, 2, 4)
# Generate population and samples
population = np.random.normal(50, 10, 10000)
sample_means = []
for _ in range(1000):
    sample = np.random.choice(population, 30)
    sample_means.append(np.mean(sample))

plt.hist(population, bins=50, alpha=0.5, label='Population', density=True)
plt.hist(sample_means, bins=30, alpha=0.7, label='Sample Means', density=True)
plt.title('Central Limit Theorem')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()

plt.tight_layout()
plt.show()

# 3. HYPOTHESIS TESTING
print("\n3. HYPOTHESIS TESTING")
print("=" * 40)

# One-sample t-test
print("One-Sample t-test:")
sample_data = np.random.normal(52000, 8000, 30)  # Sample of 30 salaries
hypothesized_mean = 50000

t_stat, p_value = stats.ttest_1samp(sample_data, hypothesized_mean)
print(f"Sample mean: ${np.mean(sample_data):,.0f}")
print(f"Hypothesized mean: ${hypothesized_mean:,.0f}")
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.3f}")

alpha = 0.05
if p_value < alpha:
    print(f"Reject null hypothesis (p < {alpha})")
else:
    print(f"Fail to reject null hypothesis (p ≥ {alpha})")

# Two-sample t-test
print("\nTwo-Sample t-test:")
group_a = np.random.normal(100, 15, 50)  # Control group
group_b = np.random.normal(105, 15, 50)  # Treatment group

t_stat, p_value = stats.ttest_ind(group_a, group_b)
print(f"Group A mean: {np.mean(group_a):.2f}")
print(f"Group B mean: {np.mean(group_b):.2f}")
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.3f}")

# 4. CORRELATION ANALYSIS
print("\n4. CORRELATION ANALYSIS")
print("=" * 40)

# Generate correlated data
n_samples = 100
x1 = np.random.normal(50, 10, n_samples)
x2 = 2 * x1 + np.random.normal(0, 5, n_samples)  # Strong positive correlation
x3 = -0.5 * x1 + np.random.normal(50, 15, n_samples)  # Weak negative correlation
x4 = np.random.normal(50, 10, n_samples)  # No correlation

# Calculate correlations
corr_matrix = np.corrcoef([x1, x2, x3, x4])
print("Correlation Matrix:")
print(corr_matrix.round(3))

# Visualize correlations
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].scatter(x1, x2, alpha=0.6)
axes[0].set_title(f'Strong Positive Correlation\nr = {np.corrcoef(x1, x2)[0,1]:.3f}')
axes[0].set_xlabel('X1')
axes[0].set_ylabel('X2')

axes[1].scatter(x1, x3, alpha=0.6, color='orange')
axes[1].set_title(f'Weak Negative Correlation\nr = {np.corrcoef(x1, x3)[0,1]:.3f}')
axes[1].set_xlabel('X1')
axes[1].set_ylabel('X3')

axes[2].scatter(x1, x4, alpha=0.6, color='green')
axes[2].set_title(f'No Correlation\nr = {np.corrcoef(x1, x4)[0,1]:.3f}')
axes[2].set_xlabel('X1')
axes[2].set_ylabel('X4')

plt.tight_layout()
plt.show()

# 5. CONFIDENCE INTERVALS
print("\n5. CONFIDENCE INTERVALS")
print("=" * 40)

def confidence_interval_mean(data, confidence=0.95):
    """Calculate confidence interval for the mean"""
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)
    
    alpha = 1 - confidence
    t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
    
    margin_error = t_crit * std_err
    ci_lower = mean - margin_error
    ci_upper = mean + margin_error
    
    return mean, ci_lower, ci_upper, margin_error

# Example with sample data
sample = np.random.normal(75, 12, 25)
mean_val, ci_low, ci_high, margin = confidence_interval_mean(sample, 0.95)

print(f"Sample mean: {mean_val:.2f}")
print(f"95% Confidence Interval: [{ci_low:.2f}, {ci_high:.2f}]")
print(f"Margin of error: ±{margin:.2f}")
print(f"Interpretation: We are 95% confident the true population mean is between {ci_low:.2f} and {ci_high:.2f}")

# 6. PRACTICAL BUSINESS EXAMPLE
print("\n6. PRACTICAL BUSINESS EXAMPLE")
print("=" * 40)

# A/B Test Analysis
def ab_test_analysis(control_conversions, control_visitors, 
                    treatment_conversions, treatment_visitors):
    """Analyze A/B test results"""
    
    # Conversion rates
    control_rate = control_conversions / control_visitors
    treatment_rate = treatment_conversions / treatment_visitors
    
    # Pooled standard error
    pooled_rate = (control_conversions + treatment_conversions) / (control_visitors + treatment_visitors)
    se = np.sqrt(pooled_rate * (1 - pooled_rate) * (1/control_visitors + 1/treatment_visitors))
    
    # Z-test
    z_stat = (treatment_rate - control_rate) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    # Effect size
    lift = (treatment_rate - control_rate) / control_rate * 100
    
    return {
        'control_rate': control_rate,
        'treatment_rate': treatment_rate,
        'lift': lift,
        'z_statistic': z_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }

# Example A/B test
control_conv, control_vis = 45, 1000
treatment_conv, treatment_vis = 52, 1000

results = ab_test_analysis(control_conv, control_vis, treatment_conv, treatment_vis)

print("A/B Test Results:")
print(f"Control conversion rate: {results['control_rate']:.3f} ({results['control_rate']*100:.1f}%)")
print(f"Treatment conversion rate: {results['treatment_rate']:.3f} ({results['treatment_rate']*100:.1f}%)")
print(f"Lift: {results['lift']:.1f}%")
print(f"Z-statistic: {results['z_statistic']:.3f}")
print(f"P-value: {results['p_value']:.3f}")
print(f"Statistically significant: {results['significant']}")

print("\n=== Statistics Demo Complete ===")
print("Key concepts covered:")
print("• Descriptive statistics (mean, median, std, percentiles)")
print("• Probability distributions (normal, binomial, poisson)")
print("• Hypothesis testing (t-tests, z-tests)")
print("• Correlation analysis")
print("• Confidence intervals")
print("• A/B testing analysis")
