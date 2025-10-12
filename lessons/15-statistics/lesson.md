# Lesson 13: Descriptive Statistics and Probability

## Learning Objectives
By the end of this lesson, you will be able to:
- Calculate and interpret descriptive statistics
- Understand probability distributions and their applications
- Perform correlation analysis between variables
- Conduct basic hypothesis testing
- Calculate confidence intervals
- Apply statistical concepts to real-world data analysis

## Why Statistics Matter in Data Analysis

### Making Sense of Data
```python
# Raw numbers don't tell the whole story
sales_data = [1200, 1350, 1100, 1400, 1250, 1600, 1450, 1300, 1750, 1550]

# But statistics reveal patterns
import numpy as np
print(f"Average: ${np.mean(sales_data):,.0f}")
print(f"Growth trend: {np.corrcoef(range(len(sales_data)), sales_data)[0,1]:.3f}")
print(f"Variability: {np.std(sales_data)/np.mean(sales_data)*100:.1f}%")
```

### Statistical Thinking
- **Descriptive**: What happened? (summarize data)
- **Inferential**: What does it mean? (draw conclusions)
- **Predictive**: What will happen? (forecast future)

## Descriptive Statistics

### Measures of Central Tendency
```python
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# Sample dataset: Employee salaries
salaries = np.array([45000, 52000, 48000, 65000, 58000, 72000, 51000, 
                    49000, 67000, 55000, 120000, 46000, 53000, 61000])

# Mean (average)
mean_salary = np.mean(salaries)
print(f"Mean salary: ${mean_salary:,.0f}")

# Median (middle value)
median_salary = np.median(salaries)
print(f"Median salary: ${median_salary:,.0f}")

# Mode (most frequent value)
mode_result = stats.mode(salaries, keepdims=True)
print(f"Mode salary: ${mode_result.mode[0]:,.0f}")

# When to use each:
# Mean: Normal distribution, no extreme outliers
# Median: Skewed data, presence of outliers
# Mode: Categorical data, finding most common value
```

### Measures of Spread (Variability)
```python
# Range
salary_range = np.max(salaries) - np.min(salaries)
print(f"Salary range: ${salary_range:,.0f}")

# Variance
variance = np.var(salaries, ddof=1)  # ddof=1 for sample variance
print(f"Variance: {variance:,.0f}")

# Standard deviation
std_dev = np.std(salaries, ddof=1)
print(f"Standard deviation: ${std_dev:,.0f}")

# Coefficient of variation (relative variability)
cv = (std_dev / mean_salary) * 100
print(f"Coefficient of variation: {cv:.1f}%")

# Interquartile Range (IQR)
q1 = np.percentile(salaries, 25)
q3 = np.percentile(salaries, 75)
iqr = q3 - q1
print(f"IQR: ${iqr:,.0f} (Q1: ${q1:,.0f}, Q3: ${q3:,.0f})")
```

### Percentiles and Quartiles
```python
# Calculate percentiles
percentiles = [10, 25, 50, 75, 90, 95, 99]
salary_percentiles = np.percentile(salaries, percentiles)

print("Salary Percentiles:")
for p, value in zip(percentiles, salary_percentiles):
    print(f"{p}th percentile: ${value:,.0f}")

# Five-number summary
five_num_summary = {
    'Min': np.min(salaries),
    'Q1': np.percentile(salaries, 25),
    'Median': np.median(salaries),
    'Q3': np.percentile(salaries, 75),
    'Max': np.max(salaries)
}

print("\nFive-Number Summary:")
for stat, value in five_num_summary.items():
    print(f"{stat}: ${value:,.0f}")
```

### Skewness and Kurtosis
```python
from scipy.stats import skew, kurtosis

# Skewness (measure of asymmetry)
salary_skew = skew(salaries)
print(f"Skewness: {salary_skew:.3f}")

# Interpretation:
# > 0: Right-skewed (tail extends to right)
# < 0: Left-skewed (tail extends to left)
# ≈ 0: Symmetric

# Kurtosis (measure of tail heaviness)
salary_kurtosis = kurtosis(salaries)
print(f"Kurtosis: {salary_kurtosis:.3f}")

# Interpretation:
# > 0: Heavy tails (more outliers)
# < 0: Light tails (fewer outliers)
# ≈ 0: Normal distribution tails
```

## Probability Distributions

### Normal Distribution
```python
from scipy.stats import norm
import matplotlib.pyplot as plt

# Generate normal distribution
mu, sigma = 100, 15  # mean and standard deviation
x = np.linspace(50, 150, 100)
y = norm.pdf(x, mu, sigma)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2, label='Normal Distribution')
plt.axvline(mu, color='r', linestyle='--', label=f'Mean = {mu}')
plt.axvline(mu - sigma, color='orange', linestyle='--', alpha=0.7, label=f'±1σ')
plt.axvline(mu + sigma, color='orange', linestyle='--', alpha=0.7)
plt.fill_between(x, y, alpha=0.3)
plt.title('Normal Distribution (μ=100, σ=15)')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Calculate probabilities
prob_less_than_85 = norm.cdf(85, mu, sigma)
prob_between_85_115 = norm.cdf(115, mu, sigma) - norm.cdf(85, mu, sigma)
print(f"P(X < 85) = {prob_less_than_85:.3f}")
print(f"P(85 < X < 115) = {prob_between_85_115:.3f}")

# 68-95-99.7 Rule
print(f"68% of data within 1σ: [{mu-sigma:.1f}, {mu+sigma:.1f}]")
print(f"95% of data within 2σ: [{mu-2*sigma:.1f}, {mu+2*sigma:.1f}]")
print(f"99.7% of data within 3σ: [{mu-3*sigma:.1f}, {mu+3*sigma:.1f}]")
```

### Other Important Distributions
```python
from scipy.stats import binom, poisson, uniform, expon

# Binomial Distribution (discrete)
# Example: 10 coin flips, probability of heads = 0.5
n, p = 10, 0.5
x_binom = np.arange(0, 11)
y_binom = binom.pmf(x_binom, n, p)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.bar(x_binom, y_binom, alpha=0.7)
plt.title('Binomial Distribution (n=10, p=0.5)')
plt.xlabel('Number of Successes')
plt.ylabel('Probability')

# Poisson Distribution (discrete)
# Example: Number of emails per hour
lambda_param = 3
x_poisson = np.arange(0, 15)
y_poisson = poisson.pmf(x_poisson, lambda_param)

plt.subplot(2, 2, 2)
plt.bar(x_poisson, y_poisson, alpha=0.7, color='orange')
plt.title(f'Poisson Distribution (λ={lambda_param})')
plt.xlabel('Number of Events')
plt.ylabel('Probability')

# Uniform Distribution (continuous)
a, b = 0, 10  # range
x_uniform = np.linspace(-1, 11, 100)
y_uniform = uniform.pdf(x_uniform, a, b-a)

plt.subplot(2, 2, 3)
plt.plot(x_uniform, y_uniform, 'g-', linewidth=2)
plt.title(f'Uniform Distribution [{a}, {b}]')
plt.xlabel('Value')
plt.ylabel('Probability Density')

# Exponential Distribution (continuous)
# Example: Time between events
scale = 2
x_exp = np.linspace(0, 10, 100)
y_exp = expon.pdf(x_exp, scale=scale)

plt.subplot(2, 2, 4)
plt.plot(x_exp, y_exp, 'r-', linewidth=2)
plt.title(f'Exponential Distribution (λ={1/scale})')
plt.xlabel('Time')
plt.ylabel('Probability Density')

plt.tight_layout()
plt.show()
```

## Correlation Analysis

### Pearson Correlation
```python
# Generate correlated data
np.random.seed(42)
n = 100

# Strong positive correlation
x1 = np.random.normal(50, 10, n)
y1 = 2 * x1 + np.random.normal(0, 5, n)

# Weak negative correlation  
x2 = np.random.normal(50, 10, n)
y2 = -0.5 * x2 + np.random.normal(50, 15, n)

# No correlation
x3 = np.random.normal(50, 10, n)
y3 = np.random.normal(50, 10, n)

# Calculate correlations
corr1 = np.corrcoef(x1, y1)[0, 1]
corr2 = np.corrcoef(x2, y2)[0, 1]
corr3 = np.corrcoef(x3, y3)[0, 1]

# Visualize correlations
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].scatter(x1, y1, alpha=0.6)
axes[0].set_title(f'Strong Positive Correlation\nr = {corr1:.3f}')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')

axes[1].scatter(x2, y2, alpha=0.6, color='orange')
axes[1].set_title(f'Weak Negative Correlation\nr = {corr2:.3f}')
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')

axes[2].scatter(x3, y3, alpha=0.6, color='green')
axes[2].set_title(f'No Correlation\nr = {corr3:.3f}')
axes[2].set_xlabel('X')
axes[2].set_ylabel('Y')

plt.tight_layout()
plt.show()

# Interpretation of correlation coefficients:
# |r| = 1.0: Perfect correlation
# |r| = 0.7-0.9: Strong correlation
# |r| = 0.3-0.7: Moderate correlation
# |r| = 0.1-0.3: Weak correlation
# |r| = 0.0: No linear correlation
```

### Correlation Matrix
```python
# Create sample dataset
np.random.seed(42)
data = pd.DataFrame({
    'Sales': np.random.normal(1000, 200, 100),
    'Marketing_Spend': np.random.normal(500, 100, 100),
    'Temperature': np.random.normal(70, 15, 100),
    'Customer_Satisfaction': np.random.uniform(1, 10, 100)
})

# Add some relationships
data['Sales'] = data['Sales'] + 0.8 * data['Marketing_Spend'] + np.random.normal(0, 50, 100)
data['Customer_Satisfaction'] = 5 + 0.002 * data['Sales'] + np.random.normal(0, 1, 100)

# Calculate correlation matrix
corr_matrix = data.corr()
print("Correlation Matrix:")
print(corr_matrix.round(3))

# Visualize correlation matrix
plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Hide upper triangle
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.3f', cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Correlation Matrix Heatmap')
plt.tight_layout()
plt.show()
```

## Hypothesis Testing

### One-Sample t-test
```python
from scipy.stats import ttest_1samp

# Example: Test if average customer satisfaction is significantly different from 7
satisfaction_scores = np.random.normal(7.2, 1.5, 50)

# Null hypothesis: μ = 7
# Alternative hypothesis: μ ≠ 7
hypothesized_mean = 7
t_statistic, p_value = ttest_1samp(satisfaction_scores, hypothesized_mean)

print(f"Sample mean: {np.mean(satisfaction_scores):.3f}")
print(f"Hypothesized mean: {hypothesized_mean}")
print(f"t-statistic: {t_statistic:.3f}")
print(f"p-value: {p_value:.3f}")

alpha = 0.05
if p_value < alpha:
    print(f"Reject null hypothesis (p < {alpha})")
    print("The sample mean is significantly different from 7")
else:
    print(f"Fail to reject null hypothesis (p ≥ {alpha})")
    print("No significant difference from 7")
```

### Two-Sample t-test
```python
from scipy.stats import ttest_ind

# Example: Compare sales between two regions
region_a_sales = np.random.normal(1200, 200, 30)
region_b_sales = np.random.normal(1100, 180, 35)

# Test if there's a significant difference
t_stat, p_val = ttest_ind(region_a_sales, region_b_sales)

print(f"Region A mean: ${np.mean(region_a_sales):.0f}")
print(f"Region B mean: ${np.mean(region_b_sales):.0f}")
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_val:.3f}")

if p_val < 0.05:
    print("Significant difference between regions")
else:
    print("No significant difference between regions")
```

### Chi-Square Test
```python
from scipy.stats import chi2_contingency

# Example: Test relationship between product preference and age group
# Contingency table
observed = np.array([[20, 30, 25],   # Young: Product A, B, C
                    [25, 25, 20],   # Middle: Product A, B, C  
                    [15, 20, 30]])  # Senior: Product A, B, C

chi2_stat, p_val, dof, expected = chi2_contingency(observed)

print("Observed frequencies:")
print(observed)
print(f"\nChi-square statistic: {chi2_stat:.3f}")
print(f"p-value: {p_val:.3f}")
print(f"Degrees of freedom: {dof}")

if p_val < 0.05:
    print("Significant association between age and product preference")
else:
    print("No significant association")
```

## Confidence Intervals

### Confidence Interval for Mean
```python
from scipy.stats import t

def confidence_interval_mean(data, confidence=0.95):
    """Calculate confidence interval for the mean"""
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)  # Standard error of mean
    
    # t-critical value
    alpha = 1 - confidence
    t_crit = t.ppf(1 - alpha/2, df=n-1)
    
    # Margin of error
    margin_error = t_crit * std_err
    
    # Confidence interval
    ci_lower = mean - margin_error
    ci_upper = mean + margin_error
    
    return mean, ci_lower, ci_upper, margin_error

# Example: Customer satisfaction scores
satisfaction = np.random.normal(7.5, 1.2, 50)

mean_sat, ci_low, ci_high, margin = confidence_interval_mean(satisfaction, 0.95)

print(f"Sample mean: {mean_sat:.3f}")
print(f"95% Confidence Interval: [{ci_low:.3f}, {ci_high:.3f}]")
print(f"Margin of error: ±{margin:.3f}")
print(f"Interpretation: We are 95% confident that the true population mean")
print(f"lies between {ci_low:.3f} and {ci_high:.3f}")
```

### Bootstrap Confidence Intervals
```python
def bootstrap_ci(data, n_bootstrap=1000, confidence=0.95):
    """Calculate bootstrap confidence interval"""
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    # Calculate percentiles
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_means, 100 * alpha/2)
    ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha/2))
    
    return ci_lower, ci_upper, bootstrap_means

# Example with skewed data
skewed_data = np.random.exponential(2, 100)

boot_low, boot_high, boot_means = bootstrap_ci(skewed_data)

print(f"Original mean: {np.mean(skewed_data):.3f}")
print(f"Bootstrap 95% CI: [{boot_low:.3f}, {boot_high:.3f}]")

# Visualize bootstrap distribution
plt.figure(figsize=(10, 6))
plt.hist(boot_means, bins=50, alpha=0.7, density=True)
plt.axvline(boot_low, color='red', linestyle='--', label=f'CI Lower: {boot_low:.3f}')
plt.axvline(boot_high, color='red', linestyle='--', label=f'CI Upper: {boot_high:.3f}')
plt.axvline(np.mean(skewed_data), color='black', linewidth=2, label=f'Sample Mean: {np.mean(skewed_data):.3f}')
plt.title('Bootstrap Distribution of Sample Means')
plt.xlabel('Sample Mean')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Practical Statistical Analysis

### Example: A/B Test Analysis
```python
def ab_test_analysis(control_group, treatment_group, alpha=0.05):
    """Comprehensive A/B test analysis"""
    
    # Descriptive statistics
    control_stats = {
        'n': len(control_group),
        'mean': np.mean(control_group),
        'std': np.std(control_group, ddof=1),
        'median': np.median(control_group)
    }
    
    treatment_stats = {
        'n': len(treatment_group),
        'mean': np.mean(treatment_group),
        'std': np.std(treatment_group, ddof=1),
        'median': np.median(treatment_group)
    }
    
    # Statistical test
    t_stat, p_value = ttest_ind(treatment_group, control_group)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((control_stats['n']-1)*control_stats['std']**2 + 
                         (treatment_stats['n']-1)*treatment_stats['std']**2) / 
                        (control_stats['n'] + treatment_stats['n'] - 2))
    cohens_d = (treatment_stats['mean'] - control_stats['mean']) / pooled_std
    
    # Confidence interval for difference
    diff_mean = treatment_stats['mean'] - control_stats['mean']
    se_diff = pooled_std * np.sqrt(1/control_stats['n'] + 1/treatment_stats['n'])
    df = control_stats['n'] + treatment_stats['n'] - 2
    t_crit = t.ppf(1 - alpha/2, df)
    ci_diff = (diff_mean - t_crit*se_diff, diff_mean + t_crit*se_diff)
    
    return {
        'control': control_stats,
        'treatment': treatment_stats,
        'difference': diff_mean,
        'p_value': p_value,
        'significant': p_value < alpha,
        'effect_size': cohens_d,
        'ci_difference': ci_diff
    }

# Example: Website conversion rates
np.random.seed(42)
control_conversions = np.random.binomial(1, 0.12, 1000)  # 12% conversion rate
treatment_conversions = np.random.binomial(1, 0.15, 1000)  # 15% conversion rate

results = ab_test_analysis(control_conversions, treatment_conversions)

print("A/B Test Results:")
print(f"Control group: {results['control']['mean']:.3f} conversion rate")
print(f"Treatment group: {results['treatment']['mean']:.3f} conversion rate")
print(f"Difference: {results['difference']:.3f}")
print(f"95% CI for difference: [{results['ci_difference'][0]:.3f}, {results['ci_difference'][1]:.3f}]")
print(f"p-value: {results['p_value']:.3f}")
print(f"Statistically significant: {results['significant']}")
print(f"Effect size (Cohen's d): {results['effect_size']:.3f}")
```

## Key Statistical Concepts

### Type I and Type II Errors
```python
# Simulation to demonstrate error types
def simulate_hypothesis_test(true_effect=0, sample_size=30, alpha=0.05, n_simulations=1000):
    """Simulate hypothesis tests to show error rates"""
    
    type_1_errors = 0  # False positives
    type_2_errors = 0  # False negatives
    correct_rejections = 0
    correct_acceptances = 0
    
    for _ in range(n_simulations):
        # Generate data
        if true_effect == 0:
            # Null hypothesis is true
            sample = np.random.normal(0, 1, sample_size)
            null_is_true = True
        else:
            # Alternative hypothesis is true
            sample = np.random.normal(true_effect, 1, sample_size)
            null_is_true = False
        
        # Perform t-test
        t_stat, p_value = ttest_1samp(sample, 0)
        reject_null = p_value < alpha
        
        # Count error types
        if null_is_true and reject_null:
            type_1_errors += 1
        elif null_is_true and not reject_null:
            correct_acceptances += 1
        elif not null_is_true and reject_null:
            correct_rejections += 1
        elif not null_is_true and not reject_null:
            type_2_errors += 1
    
    return {
        'type_1_rate': type_1_errors / n_simulations,
        'type_2_rate': type_2_errors / n_simulations,
        'power': correct_rejections / n_simulations if true_effect != 0 else None
    }

# Demonstrate Type I error rate (should be ≈ α)
results_null = simulate_hypothesis_test(true_effect=0)
print(f"Type I error rate (α = 0.05): {results_null['type_1_rate']:.3f}")

# Demonstrate statistical power
results_alt = simulate_hypothesis_test(true_effect=0.5)
print(f"Statistical power (effect = 0.5): {results_alt['power']:.3f}")
print(f"Type II error rate (β): {results_alt['type_2_rate']:.3f}")
```

## Key Terminology

- **Population**: The entire group being studied
- **Sample**: A subset of the population
- **Parameter**: A numerical characteristic of a population (μ, σ)
- **Statistic**: A numerical characteristic of a sample (x̄, s)
- **p-value**: Probability of observing results as extreme as observed, assuming null hypothesis is true
- **Confidence Interval**: Range of plausible values for a population parameter
- **Effect Size**: Magnitude of difference between groups
- **Statistical Power**: Probability of detecting an effect when it exists

## Looking Ahead

In Lesson 14, we'll learn about:
- **Feature Engineering**: Creating and transforming variables for analysis
- **Categorical Encoding**: Converting categories to numbers
- **Feature Scaling**: Normalizing data for machine learning
- **Feature Selection**: Choosing the most important variables
- **Dimensionality Reduction**: Reducing the number of features while preserving information
