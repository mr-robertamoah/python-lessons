# Lesson 13 Solutions: Statistics for Data Analysis

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

print("=== GUIDED EXERCISE SOLUTIONS ===\n")

# Exercise 1: Descriptive Statistics Fundamentals
print("Exercise 1: Descriptive Statistics Fundamentals")
print("-" * 50)

test_scores = [78, 85, 92, 88, 76, 95, 82, 89, 91, 87, 83, 90, 86, 84, 93]

# Measures of central tendency
mean_score = np.mean(test_scores)
median_score = np.median(test_scores)
mode_result = stats.mode(test_scores)
mode_score = mode_result.mode[0] if len(mode_result.mode) > 0 else "No mode"

print("Measures of Central Tendency:")
print(f"  Mean: {mean_score:.2f}")
print(f"  Median: {median_score:.2f}")
print(f"  Mode: {mode_score}")

# Measures of spread
variance = np.var(test_scores, ddof=1)  # Sample variance
std_dev = np.std(test_scores, ddof=1)   # Sample standard deviation
range_score = np.max(test_scores) - np.min(test_scores)
iqr = np.percentile(test_scores, 75) - np.percentile(test_scores, 25)

print(f"\nMeasures of Spread:")
print(f"  Variance: {variance:.2f}")
print(f"  Standard Deviation: {std_dev:.2f}")
print(f"  Range: {range_score}")
print(f"  Interquartile Range (IQR): {iqr:.2f}")

# Interpretation
print(f"\nInterpretation:")
print(f"  The average test score is {mean_score:.1f} with a standard deviation of {std_dev:.1f}")
print(f"  This means most students scored between {mean_score-std_dev:.1f} and {mean_score+std_dev:.1f}")

# Exercise 2: Data Distribution Analysis
print("\n" + "="*50)
print("Exercise 2: Data Distribution Analysis")
print("-" * 50)

# Generate different distributions
np.random.seed(42)
normal_data = np.random.normal(50, 10, 1000)
skewed_data = np.random.exponential(2, 1000)
uniform_data = np.random.uniform(0, 100, 1000)

datasets = {
    'Normal': normal_data,
    'Skewed (Exponential)': skewed_data,
    'Uniform': uniform_data
}

print("Distribution Analysis:")
for name, data in datasets.items():
    print(f"\n{name} Distribution:")
    print(f"  Mean: {np.mean(data):.2f}")
    print(f"  Median: {np.median(data):.2f}")
    print(f"  Std Dev: {np.std(data):.2f}")
    print(f"  Skewness: {stats.skew(data):.2f}")
    print(f"  Kurtosis: {stats.kurtosis(data):.2f}")
    
    # Normality test
    shapiro_stat, shapiro_p = stats.shapiro(data[:100])  # Use subset for Shapiro-Wilk
    print(f"  Shapiro-Wilk p-value: {shapiro_p:.4f}")
    print(f"  Normal distribution: {'Yes' if shapiro_p > 0.05 else 'No'}")

# Exercise 3: Correlation and Relationships
print("\n" + "="*50)
print("Exercise 3: Correlation and Relationships")
print("-" * 50)

hours_studied = [2, 4, 6, 8, 3, 5, 7, 9, 1, 10]
test_scores_corr = [65, 75, 85, 90, 70, 80, 88, 95, 60, 98]

# Calculate correlations
pearson_corr, pearson_p = stats.pearsonr(hours_studied, test_scores_corr)
spearman_corr, spearman_p = stats.spearmanr(hours_studied, test_scores_corr)

print("Correlation Analysis:")
print(f"  Pearson correlation: {pearson_corr:.3f} (p-value: {pearson_p:.4f})")
print(f"  Spearman correlation: {spearman_corr:.3f} (p-value: {spearman_p:.4f})")

# Interpretation
if abs(pearson_corr) > 0.8:
    strength = "very strong"
elif abs(pearson_corr) > 0.6:
    strength = "strong"
elif abs(pearson_corr) > 0.4:
    strength = "moderate"
elif abs(pearson_corr) > 0.2:
    strength = "weak"
else:
    strength = "very weak"

direction = "positive" if pearson_corr > 0 else "negative"
print(f"  Interpretation: {strength} {direction} correlation")
print(f"  As hours studied increase, test scores tend to {'increase' if pearson_corr > 0 else 'decrease'}")

print("\n=== INDEPENDENT EXERCISE SOLUTIONS ===\n")

# Exercise 4: Sales Performance Analysis
print("Exercise 4: Sales Performance Analysis")
print("-" * 50)

# Generate sales data
np.random.seed(42)
regions = ['North', 'South', 'East', 'West']
quarters = ['Q1', 'Q2', 'Q3', 'Q4']

sales_data = []
for region in regions:
    for quarter in quarters:
        if region == 'North':
            sales = np.random.normal(100000, 15000)
        elif region == 'South':
            sales = np.random.normal(120000, 20000)
        elif region == 'East':
            sales = np.random.normal(90000, 10000)
        else:  # West
            sales = np.random.normal(110000, 25000)
        
        sales_data.append({
            'Region': region,
            'Quarter': quarter,
            'Sales': max(0, sales)
        })

sales_df = pd.DataFrame(sales_data)

# Analysis by region
print("Sales Analysis by Region:")
region_stats = sales_df.groupby('Region')['Sales'].agg(['mean', 'std', 'min', 'max']).round(0)
print(region_stats)

# Best and worst performing regions
best_region = region_stats['mean'].idxmax()
worst_region = region_stats['mean'].idxmin()
print(f"\nBest performing region: {best_region} (${region_stats.loc[best_region, 'mean']:,.0f})")
print(f"Worst performing region: {worst_region} (${region_stats.loc[worst_region, 'mean']:,.0f})")

# Quarterly trends
quarterly_stats = sales_df.groupby('Quarter')['Sales'].mean().round(0)
print(f"\nQuarterly trends:")
for quarter, avg_sales in quarterly_stats.items():
    print(f"  {quarter}: ${avg_sales:,.0f}")

# Outlier detection using IQR method
Q1 = sales_df['Sales'].quantile(0.25)
Q3 = sales_df['Sales'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = sales_df[(sales_df['Sales'] < lower_bound) | (sales_df['Sales'] > upper_bound)]
print(f"\nOutliers detected: {len(outliers)}")
if len(outliers) > 0:
    for _, row in outliers.iterrows():
        print(f"  {row['Region']} {row['Quarter']}: ${row['Sales']:,.0f}")

# Exercise 5: A/B Testing Analysis
print("\n" + "="*50)
print("Exercise 5: A/B Testing Analysis")
print("-" * 50)

# Simulate A/B test data
np.random.seed(42)

# Control group (Design A)
control_visitors = 1000
control_conversions = np.random.binomial(control_visitors, 0.12)

# Treatment group (Design B)  
treatment_visitors = 1000
treatment_conversions = np.random.binomial(treatment_visitors, 0.15)

# Calculate conversion rates
control_rate = control_conversions / control_visitors
treatment_rate = treatment_conversions / treatment_visitors

print("A/B Test Results:")
print(f"Control Group (A):")
print(f"  Visitors: {control_visitors}")
print(f"  Conversions: {control_conversions}")
print(f"  Conversion Rate: {control_rate:.3f} ({control_rate*100:.1f}%)")

print(f"\nTreatment Group (B):")
print(f"  Visitors: {treatment_visitors}")
print(f"  Conversions: {treatment_conversions}")
print(f"  Conversion Rate: {treatment_rate:.3f} ({treatment_rate*100:.1f}%)")

# Statistical significance test (two-proportion z-test)
def two_proportion_z_test(x1, n1, x2, n2):
    p1 = x1 / n1
    p2 = x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    z = (p2 - p1) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    return z, p_value, p1, p2

z_stat, p_value, p1, p2 = two_proportion_z_test(control_conversions, control_visitors, 
                                                treatment_conversions, treatment_visitors)

print(f"\nStatistical Test Results:")
print(f"  Z-statistic: {z_stat:.3f}")
print(f"  P-value: {p_value:.4f}")
print(f"  Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")

# Confidence intervals
def proportion_ci(x, n, confidence=0.95):
    p = x / n
    z = stats.norm.ppf((1 + confidence) / 2)
    se = np.sqrt(p * (1 - p) / n)
    margin = z * se
    return p - margin, p + margin

control_ci = proportion_ci(control_conversions, control_visitors)
treatment_ci = proportion_ci(treatment_conversions, treatment_visitors)

print(f"\n95% Confidence Intervals:")
print(f"  Control: [{control_ci[0]:.3f}, {control_ci[1]:.3f}]")
print(f"  Treatment: [{treatment_ci[0]:.3f}, {treatment_ci[1]:.3f}]")

# Practical significance
lift = (treatment_rate - control_rate) / control_rate
print(f"\nPractical Significance:")
print(f"  Absolute difference: {treatment_rate - control_rate:.3f}")
print(f"  Relative lift: {lift:.1%}")

# Exercise 6: Hypothesis Testing Suite
print("\n" + "="*50)
print("Exercise 6: Hypothesis Testing Suite")
print("-" * 50)

# One-sample t-test
print("1. One-Sample T-Test:")
print("   Testing if average delivery time < 30 minutes")
delivery_times = [28, 32, 25, 29, 31, 27, 33, 26, 30, 28, 29, 31, 27, 32, 26]

t_stat, p_value = stats.ttest_1samp(delivery_times, 30)
print(f"   Sample mean: {np.mean(delivery_times):.2f} minutes")
print(f"   T-statistic: {t_stat:.3f}")
print(f"   P-value: {p_value:.4f}")
print(f"   Conclusion: {'Reject H0' if p_value < 0.05 else 'Fail to reject H0'}")

# Two-sample t-test
print(f"\n2. Two-Sample T-Test:")
print("   Comparing customer satisfaction between two service levels")
service_a = [4.2, 4.5, 4.1, 4.3, 4.6, 4.0, 4.4, 4.2, 4.5, 4.3]
service_b = [4.6, 4.8, 4.5, 4.7, 4.9, 4.4, 4.6, 4.8, 4.7, 4.5]

t_stat, p_value = stats.ttest_ind(service_a, service_b)
print(f"   Service A mean: {np.mean(service_a):.2f}")
print(f"   Service B mean: {np.mean(service_b):.2f}")
print(f"   T-statistic: {t_stat:.3f}")
print(f"   P-value: {p_value:.4f}")
print(f"   Conclusion: {'Significant difference' if p_value < 0.05 else 'No significant difference'}")

# Chi-square test of independence
print(f"\n3. Chi-Square Test of Independence:")
print("   Testing relationship between gender and product preference")

# Create contingency table
observed = np.array([[30, 20, 15],   # Male preferences
                    [25, 35, 20]])   # Female preferences

chi2_stat, p_value, dof, expected = stats.chi2_contingency(observed)
print(f"   Chi-square statistic: {chi2_stat:.3f}")
print(f"   P-value: {p_value:.4f}")
print(f"   Degrees of freedom: {dof}")
print(f"   Conclusion: {'Dependent variables' if p_value < 0.05 else 'Independent variables'}")

# ANOVA
print(f"\n4. One-Way ANOVA:")
print("   Comparing sales across multiple regions")

region_north = [95, 102, 98, 105, 100, 97, 103, 99, 101, 96]
region_south = [110, 115, 108, 112, 118, 105, 113, 109, 116, 111]
region_east = [88, 92, 85, 90, 94, 87, 91, 89, 93, 86]

f_stat, p_value = stats.f_oneway(region_north, region_south, region_east)
print(f"   F-statistic: {f_stat:.3f}")
print(f"   P-value: {p_value:.4f}")
print(f"   Conclusion: {'Significant differences between groups' if p_value < 0.05 else 'No significant differences'}")

print("\n=== CHALLENGE EXERCISE SOLUTIONS ===\n")

# Challenge 1: Regression Analysis
print("Challenge 1: Simple Linear Regression")
print("-" * 50)

# Advertising spend vs Sales data
advertising_spend = [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500]
sales_revenue = [50000, 65000, 75000, 85000, 95000, 105000, 115000, 125000, 135000, 145000]

# Add some noise to make it more realistic
np.random.seed(42)
sales_revenue = [s + np.random.normal(0, 5000) for s in sales_revenue]

# Fit linear regression
X = np.array(advertising_spend).reshape(-1, 1)
y = np.array(sales_revenue)

model = LinearRegression()
model.fit(X, y)

# Calculate statistics
slope = model.coef_[0]
intercept = model.intercept_
r_squared = model.score(X, y)

# Correlation
correlation = np.corrcoef(advertising_spend, sales_revenue)[0, 1]

print("Regression Analysis Results:")
print(f"  Slope: {slope:.2f} (revenue increase per $1 advertising)")
print(f"  Intercept: {intercept:.2f} (baseline revenue)")
print(f"  R-squared: {r_squared:.3f} ({r_squared*100:.1f}% variance explained)")
print(f"  Correlation: {correlation:.3f}")

# Predictions
new_spend = 6000
predicted_revenue = model.predict([[new_spend]])[0]
print(f"\nPrediction:")
print(f"  For ${new_spend:,} advertising spend: ${predicted_revenue:,.0f} revenue")

# Challenge 2: Advanced Distribution Analysis
print("\n" + "="*50)
print("Challenge 2: Advanced Distribution Analysis")
print("-" * 50)

# Generate sample data
np.random.seed(42)
sample_data = np.random.gamma(2, 2, 1000)  # Gamma distribution

print("Distribution Fitting Analysis:")

# Test different distributions
distributions = [stats.norm, stats.gamma, stats.expon, stats.lognorm]
distribution_names = ['Normal', 'Gamma', 'Exponential', 'Log-Normal']

best_fit = None
best_p_value = 0

for dist, name in zip(distributions, distribution_names):
    # Fit distribution
    params = dist.fit(sample_data)
    
    # Kolmogorov-Smirnov test
    ks_stat, p_value = stats.kstest(sample_data, lambda x: dist.cdf(x, *params))
    
    print(f"  {name}: KS p-value = {p_value:.4f}")
    
    if p_value > best_p_value:
        best_fit = name
        best_p_value = p_value

print(f"\nBest fitting distribution: {best_fit} (p-value: {best_p_value:.4f})")

# Calculate percentiles
percentiles = [25, 50, 75, 90, 95, 99]
print(f"\nPercentiles of the data:")
for p in percentiles:
    value = np.percentile(sample_data, p)
    print(f"  {p}th percentile: {value:.2f}")

print("\n" + "="*50)
print("Statistics exercise solutions complete!")
print("\nKey concepts demonstrated:")
print("- Descriptive statistics and interpretation")
print("- Distribution analysis and normality testing")
print("- Correlation and relationship analysis")
print("- Comprehensive hypothesis testing suite")
print("- A/B testing with statistical significance")
print("- Regression analysis and prediction")
print("- Advanced distribution fitting")
print("- Practical business applications")
