"""
Lesson 12 Solutions: Data Visualization Exercises
Complete solutions for all exercises
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

print("=== LESSON 12 EXERCISE SOLUTIONS ===\n")

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

# ============================================================================
# GUIDED EXERCISES
# ============================================================================

print("GUIDED EXERCISE 1: Basic Plot Creation")
print("=" * 50)

# Temperature data
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
temps = [72, 75, 68, 80, 77, 73, 76]

plt.figure(figsize=(10, 6))
plt.plot(days, temps, marker='o', linewidth=2, markersize=8, color='steelblue')
plt.title('Weekly Temperature Trend', fontsize=14, fontweight='bold')
plt.xlabel('Day of Week', fontsize=12)
plt.ylabel('Temperature (°F)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('temperature_trend.png', dpi=300, bbox_inches='tight')
plt.show()

print("GUIDED EXERCISE 2: Comparative Bar Chart")
print("=" * 50)

quarters = ['Q1', 'Q2', 'Q3', 'Q4']
product_a = [120, 135, 148, 162]
product_b = [100, 110, 125, 140]
product_c = [80, 95, 105, 120]

x = np.arange(len(quarters))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 8))
bars1 = ax.bar(x - width, product_a, width, label='Product A', color='skyblue')
bars2 = ax.bar(x, product_b, width, label='Product B', color='lightcoral')
bars3 = ax.bar(x + width, product_c, width, label='Product C', color='lightgreen')

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height}', ha='center', va='bottom', fontweight='bold')

ax.set_xlabel('Quarter', fontsize=12)
ax.set_ylabel('Sales (thousands)', fontsize=12)
ax.set_title('Quarterly Sales Comparison by Product', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(quarters)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("GUIDED EXERCISE 3: Distribution Analysis")
print("=" * 50)

np.random.seed(42)
scores = np.random.normal(85, 12, 200)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Histogram
ax1.hist(scores, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
ax1.axvline(np.mean(scores), color='red', linestyle='--', linewidth=2, 
           label=f'Mean: {np.mean(scores):.1f}')
ax1.set_title('Distribution of Test Scores', fontsize=14, fontweight='bold')
ax1.set_xlabel('Score')
ax1.set_ylabel('Frequency')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Box plot
ax2.boxplot(scores, vert=True)
ax2.set_title('Box Plot of Test Scores', fontsize=14, fontweight='bold')
ax2.set_ylabel('Score')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# INDEPENDENT EXERCISES
# ============================================================================

print("INDEPENDENT EXERCISE 4: Sales Dashboard")
print("=" * 50)

# Sample data
np.random.seed(42)
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
online_sales = [45000, 52000, 48000, 61000, 58000, 67000]
store_sales = [38000, 41000, 39000, 45000, 47000, 52000]
categories = ['Electronics', 'Clothing', 'Books', 'Home', 'Sports']
category_sales = [125000, 89000, 45000, 78000, 63000]

# Create dashboard
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('E-commerce Sales Dashboard', fontsize=16, fontweight='bold')

# 1. Line plot: Monthly sales trend
axes[0, 0].plot(months, online_sales, marker='o', linewidth=2, label='Online Sales')
axes[0, 0].plot(months, store_sales, marker='s', linewidth=2, label='Store Sales')
axes[0, 0].set_title('Monthly Sales Trend')
axes[0, 0].set_ylabel('Sales ($)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Bar chart: Sales by category
bars = axes[0, 1].bar(categories, category_sales, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
axes[0, 1].set_title('Sales by Category')
axes[0, 1].set_ylabel('Total Sales ($)')
axes[0, 1].tick_params(axis='x', rotation=45)

# Add value labels
for bar, value in zip(bars, category_sales):
    axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2000,
                    f'${value:,}', ha='center', va='bottom', fontweight='bold')

# 3. Pie chart: Market share by category
axes[1, 0].pie(category_sales, labels=categories, autopct='%1.1f%%', startangle=90,
               colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
axes[1, 0].set_title('Market Share by Category')

# 4. Stacked bar chart: Online vs store sales
width = 0.6
axes[1, 1].bar(months, online_sales, width, label='Online', color='skyblue')
axes[1, 1].bar(months, store_sales, width, bottom=online_sales, label='Store', color='lightcoral')
axes[1, 1].set_title('Online vs Store Sales by Month')
axes[1, 1].set_ylabel('Sales ($)')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('sales_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()

print("INDEPENDENT EXERCISE 5: Customer Analysis Visualization")
print("=" * 50)

# Generate customer dataset
np.random.seed(42)
n_customers = 300

customers = pd.DataFrame({
    'age': np.random.normal(35, 12, n_customers),
    'income': np.random.normal(50000, 15000, n_customers),
    'spending': np.random.normal(2000, 600, n_customers),
    'satisfaction': np.random.uniform(1, 10, n_customers),
    'segment': np.random.choice(['Premium', 'Standard', 'Budget'], n_customers)
})

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Customer Analysis Dashboard', fontsize=16, fontweight='bold')

# 1. Scatter plot: Income vs Spending by segment
for segment in customers['segment'].unique():
    segment_data = customers[customers['segment'] == segment]
    axes[0, 0].scatter(segment_data['income'], segment_data['spending'], 
                      label=segment, alpha=0.6, s=50)
axes[0, 0].set_xlabel('Income ($)')
axes[0, 0].set_ylabel('Spending ($)')
axes[0, 0].set_title('Income vs Spending by Segment')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Box plot: Age distribution by segment
sns.boxplot(data=customers, x='segment', y='age', ax=axes[0, 1])
axes[0, 1].set_title('Age Distribution by Segment')

# 3. Correlation heatmap
numeric_cols = ['age', 'income', 'spending', 'satisfaction']
corr_matrix = customers[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 0])
axes[1, 0].set_title('Correlation Matrix')

# 4. Distribution plot: Satisfaction by segment
for segment in customers['segment'].unique():
    segment_data = customers[customers['segment'] == segment]
    axes[1, 1].hist(segment_data['satisfaction'], alpha=0.6, label=segment, bins=15)
axes[1, 1].set_xlabel('Satisfaction Score')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Satisfaction Distribution by Segment')
axes[1, 1].legend()

plt.tight_layout()
plt.show()

print("INDEPENDENT EXERCISE 6: Time Series Visualization")
print("=" * 50)

# Generate time series data
dates = pd.date_range('2024-01-01', periods=365, freq='D')
trend = np.linspace(1000, 1500, 365)
seasonal = 200 * np.sin(2 * np.pi * np.arange(365) / 7)
noise = np.random.normal(0, 50, 365)
traffic = trend + seasonal + noise

time_df = pd.DataFrame({'date': dates, 'traffic': traffic})
time_df.set_index('date', inplace=True)

# Create time series plots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Website Traffic Analysis', fontsize=16, fontweight='bold')

# 1. Daily traffic
axes[0, 0].plot(time_df.index, time_df['traffic'], alpha=0.7, linewidth=1)
axes[0, 0].set_title('Daily Website Traffic')
axes[0, 0].set_ylabel('Visitors')
axes[0, 0].grid(True, alpha=0.3)

# 2. Weekly average
weekly_avg = time_df.resample('W')['traffic'].mean()
axes[0, 1].plot(weekly_avg.index, weekly_avg.values, marker='o', linewidth=2)
axes[0, 1].set_title('Weekly Average Traffic')
axes[0, 1].set_ylabel('Average Visitors')
axes[0, 1].grid(True, alpha=0.3)

# 3. Monthly totals
monthly_totals = time_df.resample('M')['traffic'].sum()
axes[1, 0].bar(range(len(monthly_totals)), monthly_totals.values, 
               color='steelblue', alpha=0.7)
axes[1, 0].set_title('Monthly Total Traffic')
axes[1, 0].set_ylabel('Total Visitors')
axes[1, 0].set_xticks(range(len(monthly_totals)))
axes[1, 0].set_xticklabels([d.strftime('%b') for d in monthly_totals.index], rotation=45)

# 4. Moving averages
time_df['7_day_avg'] = time_df['traffic'].rolling(window=7).mean()
time_df['30_day_avg'] = time_df['traffic'].rolling(window=30).mean()

axes[1, 1].plot(time_df.index, time_df['traffic'], alpha=0.3, label='Daily', linewidth=1)
axes[1, 1].plot(time_df.index, time_df['7_day_avg'], label='7-day MA', linewidth=2)
axes[1, 1].plot(time_df.index, time_df['30_day_avg'], label='30-day MA', linewidth=2)
axes[1, 1].set_title('Traffic with Moving Averages')
axes[1, 1].set_ylabel('Visitors')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("=== ALL EXERCISES COMPLETED SUCCESSFULLY ===")
print("\nKey Visualization Principles Applied:")
print("✓ Clear titles and labels")
print("✓ Appropriate chart types for data")
print("✓ Consistent color schemes")
print("✓ Professional formatting")
print("✓ Proper legends and annotations")
print("✓ Grid lines for readability")
print("✓ High-resolution output for publication")
