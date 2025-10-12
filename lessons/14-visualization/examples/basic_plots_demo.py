"""
Lesson 12 Example 1: Basic Plots Demo
Demonstrates fundamental plotting with matplotlib and seaborn
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

print("=== Basic Plots Demo ===\n")

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

# Generate sample data
np.random.seed(42)

# 1. LINE PLOT
print("1. Creating Line Plot...")
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
sales = [100, 120, 95, 140, 110, 160, 145, 130, 175, 155, 190, 180]

plt.figure(figsize=(10, 6))
plt.plot(months, sales, marker='o', linewidth=2, markersize=8)
plt.title('Monthly Sales Performance', fontsize=14, fontweight='bold')
plt.xlabel('Month')
plt.ylabel('Sales ($000)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('line_plot_demo.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. BAR CHART
print("2. Creating Bar Chart...")
products = ['Product A', 'Product B', 'Product C', 'Product D']
revenue = [250000, 180000, 320000, 150000]

plt.figure(figsize=(10, 6))
bars = plt.bar(products, revenue, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
plt.title('Product Revenue Comparison', fontsize=14, fontweight='bold')
plt.ylabel('Revenue ($)')

# Add value labels on bars
for bar, value in zip(bars, revenue):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
             f'${value:,}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('bar_chart_demo.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. HISTOGRAM
print("3. Creating Histogram...")
# Generate normal distribution data
data = np.random.normal(100, 15, 1000)

plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
plt.title('Distribution of Test Scores', fontsize=14, fontweight='bold')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.axvline(np.mean(data), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(data):.1f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('histogram_demo.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. SCATTER PLOT
print("4. Creating Scatter Plot...")
# Generate correlated data
n_points = 100
x = np.random.normal(50, 15, n_points)
y = 2 * x + np.random.normal(0, 20, n_points)

plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.6, s=60, color='purple', edgecolors='white', linewidth=1)
plt.title('Relationship Between Variables', fontsize=14, fontweight='bold')
plt.xlabel('X Variable')
plt.ylabel('Y Variable')

# Add trend line
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x, p(x), "r--", alpha=0.8, linewidth=2, label='Trend Line')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('scatter_plot_demo.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. PIE CHART
print("5. Creating Pie Chart...")
market_share = [35, 25, 20, 12, 8]
companies = ['Company A', 'Company B', 'Company C', 'Company D', 'Others']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

plt.figure(figsize=(10, 8))
wedges, texts, autotexts = plt.pie(market_share, labels=companies, colors=colors,
                                   autopct='%1.1f%%', startangle=90,
                                   explode=(0.05, 0, 0, 0, 0))
plt.title('Market Share Distribution', fontsize=14, fontweight='bold')

# Enhance text
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

plt.axis('equal')
plt.tight_layout()
plt.savefig('pie_chart_demo.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. SEABORN STATISTICAL PLOTS
print("6. Creating Seaborn Statistical Plots...")

# Generate sample dataset
n_samples = 500
df = pd.DataFrame({
    'category': np.random.choice(['A', 'B', 'C'], n_samples),
    'value1': np.random.normal(50, 15, n_samples),
    'value2': np.random.normal(30, 10, n_samples),
    'group': np.random.choice(['Group 1', 'Group 2'], n_samples)
})

# Create subplot figure
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Seaborn Statistical Plots', fontsize=16, fontweight='bold')

# Box plot
sns.boxplot(data=df, x='category', y='value1', ax=axes[0, 0])
axes[0, 0].set_title('Box Plot: Value1 by Category')

# Violin plot
sns.violinplot(data=df, x='category', y='value1', hue='group', ax=axes[0, 1])
axes[0, 1].set_title('Violin Plot: Value1 by Category and Group')

# Correlation heatmap
corr_data = df[['value1', 'value2']].corr()
sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, ax=axes[1, 0])
axes[1, 0].set_title('Correlation Heatmap')

# Distribution plot
sns.histplot(data=df, x='value1', hue='group', kde=True, ax=axes[1, 1])
axes[1, 1].set_title('Distribution Plot with KDE')

plt.tight_layout()
plt.savefig('seaborn_plots_demo.png', dpi=300, bbox_inches='tight')
plt.show()

print("All plots created successfully!")
print("Saved plot files:")
print("- line_plot_demo.png")
print("- bar_chart_demo.png") 
print("- histogram_demo.png")
print("- scatter_plot_demo.png")
print("- pie_chart_demo.png")
print("- seaborn_plots_demo.png")
