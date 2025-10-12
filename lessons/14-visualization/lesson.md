# Lesson 12: Data Visualization - Creating Charts and Graphs

## Learning Objectives
By the end of this lesson, you will be able to:
- Create basic plots using matplotlib
- Use seaborn for statistical visualizations
- Customize plots with labels, colors, and styles
- Choose appropriate chart types for different data
- Create publication-ready visualizations
- Build interactive dashboards with basic techniques

## Why Data Visualization Matters

### The Power of Visual Communication
```python
# Numbers alone don't tell the story
sales_data = [100, 120, 95, 140, 110, 160, 145, 130, 175, 155, 190, 180]

# But a chart reveals the trend immediately
import matplotlib.pyplot as plt
plt.plot(sales_data)
plt.title('Monthly Sales Growth')
plt.show()
```

### Anscombe's Quartet - Why Visualization is Essential
Four datasets with identical statistics but completely different patterns:
- Same mean, variance, correlation
- Completely different when visualized
- Demonstrates why we must always plot our data

## Setting Up Visualization Libraries

### Installation
```bash
pip install matplotlib seaborn plotly
```

### Import Conventions
```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")
```

## Matplotlib Fundamentals

### Basic Plot Types
```python
# Line plot
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.title('Sine Wave')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.show()

# Scatter plot
x = np.random.randn(100)
y = np.random.randn(100)
plt.scatter(x, y, alpha=0.6)
plt.title('Random Scatter Plot')
plt.show()

# Bar chart
categories = ['A', 'B', 'C', 'D']
values = [23, 45, 56, 78]
plt.bar(categories, values)
plt.title('Category Comparison')
plt.show()

# Histogram
data = np.random.normal(100, 15, 1000)
plt.hist(data, bins=30, alpha=0.7)
plt.title('Normal Distribution')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()
```

### Customizing Plots
```python
# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot with customization
x = range(12)
sales = [100, 120, 95, 140, 110, 160, 145, 130, 175, 155, 190, 180]

ax.plot(x, sales, marker='o', linewidth=2, markersize=8, color='steelblue')
ax.set_title('Monthly Sales Performance', fontsize=16, fontweight='bold')
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Sales ($000)', fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xticks(x)
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

plt.tight_layout()
plt.show()
```

## Seaborn for Statistical Plots

### Distribution Plots
```python
# Sample data
tips = sns.load_dataset('tips')

# Histogram with KDE
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
sns.histplot(tips['total_bill'], kde=True)
plt.title('Distribution of Total Bill')

plt.subplot(1, 3, 2)
sns.boxplot(y=tips['total_bill'])
plt.title('Box Plot of Total Bill')

plt.subplot(1, 3, 3)
sns.violinplot(y=tips['total_bill'])
plt.title('Violin Plot of Total Bill')

plt.tight_layout()
plt.show()
```

### Relationship Plots
```python
# Scatter plot with regression line
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
sns.scatterplot(data=tips, x='total_bill', y='tip')
plt.title('Bill vs Tip')

plt.subplot(1, 3, 2)
sns.regplot(data=tips, x='total_bill', y='tip')
plt.title('Bill vs Tip (with regression)')

plt.subplot(1, 3, 3)
sns.heatmap(tips.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')

plt.tight_layout()
plt.show()
```

### Categorical Plots
```python
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.barplot(data=tips, x='day', y='total_bill')
plt.title('Average Bill by Day')

plt.subplot(1, 3, 2)
sns.boxplot(data=tips, x='day', y='total_bill')
plt.title('Bill Distribution by Day')

plt.subplot(1, 3, 3)
sns.countplot(data=tips, x='day')
plt.title('Number of Customers by Day')

plt.tight_layout()
plt.show()
```

## Practical Visualization Examples

### Example 1: Sales Dashboard
```python
# Generate sample sales data
np.random.seed(42)
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
products = ['Product A', 'Product B', 'Product C', 'Product D']

# Create comprehensive sales dashboard
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Sales Dashboard 2024', fontsize=16, fontweight='bold')

# 1. Monthly sales trend
monthly_sales = np.random.normal(100000, 15000, 12)
monthly_sales = np.cumsum(np.random.normal(5000, 2000, 12)) + 80000
axes[0, 0].plot(months, monthly_sales, marker='o', linewidth=2)
axes[0, 0].set_title('Monthly Sales Trend')
axes[0, 0].set_ylabel('Sales ($)')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(True, alpha=0.3)

# 2. Product performance
product_sales = np.random.normal([250000, 180000, 320000, 150000], 20000, 4)
bars = axes[0, 1].bar(products, product_sales, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
axes[0, 1].set_title('Product Performance')
axes[0, 1].set_ylabel('Total Sales ($)')
for bar, value in zip(bars, product_sales):
    axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
                     f'${value:,.0f}', ha='center', va='bottom')

# 3. Regional distribution (pie chart)
regions = ['North', 'South', 'East', 'West']
regional_sales = [300000, 250000, 200000, 150000]
axes[1, 0].pie(regional_sales, labels=regions, autopct='%1.1f%%', startangle=90)
axes[1, 0].set_title('Sales by Region')

# 4. Sales distribution histogram
all_sales = np.random.normal(5000, 1500, 1000)
axes[1, 1].hist(all_sales, bins=30, alpha=0.7, color='steelblue')
axes[1, 1].set_title('Individual Sale Distribution')
axes[1, 1].set_xlabel('Sale Amount ($)')
axes[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()
```

### Example 2: Customer Analysis
```python
# Generate customer data
np.random.seed(42)
n_customers = 1000

customer_data = pd.DataFrame({
    'age': np.random.normal(35, 12, n_customers),
    'income': np.random.normal(50000, 15000, n_customers),
    'spending': np.random.normal(2000, 800, n_customers),
    'satisfaction': np.random.normal(7.5, 1.5, n_customers),
    'segment': np.random.choice(['Premium', 'Standard', 'Budget'], n_customers, p=[0.2, 0.5, 0.3])
})

# Ensure realistic ranges
customer_data['age'] = np.clip(customer_data['age'], 18, 80)
customer_data['income'] = np.clip(customer_data['income'], 20000, 150000)
customer_data['spending'] = np.clip(customer_data['spending'], 100, 8000)
customer_data['satisfaction'] = np.clip(customer_data['satisfaction'], 1, 10)

# Create customer analysis plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Customer Analysis Dashboard', fontsize=16, fontweight='bold')

# Age distribution by segment
sns.boxplot(data=customer_data, x='segment', y='age', ax=axes[0, 0])
axes[0, 0].set_title('Age Distribution by Segment')

# Income vs Spending
sns.scatterplot(data=customer_data, x='income', y='spending', 
                hue='segment', alpha=0.6, ax=axes[0, 1])
axes[0, 1].set_title('Income vs Spending by Segment')

# Satisfaction by segment
sns.violinplot(data=customer_data, x='segment', y='satisfaction', ax=axes[0, 2])
axes[0, 2].set_title('Satisfaction by Segment')

# Correlation heatmap
numeric_cols = ['age', 'income', 'spending', 'satisfaction']
corr_matrix = customer_data[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 0])
axes[1, 0].set_title('Correlation Matrix')

# Spending distribution
axes[1, 1].hist(customer_data['spending'], bins=30, alpha=0.7, color='green')
axes[1, 1].set_title('Spending Distribution')
axes[1, 1].set_xlabel('Monthly Spending ($)')
axes[1, 1].set_ylabel('Number of Customers')

# Segment distribution
segment_counts = customer_data['segment'].value_counts()
axes[1, 2].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%')
axes[1, 2].set_title('Customer Segment Distribution')

plt.tight_layout()
plt.show()
```

## Advanced Visualization Techniques

### Subplots and Multiple Plots
```python
# Create complex multi-plot figure
fig = plt.figure(figsize=(16, 10))

# Define grid layout
gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

# Main plot (spans multiple cells)
ax_main = fig.add_subplot(gs[0:2, 0:3])
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
ax_main.plot(x, y1, label='sin(x)', linewidth=2)
ax_main.plot(x, y2, label='cos(x)', linewidth=2)
ax_main.set_title('Trigonometric Functions', fontsize=14)
ax_main.legend()
ax_main.grid(True, alpha=0.3)

# Side histogram
ax_hist = fig.add_subplot(gs[0:2, 3])
data = np.random.normal(0, 1, 1000)
ax_hist.hist(data, bins=30, orientation='horizontal', alpha=0.7)
ax_hist.set_title('Distribution')

# Bottom plots
ax_bottom1 = fig.add_subplot(gs[2, 0:2])
categories = ['A', 'B', 'C', 'D', 'E']
values = np.random.randint(10, 100, 5)
ax_bottom1.bar(categories, values, color='skyblue')
ax_bottom1.set_title('Category Analysis')

ax_bottom2 = fig.add_subplot(gs[2, 2:4])
x_scatter = np.random.randn(100)
y_scatter = np.random.randn(100)
colors = np.random.rand(100)
ax_bottom2.scatter(x_scatter, y_scatter, c=colors, alpha=0.6, cmap='viridis')
ax_bottom2.set_title('Scatter Analysis')

plt.show()
```

### Styling and Themes
```python
# Custom styling
plt.style.use('seaborn-v0_8')  # Use seaborn style
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.patch.set_facecolor('white')

# Styled line plot
x = np.linspace(0, 10, 50)
for i, color in enumerate(colors[:3]):
    y = np.sin(x + i)
    axes[0, 0].plot(x, y, color=color, linewidth=3, label=f'Series {i+1}')
axes[0, 0].set_title('Styled Line Plot', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Styled bar chart
categories = ['Q1', 'Q2', 'Q3', 'Q4']
values = [85, 92, 78, 95]
bars = axes[0, 1].bar(categories, values, color=colors[:4])
axes[0, 1].set_title('Quarterly Performance', fontsize=14, fontweight='bold')
axes[0, 1].set_ylim(0, 100)

# Add value labels on bars
for bar, value in zip(bars, values):
    axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'{value}%', ha='center', va='bottom', fontweight='bold')

# Styled scatter plot
x = np.random.normal(50, 15, 200)
y = np.random.normal(50, 15, 200)
axes[1, 0].scatter(x, y, alpha=0.6, s=60, color=colors[0], edgecolors='white', linewidth=1)
axes[1, 0].set_title('Customer Segments', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Age')
axes[1, 0].set_ylabel('Income (k$)')

# Styled pie chart
sizes = [30, 25, 20, 15, 10]
labels = ['Product A', 'Product B', 'Product C', 'Product D', 'Others']
wedges, texts, autotexts = axes[1, 1].pie(sizes, labels=labels, colors=colors,
                                          autopct='%1.1f%%', startangle=90,
                                          explode=(0.05, 0, 0, 0, 0))
axes[1, 1].set_title('Market Share', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()
```

## Best Practices for Data Visualization

### 1. Choose the Right Chart Type
```python
# Guidelines for chart selection
chart_guide = {
    'Comparison': ['Bar chart', 'Column chart', 'Radar chart'],
    'Distribution': ['Histogram', 'Box plot', 'Violin plot'],
    'Relationship': ['Scatter plot', 'Line chart', 'Heatmap'],
    'Composition': ['Pie chart', 'Stacked bar', 'Treemap'],
    'Trend': ['Line chart', 'Area chart', 'Slope graph']
}
```

### 2. Color and Accessibility
```python
# Use colorblind-friendly palettes
import matplotlib.colors as mcolors

# Good color palettes
colorblind_friendly = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
sequential = plt.cm.Blues(np.linspace(0.3, 1, 5))
diverging = plt.cm.RdYlBu(np.linspace(0, 1, 5))
```

### 3. Clear Labels and Titles
```python
def create_professional_plot(x, y, title, xlabel, ylabel):
    """Create a professional-looking plot with proper labels"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(x, y, linewidth=2, marker='o', markersize=6)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Add grid and styling
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add data source and date
    ax.text(0.99, 0.01, 'Source: Company Data | Updated: 2024', 
            transform=ax.transAxes, ha='right', va='bottom', 
            fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    return fig, ax
```

## Interactive Visualizations

### Basic Interactivity with Matplotlib
```python
# Interactive plot with annotations
fig, ax = plt.subplots(figsize=(10, 6))

# Sample data
companies = ['Apple', 'Microsoft', 'Amazon', 'Google', 'Tesla']
revenue = [365, 168, 386, 257, 54]
profit_margin = [25.9, 34.1, 5.3, 22.5, 6.8]

# Create scatter plot
scatter = ax.scatter(revenue, profit_margin, s=200, alpha=0.7, 
                    c=range(len(companies)), cmap='viridis')

# Add company labels
for i, company in enumerate(companies):
    ax.annotate(company, (revenue[i], profit_margin[i]), 
                xytext=(5, 5), textcoords='offset points',
                fontsize=10, fontweight='bold')

ax.set_xlabel('Revenue (Billions $)', fontsize=12)
ax.set_ylabel('Profit Margin (%)', fontsize=12)
ax.set_title('Tech Companies: Revenue vs Profit Margin', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Key Terminology

- **Figure**: The entire window or page that contains plots
- **Axes**: The area where data is plotted (what we usually call "the plot")
- **Artist**: Everything you can see on the figure (lines, text, etc.)
- **Backend**: The underlying technology that renders the plot
- **DPI**: Dots per inch, determines plot resolution
- **Colormap**: A mapping from data values to colors
- **Alpha**: Transparency level (0 = transparent, 1 = opaque)

## Looking Ahead

In Lesson 13, we'll learn about:
- **Descriptive Statistics**: Mean, median, mode, standard deviation
- **Probability Distributions**: Normal, binomial, Poisson distributions
- **Correlation Analysis**: Understanding relationships between variables
- **Hypothesis Testing**: Making statistical inferences from data
