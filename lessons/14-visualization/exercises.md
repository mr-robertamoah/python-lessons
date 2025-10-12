# Lesson 12 Exercises: Data Visualization

## Guided Exercises (Do with Instructor)

### Exercise 1: Basic Plot Creation
**Goal**: Create fundamental plot types with matplotlib

**Tasks**:
1. Create a line plot showing temperature data over a week:
   ```python
   days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
   temps = [72, 75, 68, 80, 77, 73, 76]
   ```
2. Add proper title, axis labels, and grid
3. Customize the line color and add markers
4. Save the plot as 'temperature_trend.png'

**Expected Output**: Professional-looking line chart with clear labeling

---

### Exercise 2: Comparative Bar Chart
**Goal**: Compare data across categories using bar charts

**Data**: Quarterly sales for three products
```python
quarters = ['Q1', 'Q2', 'Q3', 'Q4']
product_a = [120, 135, 148, 162]
product_b = [100, 110, 125, 140]
product_c = [80, 95, 105, 120]
```

**Tasks**:
1. Create a grouped bar chart comparing all three products
2. Add value labels on top of each bar
3. Include a legend and proper formatting
4. Use different colors for each product

---

### Exercise 3: Distribution Analysis
**Goal**: Visualize data distributions using histograms and box plots

**Generate sample data**:
```python
import numpy as np
np.random.seed(42)
scores = np.random.normal(85, 12, 200)  # Test scores
```

**Create**:
1. Histogram with 20 bins
2. Add vertical line showing the mean
3. Create a box plot of the same data
4. Compare both visualizations side by side

---

## Independent Exercises (Try on Your Own)

### Exercise 4: Sales Dashboard
**Goal**: Create a comprehensive sales analysis dashboard

**Scenario**: You have sales data for an e-commerce company
```python
# Generate sample data
np.random.seed(42)
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
online_sales = [45000, 52000, 48000, 61000, 58000, 67000]
store_sales = [38000, 41000, 39000, 45000, 47000, 52000]
categories = ['Electronics', 'Clothing', 'Books', 'Home', 'Sports']
category_sales = [125000, 89000, 45000, 78000, 63000]
```

**Create a 2x2 subplot dashboard with**:
1. Line plot: Monthly sales trend (online vs store)
2. Bar chart: Sales by category
3. Pie chart: Market share by category
4. Stacked bar chart: Online vs store sales by month

**Requirements**:
- Professional styling and colors
- Clear titles and labels
- Legend where appropriate
- Save as 'sales_dashboard.png'

---

### Exercise 5: Customer Analysis Visualization
**Goal**: Analyze customer data using seaborn

**Generate customer dataset**:
```python
np.random.seed(42)
n_customers = 300

customers = pd.DataFrame({
    'age': np.random.normal(35, 12, n_customers),
    'income': np.random.normal(50000, 15000, n_customers),
    'spending': np.random.normal(2000, 600, n_customers),
    'satisfaction': np.random.uniform(1, 10, n_customers),
    'segment': np.random.choice(['Premium', 'Standard', 'Budget'], n_customers)
})
```

**Create visualizations**:
1. Scatter plot: Income vs Spending (colored by segment)
2. Box plot: Age distribution by segment
3. Correlation heatmap of numeric variables
4. Distribution plot: Satisfaction scores by segment

**Use seaborn for all plots and apply consistent styling**

---

### Exercise 6: Time Series Visualization
**Goal**: Create advanced time series plots

**Generate time series data**:
```python
dates = pd.date_range('2024-01-01', periods=365, freq='D')
# Simulate website traffic with trend and seasonality
trend = np.linspace(1000, 1500, 365)
seasonal = 200 * np.sin(2 * np.pi * np.arange(365) / 7)  # Weekly pattern
noise = np.random.normal(0, 50, 365)
traffic = trend + seasonal + noise
```

**Create**:
1. Daily traffic line plot
2. Weekly average (resample data)
3. Monthly totals with bar chart
4. Add moving averages (7-day and 30-day)

**Advanced**: Add annotations for significant events

---

### Exercise 7: Multi-Variable Analysis
**Goal**: Explore relationships between multiple variables

**Dataset**: Employee performance data
```python
np.random.seed(42)
employees = pd.DataFrame({
    'experience': np.random.uniform(0, 20, 200),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 200),
    'salary': np.random.normal(60000, 20000, 200),
    'performance': np.random.normal(7.5, 1.5, 200),
    'department': np.random.choice(['Engineering', 'Sales', 'Marketing', 'HR'], 200)
})
```

**Create comprehensive analysis**:
1. Scatter plot matrix (pairplot) for numeric variables
2. Box plots: Salary by education level
3. Violin plots: Performance by department
4. Regression plot: Experience vs Salary
5. Faceted plots: Performance by department and education

---

## Challenge Exercises (If You Finish Early)

### Challenge 1: Interactive Dashboard
**Goal**: Create an interactive-style dashboard with multiple linked views

**Requirements**:
- Use subplot_mosaic for complex layouts
- Create at least 6 different plot types
- Implement consistent color scheme
- Add statistical annotations (mean, median lines)
- Include data source and timestamp

### Challenge 2: Advanced Statistical Plots
**Goal**: Create publication-ready statistical visualizations

**Tasks**:
1. Create Anscombe's Quartet visualization
2. Build a correlation matrix with significance indicators
3. Create ridge plots for distribution comparison
4. Design a parallel coordinates plot
5. Build a radar/spider chart for multi-dimensional data

### Challenge 3: Custom Styling and Themes
**Goal**: Develop custom visualization themes

**Create**:
1. Custom color palette for your "brand"
2. Reusable plotting functions with consistent styling
3. Template for different chart types
4. Export functions for different formats (PNG, PDF, SVG)

---

## Self-Assessment Checklist

After completing these exercises, you should be able to:

- [ ] Create basic plots (line, bar, scatter, histogram, pie)
- [ ] Customize plot appearance (colors, labels, titles)
- [ ] Use seaborn for statistical visualizations
- [ ] Create multi-panel figures with subplots
- [ ] Choose appropriate chart types for different data
- [ ] Apply professional styling and formatting
- [ ] Save plots in various formats
- [ ] Interpret and communicate insights through visualizations

## Best Practices Learned

### Chart Selection
- **Line plots**: Trends over time
- **Bar charts**: Comparing categories
- **Histograms**: Data distributions
- **Scatter plots**: Relationships between variables
- **Box plots**: Distribution summaries and outliers
- **Heatmaps**: Correlation matrices

### Design Principles
- Clear, descriptive titles and labels
- Appropriate color choices (colorblind-friendly)
- Consistent styling across related plots
- Proper scaling and axis limits
- Minimal ink-to-data ratio

### Technical Tips
- Always save plots with high DPI (300+)
- Use `plt.tight_layout()` to prevent overlapping
- Choose appropriate figure sizes for your medium
- Include data sources and timestamps
- Test plots in grayscale for accessibility

## Common Mistakes to Avoid

1. **Overcomplicating plots** - Keep it simple and focused
2. **Poor color choices** - Avoid rainbow colors, use meaningful palettes
3. **Missing labels** - Always label axes and include units
4. **Wrong chart type** - Match visualization to data type and question
5. **Cluttered legends** - Place legends appropriately, consider direct labeling

## Git Reminder

Save your work:
1. Create `lesson-12-visualization` folder in your repository
2. Save all plot scripts and generated images
3. Include a README with plot descriptions
4. Commit and push:
   ```bash
   git add .
   git commit -m "Complete Lesson 12: Data Visualization"
   git push
   ```

## Next Lesson Preview

In Lesson 13, we'll learn about:
- **Descriptive Statistics**: Measures of central tendency and spread
- **Probability Distributions**: Normal, binomial, and other distributions
- **Correlation Analysis**: Understanding variable relationships
- **Statistical Testing**: Hypothesis testing and p-values
- **Confidence Intervals**: Quantifying uncertainty in estimates
