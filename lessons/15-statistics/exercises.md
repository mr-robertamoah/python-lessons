# Lesson 13 Exercises: Descriptive Statistics and Probability

## Guided Exercises (Do with Instructor)

### Exercise 1: Basic Descriptive Statistics
**Goal**: Calculate and interpret basic statistical measures

**Dataset**: Student test scores: [78, 85, 92, 88, 76, 94, 89, 82, 91, 87, 79, 93, 86, 90, 84]

**Tasks**:
1. Calculate mean, median, and mode
2. Find the range, variance, and standard deviation
3. Calculate the 25th, 50th, and 75th percentiles
4. Interpret what these statistics tell us about the data

**Template**:
```python
import numpy as np
from scipy import stats

scores = [78, 85, 92, 88, 76, 94, 89, 82, 91, 87, 79, 93, 86, 90, 84]

# Your calculations here
mean_score = np.mean(scores)
# Continue with other statistics...
```

---

### Exercise 2: Probability Distributions
**Goal**: Work with normal distribution and calculate probabilities

**Scenario**: IQ scores are normally distributed with mean=100 and standard deviation=15

**Tasks**:
1. What percentage of people have IQ scores between 85 and 115?
2. What IQ score represents the 90th percentile?
3. If someone has an IQ of 130, what percentile are they in?
4. Create a visualization of the normal distribution

---

### Exercise 3: Hypothesis Testing
**Goal**: Perform a basic t-test

**Scenario**: A company claims their new training program increases productivity. Test scores before and after training:
- Before: [72, 68, 75, 71, 69, 73, 70, 74, 67, 76]
- After: [78, 74, 82, 79, 75, 81, 77, 80, 73, 83]

**Tasks**:
1. State null and alternative hypotheses
2. Perform a paired t-test
3. Interpret the p-value
4. Make a conclusion about the training program's effectiveness

---

## Independent Exercises (Try on Your Own)

### Exercise 4: Sales Data Analysis
**Goal**: Comprehensive statistical analysis of business data

**Generate sample sales data**:
```python
np.random.seed(42)
sales_data = {
    'region': np.random.choice(['North', 'South', 'East', 'West'], 200),
    'sales_amount': np.random.normal(5000, 1500, 200),
    'customer_satisfaction': np.random.normal(7.5, 1.2, 200)
}
```

**Analyze**:
1. Descriptive statistics for each numeric variable
2. Compare sales performance across regions
3. Test if there's a significant difference in satisfaction between regions
4. Calculate correlation between sales amount and satisfaction
5. Create appropriate visualizations

---

### Exercise 5: Quality Control Analysis
**Goal**: Apply statistical process control concepts

**Scenario**: Manufacturing process produces widgets with target weight of 100g

**Generate data**:
```python
np.random.seed(123)
weights = np.random.normal(100, 2.5, 50)  # 50 widget weights
```

**Tasks**:
1. Test if the process is meeting the target (μ = 100g)
2. Calculate control limits (mean ± 3 standard deviations)
3. Identify any outliers
4. Create a control chart
5. Assess process capability

---

### Exercise 6: A/B Testing Analysis
**Goal**: Design and analyze an A/B test

**Scenario**: Website conversion rate test
- Control group: 1000 visitors, 45 conversions
- Treatment group: 1000 visitors, 52 conversions

**Tasks**:
1. Calculate conversion rates for both groups
2. Perform a two-proportion z-test
3. Calculate confidence interval for the difference
4. Determine statistical and practical significance
5. Make a recommendation based on results

---

### Exercise 7: Survey Data Analysis
**Goal**: Analyze survey responses with multiple variables

**Generate survey data**:
```python
np.random.seed(456)
n = 300
survey_data = pd.DataFrame({
    'age': np.random.randint(18, 65, n),
    'income': np.random.lognormal(10.5, 0.5, n),
    'satisfaction': np.random.randint(1, 11, n),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n),
    'department': np.random.choice(['Sales', 'Engineering', 'Marketing', 'HR'], n)
})
```

**Analyze**:
1. Descriptive statistics by education level
2. Correlation analysis between numeric variables
3. ANOVA test for satisfaction across departments
4. Chi-square test for independence between education and department
5. Create comprehensive statistical summary

---

## Challenge Exercises (If You Finish Early)

### Challenge 1: Monte Carlo Simulation
**Goal**: Use simulation to solve probability problems

**Problem**: Estimate the probability of getting at least 3 heads in 5 coin flips

**Tasks**:
1. Write a simulation with 10,000 trials
2. Compare with theoretical probability
3. Visualize the distribution of results
4. Calculate confidence interval for your estimate

### Challenge 2: Bootstrap Confidence Intervals
**Goal**: Use bootstrap method for non-parametric statistics

**Dataset**: [23, 45, 67, 34, 78, 56, 89, 12, 43, 65, 87, 32, 54, 76, 21]

**Tasks**:
1. Calculate bootstrap confidence interval for the mean
2. Compare with theoretical confidence interval
3. Bootstrap confidence interval for the median
4. Visualize bootstrap distributions

### Challenge 3: Power Analysis
**Goal**: Understand statistical power and sample size

**Scenario**: Planning an experiment to detect a 10% improvement in conversion rate

**Tasks**:
1. Calculate required sample size for 80% power
2. Create power curves for different effect sizes
3. Analyze trade-offs between sample size, effect size, and power
4. Make recommendations for experiment design

---

## Self-Assessment Checklist

After completing these exercises, you should be able to:

- [ ] Calculate and interpret descriptive statistics
- [ ] Work with probability distributions
- [ ] Perform hypothesis tests (t-tests, chi-square, ANOVA)
- [ ] Calculate and interpret confidence intervals
- [ ] Understand p-values and statistical significance
- [ ] Apply statistical concepts to business problems
- [ ] Create appropriate statistical visualizations
- [ ] Design and analyze A/B tests

## Common Statistical Mistakes to Avoid

1. **Confusing correlation with causation**
2. **Multiple testing without correction**
3. **Ignoring assumptions of statistical tests**
4. **Misinterpreting p-values**
5. **Cherry-picking significant results**
6. **Using inappropriate statistical tests**
7. **Ignoring practical significance**

## Git Reminder

Save your work:
1. Create `lesson-13-statistics` folder in your repository
2. Save exercise solutions as `.py` files
3. Include statistical interpretations in comments
4. Commit and push:
   ```bash
   git add .
   git commit -m "Complete Lesson 13: Statistics Exercises"
   git push
   ```

## Next Lesson Preview

In Lesson 14, we'll learn about:
- **Feature Engineering**: Creating and transforming variables
- **Categorical Encoding**: Converting categories to numbers
- **Feature Scaling**: Normalizing data for machine learning
- **Feature Selection**: Choosing the most important variables
