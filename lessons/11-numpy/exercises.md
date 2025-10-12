# Lesson 09 Exercises: NumPy Fundamentals

## Guided Exercises (Do with Instructor)

### Exercise 1: Array Creation and Properties
**Goal**: Practice creating arrays and understanding their properties

**Tasks**:
1. Create a 1D array with numbers 1 through 10
2. Create a 2D array (3x4) filled with zeros
3. Create an array with 5 evenly spaced numbers between 0 and 1
4. For each array, print its shape, size, dimensions, and data type

**Template**:
```python
import numpy as np

# Your arrays here
arr1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# Continue with other arrays...

# Print properties
print(f"Shape: {arr1.shape}")
# Continue with other properties...
```

---

### Exercise 2: Basic Array Operations
**Goal**: Practice mathematical operations on arrays

**Tasks**:
1. Create two arrays: [1, 2, 3, 4, 5] and [6, 7, 8, 9, 10]
2. Perform addition, subtraction, multiplication, and division
3. Calculate the square of the first array
4. Find the square root of the second array
5. Compare results with manual calculations

**Expected Output**: Verify that array operations work element-wise

---

### Exercise 3: Statistical Analysis
**Goal**: Use NumPy for basic statistical calculations

**Given data**: Daily temperatures for a week: [72, 75, 68, 80, 77, 73, 76]

**Calculate**:
1. Mean temperature
2. Median temperature
3. Standard deviation
4. Minimum and maximum temperatures
5. Range (max - min)

**Verify**: Check your results make sense for temperature data

---

## Independent Exercises (Try on Your Own)

### Exercise 4: Sales Data Analysis
**Goal**: Analyze monthly sales data using NumPy

**Scenario**: You have sales data for 4 products over 6 months:
```python
sales_data = np.array([
    [1200, 1350, 1100, 1400, 1250, 1600],  # Product A
    [800, 900, 750, 950, 850, 1000],       # Product B
    [600, 650, 580, 700, 620, 750],        # Product C
    [400, 450, 380, 500, 420, 550]         # Product D
])
```

**Calculate**:
1. Total sales for each product (sum across months)
2. Average monthly sales for each product
3. Total sales for each month (sum across products)
4. Best and worst performing months
5. Which product has the most consistent sales (lowest standard deviation)

---

### Exercise 5: Grade Processing System
**Goal**: Build a comprehensive grade analysis system

**Create a system that**:
1. Generates random test scores for 20 students and 5 tests (scores 60-100)
2. Calculates each student's average
3. Determines letter grades (A: 90+, B: 80-89, C: 70-79, D: 60-69, F: <60)
4. Finds class statistics (mean, median, std dev)
5. Identifies top 3 students

**Requirements**:
- Use `np.random.randint()` for score generation
- Use boolean indexing for grade assignment
- Display results in a clear format

---

### Exercise 6: Financial Portfolio Analysis
**Goal**: Analyze investment portfolio performance

**Scenario**: You have daily returns for 3 stocks over 30 days
```python
# Generate sample data
np.random.seed(42)  # For reproducible results
returns = np.random.normal(0.001, 0.02, (3, 30))  # 3 stocks, 30 days
```

**Calculate**:
1. Average daily return for each stock
2. Volatility (standard deviation) for each stock
3. Cumulative returns over the 30 days
4. Best and worst performing days for each stock
5. Correlation between stocks (use np.corrcoef)

**Advanced**: Calculate Sharpe ratio (return/volatility) for each stock

---

### Exercise 7: Image Processing Basics
**Goal**: Use NumPy for basic image operations

**Tasks**:
1. Create a 10x10 "image" (2D array) with random values 0-255
2. Convert to grayscale by averaging RGB channels (simulate)
3. Apply a simple filter (e.g., blur by averaging neighboring pixels)
4. Find the brightest and darkest pixels
5. Create a binary image (pixels > threshold = 1, else = 0)

**Note**: This simulates real image processing concepts

---

## Challenge Exercises (If You Finish Early)

### Challenge 1: Weather Data Analysis
**Goal**: Comprehensive analysis of weather patterns

**Create simulation of weather data**:
- Temperature, humidity, and pressure for 365 days
- Include seasonal patterns and random variation
- Analyze monthly averages, seasonal trends
- Find correlations between variables
- Identify extreme weather days

### Challenge 2: Monte Carlo Simulation
**Goal**: Use NumPy for statistical simulation

**Simulate coin flips**:
1. Simulate 1000 coin flips
2. Calculate running average of heads
3. Repeat simulation 100 times
4. Analyze distribution of results
5. Verify it approaches 0.5 (law of large numbers)

### Challenge 3: Matrix Operations
**Goal**: Advanced array manipulation

**Tasks**:
1. Create two random 5x5 matrices
2. Perform matrix multiplication (use np.dot)
3. Calculate determinant and inverse
4. Solve a system of linear equations
5. Find eigenvalues and eigenvectors

---

## Self-Assessment Checklist

After completing these exercises, you should be able to:

- [ ] Create NumPy arrays using various methods
- [ ] Understand array properties (shape, size, dtype)
- [ ] Perform mathematical operations on arrays
- [ ] Use statistical functions (mean, median, std, etc.)
- [ ] Apply boolean indexing and fancy indexing
- [ ] Understand broadcasting rules
- [ ] Analyze real-world data using NumPy
- [ ] Recognize when to use NumPy vs pure Python

## Common Mistakes to Avoid

### 1. Confusing Array Dimensions
```python
# Wrong assumption about shape
arr = np.array([[1, 2, 3]])
print(arr.shape)  # (1, 3) not (3,)

# Correct way to create 1D array
arr = np.array([1, 2, 3])
print(arr.shape)  # (3,)
```

### 2. Modifying Arrays Unintentionally
```python
# Views vs copies
original = np.array([1, 2, 3, 4, 5])
view = original[1:4]
view[0] = 999
print(original)  # [1 999 3 4 5] - original changed!

# Use copy() when you need independence
copy = original[1:4].copy()
```

### 3. Broadcasting Errors
```python
# This will fail
arr1 = np.array([[1, 2], [3, 4]])  # Shape: (2, 2)
arr2 = np.array([1, 2, 3])         # Shape: (3,)
# result = arr1 + arr2  # Error: shapes not compatible
```

## Performance Tips

1. **Use vectorized operations** instead of loops
2. **Choose appropriate data types** (int32 vs int64)
3. **Understand memory layout** (C vs Fortran order)
4. **Use views instead of copies** when possible
5. **Preallocate arrays** when size is known

## Git Reminder

Save your work:
1. Create `lesson-09-numpy` folder in your repository
2. Save exercise solutions as `.py` files
3. Include comments explaining your approach
4. Commit and push:
   ```bash
   git add .
   git commit -m "Complete Lesson 09: NumPy Fundamentals"
   git push
   ```

## Next Lesson Preview

In Lesson 10, we'll learn about:
- **Pandas DataFrames**: 2D labeled data structures
- **Data import/export**: Reading CSV, Excel files
- **Data selection**: Filtering and querying data
- **Data manipulation**: Grouping, aggregating, merging
- **Real-world data analysis**: Working with messy datasets
