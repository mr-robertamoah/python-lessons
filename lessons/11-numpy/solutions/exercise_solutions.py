# Lesson 09 Solutions: NumPy Fundamentals

import numpy as np
import time

print("=== GUIDED EXERCISE SOLUTIONS ===\n")

# Exercise 1: Array Creation and Properties
print("Exercise 1: Array Creation and Properties")
print("-" * 40)

# Create arrays
arr1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
arr2 = np.zeros((3, 4))
arr3 = np.linspace(0, 1, 5)

arrays = [("1D array (1-10)", arr1), ("3x4 zeros", arr2), ("5 points 0-1", arr3)]

for name, arr in arrays:
    print(f"\n{name}:")
    print(f"  Array: {arr}")
    print(f"  Shape: {arr.shape}")
    print(f"  Size: {arr.size}")
    print(f"  Dimensions: {arr.ndim}")
    print(f"  Data type: {arr.dtype}")

print()

# Exercise 2: Basic Array Operations
print("Exercise 2: Basic Array Operations")
print("-" * 40)

arr_a = np.array([1, 2, 3, 4, 5])
arr_b = np.array([6, 7, 8, 9, 10])

print(f"Array A: {arr_a}")
print(f"Array B: {arr_b}")
print()

# Basic operations
print("Basic Operations:")
print(f"Addition: {arr_a + arr_b}")
print(f"Subtraction: {arr_a - arr_b}")
print(f"Multiplication: {arr_a * arr_b}")
print(f"Division: {arr_a / arr_b}")
print()

# Advanced operations
print("Advanced Operations:")
print(f"Square of A: {arr_a ** 2}")
print(f"Square root of B: {np.sqrt(arr_b)}")

# Manual verification
print("\nManual verification (first elements):")
print(f"1 + 6 = {1 + 6} (NumPy: {arr_a[0] + arr_b[0]})")
print(f"1 * 6 = {1 * 6} (NumPy: {arr_a[0] * arr_b[0]})")
print()

# Exercise 3: Statistical Analysis
print("Exercise 3: Statistical Analysis")
print("-" * 40)

temperatures = np.array([72, 75, 68, 80, 77, 73, 76])
print(f"Daily temperatures: {temperatures}")
print()

# Statistical calculations
mean_temp = np.mean(temperatures)
median_temp = np.median(temperatures)
std_temp = np.std(temperatures)
min_temp = np.min(temperatures)
max_temp = np.max(temperatures)
range_temp = max_temp - min_temp

print("Statistical Analysis:")
print(f"Mean temperature: {mean_temp:.1f}°F")
print(f"Median temperature: {median_temp:.1f}°F")
print(f"Standard deviation: {std_temp:.2f}°F")
print(f"Minimum temperature: {min_temp}°F")
print(f"Maximum temperature: {max_temp}°F")
print(f"Temperature range: {range_temp}°F")
print()

print("=== INDEPENDENT EXERCISE SOLUTIONS ===\n")

# Exercise 4: Sales Data Analysis
print("Exercise 4: Sales Data Analysis")
print("-" * 40)

sales_data = np.array([
    [1200, 1350, 1100, 1400, 1250, 1600],  # Product A
    [800, 900, 750, 950, 850, 1000],       # Product B
    [600, 650, 580, 700, 620, 750],        # Product C
    [400, 450, 380, 500, 420, 550]         # Product D
])

products = ['Product A', 'Product B', 'Product C', 'Product D']
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']

print("Sales Data Analysis:")
print("Sales data shape:", sales_data.shape)
print()

# 1. Total sales for each product
product_totals = np.sum(sales_data, axis=1)
print("1. Total sales by product:")
for i, (product, total) in enumerate(zip(products, product_totals)):
    print(f"   {product}: ${total:,}")
print()

# 2. Average monthly sales for each product
product_averages = np.mean(sales_data, axis=1)
print("2. Average monthly sales by product:")
for product, avg in zip(products, product_averages):
    print(f"   {product}: ${avg:.0f}")
print()

# 3. Total sales for each month
monthly_totals = np.sum(sales_data, axis=0)
print("3. Total sales by month:")
for month, total in zip(months, monthly_totals):
    print(f"   {month}: ${total:,}")
print()

# 4. Best and worst performing months
best_month_idx = np.argmax(monthly_totals)
worst_month_idx = np.argmin(monthly_totals)
print("4. Performance by month:")
print(f"   Best month: {months[best_month_idx]} (${monthly_totals[best_month_idx]:,})")
print(f"   Worst month: {months[worst_month_idx]} (${monthly_totals[worst_month_idx]:,})")
print()

# 5. Most consistent sales (lowest standard deviation)
product_std = np.std(sales_data, axis=1)
most_consistent_idx = np.argmin(product_std)
print("5. Sales consistency (standard deviation):")
for product, std in zip(products, product_std):
    print(f"   {product}: ${std:.0f}")
print(f"   Most consistent: {products[most_consistent_idx]}")
print()

# Exercise 5: Grade Processing System
print("Exercise 5: Grade Processing System")
print("-" * 40)

# Set seed for reproducible results
np.random.seed(42)

# Generate random test scores (20 students, 5 tests, scores 60-100)
num_students = 20
num_tests = 5
grades = np.random.randint(60, 101, (num_students, num_tests))

print(f"Generated grades for {num_students} students, {num_tests} tests each")
print("Sample grades (first 5 students):")
for i in range(5):
    print(f"  Student {i+1}: {grades[i]}")
print()

# Calculate each student's average
student_averages = np.mean(grades, axis=1)

# Determine letter grades
def get_letter_grade(avg):
    if avg >= 90: return 'A'
    elif avg >= 80: return 'B'
    elif avg >= 70: return 'C'
    elif avg >= 60: return 'D'
    else: return 'F'

letter_grades = [get_letter_grade(avg) for avg in student_averages]

# Class statistics
class_mean = np.mean(student_averages)
class_median = np.median(student_averages)
class_std = np.std(student_averages)

print("Class Statistics:")
print(f"Class average: {class_mean:.1f}")
print(f"Class median: {class_median:.1f}")
print(f"Standard deviation: {class_std:.2f}")
print()

# Top 3 students
top_3_indices = np.argsort(student_averages)[-3:][::-1]  # Get top 3, reverse order
print("Top 3 Students:")
for i, idx in enumerate(top_3_indices, 1):
    print(f"  {i}. Student {idx+1}: {student_averages[idx]:.1f} ({letter_grades[idx]})")
print()

# Grade distribution
unique_grades, counts = np.unique(letter_grades, return_counts=True)
print("Grade Distribution:")
for grade, count in zip(unique_grades, counts):
    print(f"  {grade}: {count} students")
print()

# Exercise 6: Financial Portfolio Analysis
print("Exercise 6: Financial Portfolio Analysis")
print("-" * 40)

# Generate sample data
np.random.seed(42)
returns = np.random.normal(0.001, 0.02, (3, 30))  # 3 stocks, 30 days
stock_names = ['Stock A', 'Stock B', 'Stock C']

print("Portfolio Analysis (3 stocks, 30 days)")
print()

# 1. Average daily return for each stock
avg_returns = np.mean(returns, axis=1)
print("1. Average daily returns:")
for stock, avg_return in zip(stock_names, avg_returns):
    print(f"   {stock}: {avg_return:.4f} ({avg_return*100:.2f}%)")
print()

# 2. Volatility (standard deviation) for each stock
volatilities = np.std(returns, axis=1)
print("2. Volatility (standard deviation):")
for stock, vol in zip(stock_names, volatilities):
    print(f"   {stock}: {vol:.4f} ({vol*100:.2f}%)")
print()

# 3. Cumulative returns over 30 days
cumulative_returns = np.cumprod(1 + returns, axis=1)[:, -1] - 1
print("3. Cumulative returns (30 days):")
for stock, cum_return in zip(stock_names, cumulative_returns):
    print(f"   {stock}: {cum_return:.4f} ({cum_return*100:.2f}%)")
print()

# 4. Best and worst performing days
best_days = np.argmax(returns, axis=1)
worst_days = np.argmin(returns, axis=1)
print("4. Best and worst performing days:")
for i, stock in enumerate(stock_names):
    best_return = returns[i, best_days[i]]
    worst_return = returns[i, worst_days[i]]
    print(f"   {stock}:")
    print(f"     Best day {best_days[i]+1}: {best_return:.4f} ({best_return*100:.2f}%)")
    print(f"     Worst day {worst_days[i]+1}: {worst_return:.4f} ({worst_return*100:.2f}%)")
print()

# 5. Correlation between stocks
correlation_matrix = np.corrcoef(returns)
print("5. Correlation matrix:")
print("     ", "  ".join(f"{name:>8}" for name in stock_names))
for i, stock in enumerate(stock_names):
    correlations = "  ".join(f"{correlation_matrix[i, j]:8.3f}" for j in range(3))
    print(f"{stock:>8} {correlations}")
print()

# Advanced: Sharpe ratio (assuming risk-free rate = 0)
sharpe_ratios = avg_returns / volatilities
print("Sharpe Ratios (return/volatility):")
for stock, sharpe in zip(stock_names, sharpe_ratios):
    print(f"   {stock}: {sharpe:.3f}")
print()

# Exercise 7: Array Indexing and Slicing
print("Exercise 7: Array Indexing and Slicing")
print("-" * 40)

# Create 5x5 array with values 1-25
arr_5x5 = np.arange(1, 26).reshape(5, 5)
print("Original 5x5 array:")
print(arr_5x5)
print()

# Extract middle 3x3 subarray
middle_3x3 = arr_5x5[1:4, 1:4]
print("Middle 3x3 subarray:")
print(middle_3x3)
print()

# Get first and last columns
first_last_cols = arr_5x5[:, [0, -1]]
print("First and last columns:")
print(first_last_cols)
print()

# Extract diagonal elements
diagonal = np.diag(arr_5x5)
print("Diagonal elements:", diagonal)
print()

# Boolean indexing: elements > 15
greater_than_15 = arr_5x5[arr_5x5 > 15]
print("Elements > 15:", greater_than_15)
print()

# Replace even numbers with -1
arr_modified = arr_5x5.copy()
arr_modified[arr_modified % 2 == 0] = -1
print("Array with even numbers replaced by -1:")
print(arr_modified)
print()

# Exercise 8: Broadcasting and Reshaping
print("Exercise 8: Broadcasting and Reshaping")
print("-" * 40)

# Create arrays for broadcasting
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
arr_1d = np.array([10, 20, 30])

print("2D array:")
print(arr_2d)
print("1D array:", arr_1d)
print()

# Broadcasting addition
broadcast_result = arr_2d + arr_1d
print("Broadcasting addition (2D + 1D):")
print(broadcast_result)
print()

# Reshape 1D array into different 2D configurations
arr_12 = np.arange(12)
print("Original 1D array:", arr_12)
print("Reshaped to 3x4:")
print(arr_12.reshape(3, 4))
print("Reshaped to 4x3:")
print(arr_12.reshape(4, 3))
print("Reshaped to 2x6:")
print(arr_12.reshape(2, 6))
print()

# Normalize columns (subtract mean, divide by std)
data_2d = np.random.randn(4, 3)
print("Original data:")
print(data_2d)
print()

# Column-wise normalization
col_means = np.mean(data_2d, axis=0)
col_stds = np.std(data_2d, axis=0)
normalized = (data_2d - col_means) / col_stds
print("Normalized data (mean=0, std=1 for each column):")
print(normalized)
print("Column means after normalization:", np.mean(normalized, axis=0))
print("Column stds after normalization:", np.std(normalized, axis=0))
print()

# Multiplication table using broadcasting
x = np.arange(1, 6).reshape(5, 1)  # Column vector
y = np.arange(1, 6)                # Row vector
multiplication_table = x * y
print("Multiplication table (1-5):")
print(multiplication_table)
print()

print("=== CHALLENGE EXERCISE SOLUTIONS ===\n")

# Challenge 1: Weather Data Analysis
print("Challenge 1: Weather Data Analysis")
print("-" * 40)

# Generate 365 days of weather data
np.random.seed(42)
days = np.arange(365)

# Create seasonal temperature pattern
seasonal_temp = 70 + 20 * np.sin(2 * np.pi * days / 365 - np.pi/2)  # Peak in summer
noise = np.random.normal(0, 5, 365)
temperatures = seasonal_temp + noise

# Generate humidity and pressure with some correlation to temperature
humidity = 60 + 30 * np.sin(2 * np.pi * days / 365 + np.pi) + np.random.normal(0, 10, 365)
pressure = 1013 + 20 * np.sin(2 * np.pi * days / 365) + np.random.normal(0, 5, 365)

# Ensure realistic ranges
humidity = np.clip(humidity, 20, 100)
pressure = np.clip(pressure, 980, 1040)

print("Generated weather data for 365 days")
print(f"Temperature range: {np.min(temperatures):.1f}°F to {np.max(temperatures):.1f}°F")
print(f"Humidity range: {np.min(humidity):.1f}% to {np.max(humidity):.1f}%")
print(f"Pressure range: {np.min(pressure):.1f} to {np.max(pressure):.1f} hPa")
print()

# Monthly averages
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

print("Monthly averages:")
start_day = 0
for month, days_in_month in zip(months, days_per_month):
    end_day = start_day + days_in_month
    month_temp = np.mean(temperatures[start_day:end_day])
    month_humidity = np.mean(humidity[start_day:end_day])
    month_pressure = np.mean(pressure[start_day:end_day])
    
    print(f"{month}: Temp={month_temp:.1f}°F, Humidity={month_humidity:.1f}%, Pressure={month_pressure:.1f}hPa")
    start_day = end_day
print()

# Find extreme weather days
hottest_day = np.argmax(temperatures)
coldest_day = np.argmin(temperatures)
print("Extreme weather days:")
print(f"Hottest day: Day {hottest_day+1} ({temperatures[hottest_day]:.1f}°F)")
print(f"Coldest day: Day {coldest_day+1} ({temperatures[coldest_day]:.1f}°F)")
print()

# Correlations
weather_data = np.column_stack([temperatures, humidity, pressure])
correlation_matrix = np.corrcoef(weather_data.T)
weather_vars = ['Temperature', 'Humidity', 'Pressure']

print("Weather variable correlations:")
print("              ", "  ".join(f"{var:>11}" for var in weather_vars))
for i, var in enumerate(weather_vars):
    correlations = "  ".join(f"{correlation_matrix[i, j]:11.3f}" for j in range(3))
    print(f"{var:>11} {correlations}")
print()

# Challenge 2: Monte Carlo Simulation
print("Challenge 2: Monte Carlo Simulation")
print("-" * 40)

# Simulate coin flips
np.random.seed(42)
n_flips = 1000
n_simulations = 100

print(f"Monte Carlo simulation: {n_flips} coin flips, {n_simulations} simulations")
print()

# Single simulation with running average
single_simulation = np.random.randint(0, 2, n_flips)  # 0=tails, 1=heads
running_average = np.cumsum(single_simulation) / np.arange(1, n_flips + 1)

print("Single simulation results:")
print(f"Final proportion of heads: {running_average[-1]:.4f}")
print(f"Running averages at different points:")
checkpoints = [10, 50, 100, 500, 1000]
for checkpoint in checkpoints:
    if checkpoint <= n_flips:
        print(f"  After {checkpoint} flips: {running_average[checkpoint-1]:.4f}")
print()

# Multiple simulations
all_results = []
for i in range(n_simulations):
    simulation = np.random.randint(0, 2, n_flips)
    final_proportion = np.mean(simulation)
    all_results.append(final_proportion)

all_results = np.array(all_results)

print("Multiple simulations analysis:")
print(f"Mean proportion across {n_simulations} simulations: {np.mean(all_results):.4f}")
print(f"Standard deviation: {np.std(all_results):.4f}")
print(f"Min proportion: {np.min(all_results):.4f}")
print(f"Max proportion: {np.max(all_results):.4f}")
print(f"95% of simulations between: {np.percentile(all_results, 2.5):.4f} and {np.percentile(all_results, 97.5):.4f}")
print()

# Estimate π using Monte Carlo
print("Bonus: Estimating π using Monte Carlo method")
n_points = 100000
np.random.seed(42)

# Generate random points in unit square
x = np.random.uniform(-1, 1, n_points)
y = np.random.uniform(-1, 1, n_points)

# Check which points are inside unit circle
inside_circle = (x**2 + y**2) <= 1
pi_estimate = 4 * np.sum(inside_circle) / n_points

print(f"Using {n_points} random points:")
print(f"Estimated π: {pi_estimate:.6f}")
print(f"Actual π: {np.pi:.6f}")
print(f"Error: {abs(pi_estimate - np.pi):.6f}")
print()

# Challenge 3: Matrix Operations and Linear Algebra
print("Challenge 3: Matrix Operations and Linear Algebra")
print("-" * 40)

np.random.seed(42)

# Create two random 5x5 matrices
matrix_a = np.random.randn(5, 5)
matrix_b = np.random.randn(5, 5)

print("Created two random 5x5 matrices")
print("Matrix A (first 3x3):")
print(matrix_a[:3, :3])
print("Matrix B (first 3x3):")
print(matrix_b[:3, :3])
print()

# Matrix multiplication
matrix_product = np.dot(matrix_a, matrix_b)
# Alternative: matrix_product = matrix_a @ matrix_b
print("Matrix multiplication A @ B (first 3x3):")
print(matrix_product[:3, :3])
print()

# Determinant and inverse (use smaller matrix for stability)
small_matrix = matrix_a[:3, :3]
det = np.linalg.det(small_matrix)
print(f"Determinant of 3x3 submatrix: {det:.6f}")

if abs(det) > 1e-10:  # Check if matrix is invertible
    inverse = np.linalg.inv(small_matrix)
    print("Inverse exists. Verification (A @ A^-1 should be identity):")
    identity_check = np.dot(small_matrix, inverse)
    print(identity_check)
    print(f"Max deviation from identity: {np.max(np.abs(identity_check - np.eye(3))):.2e}")
else:
    print("Matrix is singular (not invertible)")
print()

# Solve system of linear equations Ax = b
A = np.array([[2, 1, -1],
              [-3, -1, 2],
              [-2, 1, 2]])
b = np.array([8, -11, -3])

print("Solving system of linear equations Ax = b:")
print("A =")
print(A)
print("b =", b)

x = np.linalg.solve(A, b)
print("Solution x =", x)

# Verify solution
verification = np.dot(A, x)
print("Verification Ax =", verification)
print("Should equal b =", b)
print(f"Max error: {np.max(np.abs(verification - b)):.2e}")
print()

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues:", eigenvalues)
print("First eigenvector:", eigenvectors[:, 0])

# Verify first eigenvalue/eigenvector pair
verification = np.dot(A, eigenvectors[:, 0])
expected = eigenvalues[0] * eigenvectors[:, 0]
print("Verification of first eigenvalue/eigenvector:")
print(f"A @ v1 = {verification}")
print(f"λ1 @ v1 = {expected}")
print(f"Max error: {np.max(np.abs(verification - expected)):.2e}")
print()

print("=== PERFORMANCE COMPARISON ===\n")

# Performance comparison: NumPy vs Pure Python
print("Performance Comparison: NumPy vs Pure Python")
print("-" * 50)

# Test data
size = 100000
np.random.seed(42)
data1 = np.random.randn(size)
data2 = np.random.randn(size)
list1 = data1.tolist()
list2 = data2.tolist()

# Test 1: Element-wise addition
print("Test 1: Element-wise addition")

# NumPy version
start_time = time.time()
numpy_result = data1 + data2
numpy_time = time.time() - start_time

# Pure Python version
start_time = time.time()
python_result = [a + b for a, b in zip(list1, list2)]
python_time = time.time() - start_time

print(f"NumPy time: {numpy_time:.6f} seconds")
print(f"Python time: {python_time:.6f} seconds")
print(f"NumPy is {python_time/numpy_time:.1f}x faster")
print()

# Test 2: Mathematical operations
print("Test 2: Square root and sum")

# NumPy version
start_time = time.time()
numpy_sqrt_sum = np.sum(np.sqrt(np.abs(data1)))
numpy_time = time.time() - start_time

# Pure Python version
import math
start_time = time.time()
python_sqrt_sum = sum(math.sqrt(abs(x)) for x in list1)
python_time = time.time() - start_time

print(f"NumPy result: {numpy_sqrt_sum:.6f}")
print(f"Python result: {python_sqrt_sum:.6f}")
print(f"NumPy time: {numpy_time:.6f} seconds")
print(f"Python time: {python_time:.6f} seconds")
print(f"NumPy is {python_time/numpy_time:.1f}x faster")
print()

print("NumPy provides significant performance improvements for numerical operations!")
print("This is due to:")
print("- Vectorized operations implemented in C")
print("- Efficient memory layout")
print("- Optimized algorithms")
print("- Reduced Python overhead")

print("\n" + "=" * 50)
print("NumPy exercise solutions complete!")
