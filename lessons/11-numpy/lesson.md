# Lesson 09: NumPy Fundamentals - Numerical Computing with Arrays

## Learning Objectives
By the end of this lesson, you will be able to:
- Understand why NumPy is essential for data analysis
- Create and manipulate NumPy arrays efficiently
- Perform mathematical operations on arrays
- Use broadcasting for operations on different-sized arrays
- Apply NumPy functions for statistical analysis
- Understand performance benefits over pure Python

## Why NumPy Matters for Data Analysis

### The Problem with Pure Python Lists
```python
# Pure Python: Slow for numerical operations
data = [1, 2, 3, 4, 5] * 1000000  # 5 million numbers
result = []
for i in range(len(data)):
    result.append(data[i] * 2)  # Takes several seconds!
```

### The NumPy Solution
```python
import numpy as np

# NumPy: Fast vectorized operations
data = np.array([1, 2, 3, 4, 5] * 1000000)
result = data * 2  # Takes milliseconds!
```

### Key Benefits of NumPy
- **Speed**: 10-100x faster than pure Python
- **Memory efficiency**: Uses less memory than Python lists
- **Vectorization**: Operations on entire arrays at once
- **Broadcasting**: Smart handling of different array sizes
- **Foundation**: Basis for pandas, scikit-learn, matplotlib

## Installing and Importing NumPy

### Installation
```bash
# In your virtual environment
pip install numpy

# Or with specific version
pip install numpy==1.24.3
```

### Importing Convention
```python
import numpy as np  # Standard convention

# Check version
print(np.__version__)
```

## Creating NumPy Arrays

### From Python Lists
```python
import numpy as np

# 1D array
arr1d = np.array([1, 2, 3, 4, 5])
print(arr1d)  # [1 2 3 4 5]

# 2D array (matrix)
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
print(arr2d)
# [[1 2 3]
#  [4 5 6]]

# 3D array
arr3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
```

### Array Creation Functions
```python
# Arrays of zeros
zeros = np.zeros(5)          # [0. 0. 0. 0. 0.]
zeros_2d = np.zeros((3, 4))  # 3x4 matrix of zeros

# Arrays of ones
ones = np.ones(3)            # [1. 1. 1.]
ones_2d = np.ones((2, 3))    # 2x3 matrix of ones

# Arrays with specific values
full = np.full(4, 7)         # [7 7 7 7]
full_2d = np.full((2, 3), 5) # 2x3 matrix of 5s

# Identity matrix
identity = np.eye(3)         # 3x3 identity matrix

# Range arrays
range_arr = np.arange(10)         # [0 1 2 3 4 5 6 7 8 9]
range_step = np.arange(0, 10, 2)  # [0 2 4 6 8]

# Evenly spaced arrays
linspace = np.linspace(0, 1, 5)   # [0.   0.25 0.5  0.75 1.  ]

# Random arrays
random_arr = np.random.random(5)     # Random floats 0-1
random_int = np.random.randint(1, 10, 5)  # Random integers 1-9
```

## Array Properties and Attributes

### Basic Properties
```python
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

print(f"Shape: {arr.shape}")        # (2, 4) - 2 rows, 4 columns
print(f"Size: {arr.size}")          # 8 - total elements
print(f"Dimensions: {arr.ndim}")    # 2 - number of dimensions
print(f"Data type: {arr.dtype}")    # int64 (or int32 on some systems)
print(f"Item size: {arr.itemsize}") # 8 bytes per element
```

### Data Types
```python
# Specify data type during creation
int_arr = np.array([1, 2, 3], dtype=np.int32)
float_arr = np.array([1, 2, 3], dtype=np.float64)
bool_arr = np.array([True, False, True], dtype=np.bool_)

# Convert data types
arr = np.array([1.7, 2.3, 3.9])
int_converted = arr.astype(np.int32)  # [1 2 3]

# Common data types
# np.int32, np.int64    - integers
# np.float32, np.float64 - floating point
# np.bool_              - boolean
# np.str_               - string
```

## Array Indexing and Slicing

### 1D Array Indexing
```python
arr = np.array([10, 20, 30, 40, 50])

# Basic indexing
print(arr[0])    # 10 (first element)
print(arr[-1])   # 50 (last element)
print(arr[1:4])  # [20 30 40] (slice)
print(arr[::2])  # [10 30 50] (every 2nd element)

# Boolean indexing
mask = arr > 25
print(arr[mask])  # [30 40 50] (elements > 25)
```

### 2D Array Indexing
```python
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Access elements
print(arr2d[0, 1])    # 2 (row 0, column 1)
print(arr2d[1])       # [4 5 6] (entire row 1)
print(arr2d[:, 2])    # [3 6 9] (entire column 2)

# Slicing
print(arr2d[0:2, 1:3])  # First 2 rows, columns 1-2
# [[2 3]
#  [5 6]]

# Advanced indexing
rows = [0, 2]
cols = [1, 2]
print(arr2d[rows, cols])  # [2 9] (elements at (0,1) and (2,2))
```

### Fancy Indexing
```python
arr = np.array([10, 20, 30, 40, 50])

# Index with array of indices
indices = [0, 2, 4]
print(arr[indices])  # [10 30 50]

# Boolean indexing with conditions
print(arr[arr > 25])           # [30 40 50]
print(arr[(arr > 15) & (arr < 45)])  # [20 30 40]
```

## Array Operations

### Arithmetic Operations
```python
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([5, 6, 7, 8])

# Element-wise operations
print(arr1 + arr2)   # [6 8 10 12]
print(arr1 - arr2)   # [-4 -4 -4 -4]
print(arr1 * arr2)   # [5 12 21 32]
print(arr1 / arr2)   # [0.2 0.33 0.43 0.5]
print(arr1 ** 2)     # [1 4 9 16]

# Operations with scalars
print(arr1 + 10)     # [11 12 13 14]
print(arr1 * 3)      # [3 6 9 12]
```

### Mathematical Functions
```python
arr = np.array([1, 4, 9, 16, 25])

# Square root, logarithm, exponential
print(np.sqrt(arr))    # [1. 2. 3. 4. 5.]
print(np.log(arr))     # Natural logarithm
print(np.exp(arr))     # e^x

# Trigonometric functions
angles = np.array([0, np.pi/2, np.pi])
print(np.sin(angles))  # [0. 1. 0.]
print(np.cos(angles))  # [1. 0. -1.]

# Rounding
decimals = np.array([1.234, 2.567, 3.891])
print(np.round(decimals, 2))  # [1.23 2.57 3.89]
print(np.floor(decimals))     # [1. 2. 3.]
print(np.ceil(decimals))      # [2. 3. 4.]
```

## Statistical Operations

### Basic Statistics
```python
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

print(f"Mean: {np.mean(data)}")           # 5.5
print(f"Median: {np.median(data)}")       # 5.5
print(f"Standard deviation: {np.std(data)}")  # 2.87
print(f"Variance: {np.var(data)}")        # 8.25
print(f"Min: {np.min(data)}")             # 1
print(f"Max: {np.max(data)}")             # 10
print(f"Sum: {np.sum(data)}")             # 55
```

### Statistics Along Axes
```python
data2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Statistics along different axes
print(f"Mean of all elements: {np.mean(data2d)}")      # 5.0
print(f"Mean along rows (axis=0): {np.mean(data2d, axis=0)}")  # [4. 5. 6.]
print(f"Mean along columns (axis=1): {np.mean(data2d, axis=1)}")  # [2. 5. 8.]

# Other statistics
print(f"Sum along rows: {np.sum(data2d, axis=0)}")     # [12 15 18]
print(f"Max along columns: {np.max(data2d, axis=1)}")  # [3 6 9]
```

## Broadcasting

### What is Broadcasting?
Broadcasting allows operations between arrays of different shapes:

```python
# Scalar with array
arr = np.array([1, 2, 3, 4])
result = arr + 10  # Broadcasts 10 to [10, 10, 10, 10]
print(result)      # [11 12 13 14]

# 1D array with 2D array
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
arr1d = np.array([10, 20, 30])
result = arr2d + arr1d  # Broadcasts arr1d to each row
print(result)
# [[11 22 33]
#  [14 25 36]]
```

### Broadcasting Rules
```python
# Compatible shapes for broadcasting:
# (3,)     + (3,)     = (3,)     ✓
# (3, 1)   + (3,)     = (3, 3)   ✓
# (3, 4)   + (4,)     = (3, 4)   ✓
# (3, 4)   + (3, 1)   = (3, 4)   ✓

# Examples
a = np.array([[1], [2], [3]])  # Shape: (3, 1)
b = np.array([10, 20, 30, 40]) # Shape: (4,)
result = a + b                 # Shape: (3, 4)
print(result)
# [[11 21 31 41]
#  [12 22 32 42]
#  [13 23 33 43]]
```

## Array Manipulation

### Reshaping Arrays
```python
arr = np.arange(12)  # [0 1 2 3 4 5 6 7 8 9 10 11]

# Reshape to 2D
reshaped = arr.reshape(3, 4)
print(reshaped)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

# Reshape to 3D
reshaped_3d = arr.reshape(2, 2, 3)

# Flatten back to 1D
flattened = reshaped.flatten()  # or .ravel()
```

### Joining and Splitting Arrays
```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# Concatenate
concatenated = np.concatenate([arr1, arr2])  # [1 2 3 4 5 6]

# Stack arrays
stacked = np.stack([arr1, arr2])  # 2D array
# [[1 2 3]
#  [4 5 6]]

# Split arrays
arr = np.array([1, 2, 3, 4, 5, 6])
split_result = np.split(arr, 3)  # [array([1, 2]), array([3, 4]), array([5, 6])]
```

## Practical Data Analysis Examples

### Example 1: Sales Data Analysis
```python
import numpy as np

# Monthly sales data for 4 products over 12 months
sales_data = np.array([
    [1200, 1350, 1100, 1400, 1250, 1600, 1450, 1300, 1550, 1400, 1650, 1500],  # Product A
    [800, 900, 750, 950, 850, 1000, 920, 880, 980, 900, 1050, 950],             # Product B
    [600, 650, 580, 700, 620, 750, 680, 640, 720, 680, 780, 720],              # Product C
    [400, 450, 380, 500, 420, 550, 480, 440, 520, 460, 580, 520]               # Product D
])

print("Sales Data Analysis:")
print(f"Shape: {sales_data.shape}")  # (4, 12) - 4 products, 12 months

# Total sales per product
product_totals = np.sum(sales_data, axis=1)
print(f"Total sales per product: {product_totals}")

# Average monthly sales per product
product_averages = np.mean(sales_data, axis=1)
print(f"Average monthly sales: {product_averages}")

# Monthly totals across all products
monthly_totals = np.sum(sales_data, axis=0)
print(f"Monthly totals: {monthly_totals}")

# Best and worst performing months
best_month = np.argmax(monthly_totals) + 1  # +1 for 1-based indexing
worst_month = np.argmin(monthly_totals) + 1
print(f"Best month: {best_month}, Worst month: {worst_month}")

# Growth analysis (month-over-month)
monthly_growth = np.diff(monthly_totals) / monthly_totals[:-1] * 100
print(f"Monthly growth rates (%): {np.round(monthly_growth, 2)}")
```

### Example 2: Student Grade Analysis
```python
# Student test scores (rows=students, columns=tests)
grades = np.array([
    [85, 92, 78, 96, 88],  # Student 1
    [76, 84, 91, 83, 79],  # Student 2
    [92, 95, 89, 94, 91],  # Student 3
    [68, 72, 75, 70, 74],  # Student 4
    [88, 86, 92, 89, 87]   # Student 5
])

print("Grade Analysis:")

# Student averages
student_averages = np.mean(grades, axis=1)
print(f"Student averages: {np.round(student_averages, 2)}")

# Test averages
test_averages = np.mean(grades, axis=0)
print(f"Test averages: {np.round(test_averages, 2)}")

# Standard deviations
student_std = np.std(grades, axis=1)
test_std = np.std(grades, axis=0)
print(f"Student consistency (lower std = more consistent): {np.round(student_std, 2)}")
print(f"Test difficulty variation: {np.round(test_std, 2)}")

# Grade distribution
passing_grades = grades >= 70
passing_rate = np.mean(passing_grades) * 100
print(f"Overall passing rate: {passing_rate:.1f}%")

# Find top performers
top_student_idx = np.argmax(student_averages)
print(f"Top student: Student {top_student_idx + 1} with average {student_averages[top_student_idx]:.2f}")
```

### Example 3: Financial Data Processing
```python
# Stock price data simulation
np.random.seed(42)  # For reproducible results
days = 252  # Trading days in a year
initial_price = 100

# Generate random daily returns (normally distributed)
daily_returns = np.random.normal(0.001, 0.02, days)  # 0.1% mean, 2% std

# Calculate cumulative stock prices
prices = np.zeros(days + 1)
prices[0] = initial_price

for i in range(days):
    prices[i + 1] = prices[i] * (1 + daily_returns[i])

print("Financial Analysis:")
print(f"Starting price: ${initial_price:.2f}")
print(f"Ending price: ${prices[-1]:.2f}")
print(f"Total return: {(prices[-1] / prices[0] - 1) * 100:.2f}%")

# Calculate statistics
print(f"Average daily return: {np.mean(daily_returns) * 100:.3f}%")
print(f"Volatility (daily std): {np.std(daily_returns) * 100:.3f}%")
print(f"Annualized volatility: {np.std(daily_returns) * np.sqrt(252) * 100:.2f}%")

# Risk metrics
max_price = np.max(prices)
min_price = np.min(prices)
max_drawdown = (max_price - min_price) / max_price * 100
print(f"Maximum drawdown: {max_drawdown:.2f}%")

# Moving averages
window = 20
moving_avg = np.convolve(prices[1:], np.ones(window)/window, mode='valid')
print(f"20-day moving average (last): ${moving_avg[-1]:.2f}")
```

## Performance Comparison

### NumPy vs Pure Python
```python
import time
import numpy as np

# Large dataset
size = 1000000
python_list = list(range(size))
numpy_array = np.arange(size)

# Pure Python operation
start_time = time.time()
python_result = [x * 2 for x in python_list]
python_time = time.time() - start_time

# NumPy operation
start_time = time.time()
numpy_result = numpy_array * 2
numpy_time = time.time() - start_time

print(f"Pure Python time: {python_time:.4f} seconds")
print(f"NumPy time: {numpy_time:.4f} seconds")
print(f"NumPy is {python_time / numpy_time:.1f}x faster")

# Memory usage comparison
import sys
python_memory = sys.getsizeof(python_list)
numpy_memory = numpy_array.nbytes
print(f"Python list memory: {python_memory:,} bytes")
print(f"NumPy array memory: {numpy_memory:,} bytes")
print(f"NumPy uses {python_memory / numpy_memory:.1f}x less memory")
```

## Best Practices and Tips

### 1. Use Vectorized Operations
```python
# Avoid loops when possible
# Slow:
result = []
for i in range(len(arr)):
    result.append(arr[i] ** 2)

# Fast:
result = arr ** 2
```

### 2. Choose Appropriate Data Types
```python
# Use smaller data types when possible
large_ints = np.array([1, 2, 3], dtype=np.int64)    # 8 bytes per element
small_ints = np.array([1, 2, 3], dtype=np.int32)    # 4 bytes per element

# For boolean operations
mask = arr > 5  # Returns boolean array automatically
```

### 3. Understand Memory Layout
```python
# Row-major (C-style) vs column-major (Fortran-style)
arr_c = np.array([[1, 2, 3], [4, 5, 6]], order='C')  # Default
arr_f = np.array([[1, 2, 3], [4, 5, 6]], order='F')

# Check if array is contiguous
print(arr_c.flags['C_CONTIGUOUS'])  # True
print(arr_f.flags['F_CONTIGUOUS'])  # True
```

## Common Pitfalls and Solutions

### 1. Array Copying vs Views
```python
arr = np.array([1, 2, 3, 4, 5])

# View (shares memory)
view = arr[1:4]
view[0] = 999
print(arr)  # [1 999 3 4 5] - original changed!

# Copy (independent)
copy = arr[1:4].copy()
copy[0] = 777
print(arr)  # [1 999 3 4 5] - original unchanged
```

### 2. Integer Division
```python
# Be careful with integer division
arr_int = np.array([1, 2, 3])
result = arr_int / 2  # Returns float array [0.5 1.  1.5]

# For integer division
result_int = arr_int // 2  # Returns int array [0 1 1]
```

### 3. Broadcasting Mistakes
```python
# This won't work as expected
arr2d = np.array([[1, 2], [3, 4]])
arr1d = np.array([1, 2, 3])  # Wrong size!

# This will raise an error
# result = arr2d + arr1d  # ValueError: operands could not be broadcast together
```

## Key Terminology

- **Array**: NumPy's main data structure for homogeneous data
- **Vectorization**: Applying operations to entire arrays at once
- **Broadcasting**: Rules for operations between different-shaped arrays
- **Axis**: Dimension along which an operation is performed
- **View**: Array that shares memory with another array
- **Copy**: Independent array with its own memory
- **Dtype**: Data type of array elements
- **Shape**: Dimensions of an array (rows, columns, etc.)

## Looking Ahead

In Lesson 10, we'll learn about:
- **Pandas**: Data manipulation and analysis library built on NumPy
- **DataFrames**: 2D labeled data structures
- **Series**: 1D labeled arrays
- **Data import/export**: Reading CSV, Excel, and other formats
- **Data cleaning**: Handling missing values and data types
