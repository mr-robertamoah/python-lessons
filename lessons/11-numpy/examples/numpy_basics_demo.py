"""
Lesson 09 Example 1: NumPy Basics Demo
Demonstrates array creation, operations, and basic functionality
"""

import numpy as np
import time

print("=== NumPy Basics Demo ===\n")

# 1. Array Creation
print("1. ARRAY CREATION")
print("=" * 30)

# From lists
arr1d = np.array([1, 2, 3, 4, 5])
arr2d = np.array([[1, 2, 3], [4, 5, 6]])

print(f"1D array: {arr1d}")
print(f"2D array:\n{arr2d}")

# Special arrays
zeros = np.zeros(5)
ones = np.ones((2, 3))
range_arr = np.arange(0, 10, 2)
linspace_arr = np.linspace(0, 1, 5)

print(f"Zeros: {zeros}")
print(f"Ones:\n{ones}")
print(f"Range: {range_arr}")
print(f"Linspace: {linspace_arr}")
print()

# 2. Array Properties
print("2. ARRAY PROPERTIES")
print("=" * 30)
data = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

print(f"Shape: {data.shape}")
print(f"Size: {data.size}")
print(f"Dimensions: {data.ndim}")
print(f"Data type: {data.dtype}")
print()

# 3. Mathematical Operations
print("3. MATHEMATICAL OPERATIONS")
print("=" * 30)
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

print(f"Array a: {a}")
print(f"Array b: {b}")
print(f"Addition: {a + b}")
print(f"Multiplication: {a * b}")
print(f"Square: {a ** 2}")
print(f"Square root: {np.sqrt(a)}")
print()

# 4. Statistical Operations
print("4. STATISTICAL OPERATIONS")
print("=" * 30)
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

print(f"Data: {data}")
print(f"Mean: {np.mean(data)}")
print(f"Median: {np.median(data)}")
print(f"Standard deviation: {np.std(data):.2f}")
print(f"Min: {np.min(data)}")
print(f"Max: {np.max(data)}")
print()

# 5. Broadcasting Example
print("5. BROADCASTING")
print("=" * 30)
matrix = np.array([[1, 2, 3], [4, 5, 6]])
vector = np.array([10, 20, 30])

print(f"Matrix:\n{matrix}")
print(f"Vector: {vector}")
print(f"Matrix + Vector:\n{matrix + vector}")
print()

# 6. Performance Comparison
print("6. PERFORMANCE COMPARISON")
print("=" * 30)

size = 1000000
python_list = list(range(size))
numpy_array = np.arange(size)

# Python list operation
start = time.time()
python_result = [x * 2 for x in python_list]
python_time = time.time() - start

# NumPy operation
start = time.time()
numpy_result = numpy_array * 2
numpy_time = time.time() - start

print(f"Python list time: {python_time:.4f} seconds")
print(f"NumPy array time: {numpy_time:.4f} seconds")
print(f"NumPy is {python_time / numpy_time:.1f}x faster")
print()

# 7. Practical Example: Grade Analysis
print("7. PRACTICAL EXAMPLE: GRADE ANALYSIS")
print("=" * 40)

# Student grades (rows=students, columns=tests)
grades = np.array([
    [85, 92, 78, 96],  # Student 1
    [76, 84, 91, 83],  # Student 2
    [92, 95, 89, 94],  # Student 3
    [68, 72, 75, 70],  # Student 4
    [88, 86, 92, 89]   # Student 5
])

print("Grade Matrix:")
print(grades)
print()

# Calculate statistics
student_averages = np.mean(grades, axis=1)
test_averages = np.mean(grades, axis=0)
overall_average = np.mean(grades)

print("Analysis Results:")
print(f"Student averages: {student_averages}")
print(f"Test averages: {test_averages}")
print(f"Overall average: {overall_average:.2f}")

# Find top student
top_student = np.argmax(student_averages)
print(f"Top student: Student {top_student + 1} with average {student_averages[top_student]:.2f}")

# Grade distribution
passing_grades = grades >= 70
passing_rate = np.mean(passing_grades) * 100
print(f"Passing rate: {passing_rate:.1f}%")
