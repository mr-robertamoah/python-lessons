# Lesson 04: Functions - Creating Reusable Code

## Learning Objectives
By the end of this lesson, you will be able to:
- Define and call functions to organize code
- Use parameters to pass data to functions
- Return values from functions
- Understand variable scope (local vs global)
- Use built-in functions effectively
- Apply functions to solve data analysis problems

## What are Functions?

### Real-World Analogy: Kitchen Appliances
Think of functions like kitchen appliances:

**Blender Function:**
- **Input**: Fruits, ice, milk (parameters)
- **Process**: Blend ingredients (function body)
- **Output**: Smoothie (return value)
- **Reusable**: Make different smoothies with different ingredients

```python
def make_smoothie(fruit, liquid, ice_cubes):
    # Process the ingredients
    result = f"Blending {fruit} with {liquid} and {ice_cubes} ice cubes"
    return f"Delicious {fruit} smoothie!"
```

### Why Use Functions?

**Without Functions** (repetitive code):
```python
# Calculate area of rectangle 1
length1 = 10
width1 = 5
area1 = length1 * width1
print(f"Area 1: {area1}")

# Calculate area of rectangle 2
length2 = 8
width2 = 3
area2 = length2 * width2
print(f"Area 2: {area2}")

# Calculate area of rectangle 3
length3 = 12
width3 = 7
area3 = length3 * width3
print(f"Area 3: {area3}")
```

**With Functions** (clean and reusable):
```python
def calculate_rectangle_area(length, width):
    return length * width

area1 = calculate_rectangle_area(10, 5)
area2 = calculate_rectangle_area(8, 3)
area3 = calculate_rectangle_area(12, 7)

print(f"Area 1: {area1}")
print(f"Area 2: {area2}")
print(f"Area 3: {area3}")
```

## Defining Functions

### Basic Syntax
```python
def function_name(parameters):
    """Optional docstring describing the function"""
    # Function body
    return result  # Optional return statement
```

### Simple Function Examples
```python
# Function with no parameters
def greet():
    print("Hello, World!")

# Function with parameters
def greet_person(name):
    print(f"Hello, {name}!")

# Function with return value
def add_numbers(a, b):
    result = a + b
    return result

# Function with multiple parameters and return
def calculate_circle_area(radius):
    pi = 3.14159
    area = pi * radius ** 2
    return area
```

## Calling Functions

### Basic Function Calls
```python
# Define the function
def say_hello():
    print("Hello from the function!")

# Call the function
say_hello()  # Output: Hello from the function!

# Call it multiple times
say_hello()
say_hello()
```

### Functions with Parameters
```python
def introduce(name, age):
    print(f"Hi, I'm {name} and I'm {age} years old.")

# Call with arguments
introduce("Alice", 25)  # Hi, I'm Alice and I'm 25 years old.
introduce("Bob", 30)    # Hi, I'm Bob and I'm 30 years old.
```

### Functions with Return Values
```python
def multiply(x, y):
    return x * y

# Store the result
result = multiply(5, 3)
print(result)  # 15

# Use directly in expressions
total = multiply(4, 6) + multiply(2, 8)
print(total)  # 24 + 16 = 40
```

## Parameters and Arguments

### Positional Parameters
```python
def create_profile(name, age, city):
    return f"{name}, {age} years old, lives in {city}"

# Arguments must be in correct order
profile = create_profile("Alice", 25, "Seattle")
print(profile)  # Alice, 25 years old, lives in Seattle
```

### Keyword Arguments
```python
def create_profile(name, age, city):
    return f"{name}, {age} years old, lives in {city}"

# Can specify parameter names (order doesn't matter)
profile = create_profile(city="Portland", name="Bob", age=30)
print(profile)  # Bob, 30 years old, lives in Portland
```

### Default Parameters
```python
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

print(greet("Alice"))              # Hello, Alice!
print(greet("Bob", "Hi"))          # Hi, Bob!
print(greet("Charlie", "Hey"))     # Hey, Charlie!
```

### Variable Number of Arguments
```python
# *args for variable positional arguments
def sum_all(*numbers):
    total = 0
    for num in numbers:
        total += num
    return total

print(sum_all(1, 2, 3))        # 6
print(sum_all(1, 2, 3, 4, 5))  # 15

# **kwargs for variable keyword arguments
def create_student(**info):
    result = "Student Info:\n"
    for key, value in info.items():
        result += f"  {key}: {value}\n"
    return result

student = create_student(name="Alice", age=20, major="Computer Science")
print(student)
```

## Return Values

### Single Return Value
```python
def calculate_tax(amount, rate):
    tax = amount * rate
    return tax

tax_owed = calculate_tax(1000, 0.08)
print(f"Tax: ${tax_owed}")  # Tax: $80.0
```

### Multiple Return Values
```python
def analyze_number(num):
    is_positive = num > 0
    is_even = num % 2 == 0
    absolute_value = abs(num)
    return is_positive, is_even, absolute_value

# Unpack multiple return values
positive, even, abs_val = analyze_number(-8)
print(f"Positive: {positive}, Even: {even}, Absolute: {abs_val}")
# Output: Positive: False, Even: True, Absolute: 8
```

### Early Return
```python
def check_password(password):
    if len(password) < 8:
        return "Password too short"
    
    if not any(c.isupper() for c in password):
        return "Password needs uppercase letter"
    
    if not any(c.isdigit() for c in password):
        return "Password needs a number"
    
    return "Password is valid"

result = check_password("abc123")
print(result)  # Password too short
```

## Variable Scope

### Local vs Global Scope
```python
# Global variable
global_var = "I'm global"

def my_function():
    # Local variable
    local_var = "I'm local"
    print(global_var)  # Can access global
    print(local_var)   # Can access local

my_function()
print(global_var)  # Can access global
# print(local_var)  # Error! Can't access local variable outside function
```

### Modifying Global Variables
```python
counter = 0  # Global variable

def increment():
    global counter  # Declare we want to modify global
    counter += 1

print(counter)  # 0
increment()
print(counter)  # 1
increment()
print(counter)  # 2
```

### Best Practice: Avoid Global Variables
```python
# Better approach: pass values and return results
def increment(value):
    return value + 1

counter = 0
counter = increment(counter)  # 1
counter = increment(counter)  # 2
```

## Built-in Functions

### Common Built-in Functions
```python
# Math functions
print(abs(-5))        # 5 (absolute value)
print(max(1, 5, 3))   # 5 (maximum)
print(min(1, 5, 3))   # 1 (minimum)
print(sum([1, 2, 3])) # 6 (sum of list)
print(round(3.7))     # 4 (round to nearest integer)
print(round(3.14159, 2))  # 3.14 (round to 2 decimal places)

# Type conversion functions
print(int("123"))     # 123
print(float("3.14"))  # 3.14
print(str(42))        # "42"
print(bool(1))        # True

# Sequence functions
numbers = [3, 1, 4, 1, 5]
print(len(numbers))   # 5 (length)
print(sorted(numbers)) # [1, 1, 3, 4, 5] (sorted copy)
print(reversed(numbers)) # reversed iterator

# Input/Output
name = input("Enter name: ")  # Get user input
print("Hello", name)          # Display output
```

## Practical Examples

### Example 1: Temperature Converter
```python
def celsius_to_fahrenheit(celsius):
    """Convert Celsius to Fahrenheit"""
    fahrenheit = (celsius * 9/5) + 32
    return fahrenheit

def fahrenheit_to_celsius(fahrenheit):
    """Convert Fahrenheit to Celsius"""
    celsius = (fahrenheit - 32) * 5/9
    return celsius

def temperature_converter():
    """Interactive temperature converter"""
    print("=== Temperature Converter ===")
    print("1. Celsius to Fahrenheit")
    print("2. Fahrenheit to Celsius")
    
    choice = input("Choose conversion (1 or 2): ")
    
    if choice == "1":
        temp = float(input("Enter temperature in Celsius: "))
        result = celsius_to_fahrenheit(temp)
        print(f"{temp}°C = {result:.1f}°F")
    elif choice == "2":
        temp = float(input("Enter temperature in Fahrenheit: "))
        result = fahrenheit_to_celsius(temp)
        print(f"{temp}°F = {result:.1f}°C")
    else:
        print("Invalid choice!")

# Run the converter
temperature_converter()
```

### Example 2: Grade Calculator Functions
```python
def calculate_letter_grade(percentage):
    """Convert percentage to letter grade"""
    if percentage >= 97:
        return "A+"
    elif percentage >= 93:
        return "A"
    elif percentage >= 90:
        return "A-"
    elif percentage >= 87:
        return "B+"
    elif percentage >= 83:
        return "B"
    elif percentage >= 80:
        return "B-"
    elif percentage >= 77:
        return "C+"
    elif percentage >= 73:
        return "C"
    elif percentage >= 70:
        return "C-"
    elif percentage >= 60:
        return "D"
    else:
        return "F"

def calculate_gpa_points(letter_grade):
    """Convert letter grade to GPA points"""
    grade_points = {
        "A+": 4.0, "A": 4.0, "A-": 3.7,
        "B+": 3.3, "B": 3.0, "B-": 2.7,
        "C+": 2.3, "C": 2.0, "C-": 1.7,
        "D": 1.0, "F": 0.0
    }
    return grade_points.get(letter_grade, 0.0)

def analyze_grades(scores):
    """Analyze a list of test scores"""
    if not scores:
        return "No scores to analyze"
    
    average = sum(scores) / len(scores)
    highest = max(scores)
    lowest = min(scores)
    
    letter = calculate_letter_grade(average)
    gpa = calculate_gpa_points(letter)
    
    return {
        "average": round(average, 2),
        "highest": highest,
        "lowest": lowest,
        "letter_grade": letter,
        "gpa_points": gpa,
        "total_scores": len(scores)
    }

# Example usage
test_scores = [85, 92, 78, 96, 88]
analysis = analyze_grades(test_scores)

print("Grade Analysis:")
for key, value in analysis.items():
    print(f"  {key.replace('_', ' ').title()}: {value}")
```

### Example 3: Data Analysis Functions
```python
def calculate_statistics(data):
    """Calculate basic statistics for a dataset"""
    if not data:
        return None
    
    n = len(data)
    mean = sum(data) / n
    
    # Calculate median
    sorted_data = sorted(data)
    if n % 2 == 0:
        median = (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
    else:
        median = sorted_data[n//2]
    
    # Calculate mode (most frequent value)
    frequency = {}
    for value in data:
        frequency[value] = frequency.get(value, 0) + 1
    
    mode = max(frequency, key=frequency.get)
    
    return {
        "count": n,
        "mean": round(mean, 2),
        "median": median,
        "mode": mode,
        "min": min(data),
        "max": max(data),
        "range": max(data) - min(data)
    }

def filter_outliers(data, threshold=2):
    """Remove outliers beyond threshold standard deviations"""
    if len(data) < 2:
        return data
    
    mean = sum(data) / len(data)
    
    # Calculate standard deviation
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    std_dev = variance ** 0.5
    
    # Filter outliers
    filtered = []
    for value in data:
        if abs(value - mean) <= threshold * std_dev:
            filtered.append(value)
    
    return filtered

# Example usage
dataset = [10, 12, 14, 15, 16, 18, 20, 22, 100]  # 100 is an outlier
print("Original data:", dataset)

stats = calculate_statistics(dataset)
print("\nOriginal statistics:")
for key, value in stats.items():
    print(f"  {key}: {value}")

cleaned_data = filter_outliers(dataset)
print(f"\nCleaned data: {cleaned_data}")

cleaned_stats = calculate_statistics(cleaned_data)
print("\nCleaned statistics:")
for key, value in cleaned_stats.items():
    print(f"  {key}: {value}")
```

## Function Documentation

### Docstrings
```python
def calculate_bmi(weight, height):
    """
    Calculate Body Mass Index (BMI).
    
    Parameters:
    weight (float): Weight in kilograms
    height (float): Height in meters
    
    Returns:
    float: BMI value rounded to 2 decimal places
    
    Example:
    >>> calculate_bmi(70, 1.75)
    22.86
    """
    bmi = weight / (height ** 2)
    return round(bmi, 2)

# Access docstring
print(calculate_bmi.__doc__)
```

### Type Hints (Advanced)
```python
def calculate_compound_interest(principal: float, rate: float, years: int) -> float:
    """Calculate compound interest with type hints"""
    return principal * (1 + rate) ** years
```

## Best Practices

### 1. Function Naming
```python
# Good: Descriptive verb-noun combinations
def calculate_total_price(items, tax_rate):
    pass

def validate_email_address(email):
    pass

# Avoid: Unclear or abbreviated names
def calc(x, y):  # What does this calculate?
    pass

def proc_data(d):  # What processing? What data?
    pass
```

### 2. Single Responsibility
```python
# Good: Each function has one clear purpose
def calculate_tax(amount, rate):
    return amount * rate

def format_currency(amount):
    return f"${amount:.2f}"

def display_total(subtotal, tax_rate):
    tax = calculate_tax(subtotal, tax_rate)
    total = subtotal + tax
    print(f"Subtotal: {format_currency(subtotal)}")
    print(f"Tax: {format_currency(tax)}")
    print(f"Total: {format_currency(total)}")

# Avoid: Functions that do too many things
def calculate_and_display_everything(amount, rate):
    tax = amount * rate
    total = amount + tax
    formatted_amount = f"${amount:.2f}"
    formatted_tax = f"${tax:.2f}"
    formatted_total = f"${total:.2f}"
    print(f"Subtotal: {formatted_amount}")
    print(f"Tax: {formatted_tax}")
    print(f"Total: {formatted_total}")
    return total
```

### 3. Pure Functions (When Possible)
```python
# Good: Pure function (same input always gives same output)
def calculate_area(length, width):
    return length * width

# Avoid: Functions with side effects when not necessary
total_area = 0  # Global variable

def add_to_total_area(length, width):
    global total_area
    area = length * width
    total_area += area  # Side effect
    return area
```

## Common Mistakes

### 1. Forgetting Return Statement
```python
# Wrong: Function doesn't return anything
def add_numbers(a, b):
    result = a + b
    # Missing return statement!

total = add_numbers(5, 3)
print(total)  # None

# Right: Include return statement
def add_numbers(a, b):
    result = a + b
    return result
```

### 2. Modifying Mutable Parameters
```python
# Dangerous: Modifying mutable parameters
def add_item(item, shopping_list=[]):  # Default list is mutable!
    shopping_list.append(item)
    return shopping_list

list1 = add_item("apples")     # ['apples']
list2 = add_item("bananas")    # ['apples', 'bananas'] - Unexpected!

# Better: Use None as default
def add_item(item, shopping_list=None):
    if shopping_list is None:
        shopping_list = []
    shopping_list.append(item)
    return shopping_list
```

### 3. Too Many Parameters
```python
# Hard to use: Too many parameters
def create_user(first_name, last_name, email, phone, address, city, state, zip_code, country):
    pass

# Better: Use a dictionary or class
def create_user(user_info):
    # user_info is a dictionary with all the details
    pass
```

## Key Terminology

- **Function**: A reusable block of code that performs a specific task
- **Parameter**: A variable in the function definition
- **Argument**: The actual value passed to the function
- **Return Value**: The result that a function gives back
- **Scope**: The region where a variable can be accessed
- **Local Variable**: Variable defined inside a function
- **Global Variable**: Variable defined outside all functions
- **Docstring**: Documentation string describing what a function does
- **Pure Function**: Function that always returns the same output for the same input

## Looking Ahead

In Lesson 05, we'll learn about:
- **Lists**: Ordered collections of items
- **Dictionaries**: Key-value pairs for structured data
- **Sets**: Collections of unique items
- **Tuples**: Immutable sequences
- **List comprehensions**: Elegant way to create lists
