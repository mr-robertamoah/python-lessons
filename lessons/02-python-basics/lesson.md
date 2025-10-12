# Lesson 02: Python Basics - Variables and Data Types

## Learning Objectives
By the end of this lesson, you will be able to:
- Understand and use variables to store data
- Identify and work with Python's basic data types
- Get input from users with the `input()` function
- Format strings and output data clearly
- Create and run Python script files
- Apply naming conventions and best practices

## What are Variables?

### Real-World Analogy: Storage Boxes
Think of variables as labeled storage boxes:
- The **box** holds a value
- The **label** is the variable name
- You can **put things in** (assign values)
- You can **look inside** (use the value)
- You can **change contents** (reassign)

```python
# Creating a "box" labeled 'age' and putting the number 25 in it
age = 25

# Creating a "box" labeled 'name' and putting text in it
name = "Alice"

# Looking inside the boxes
print(age)    # Shows: 25
print(name)   # Shows: Alice
```

### Why Use Variables?

**Without Variables** (repetitive and error-prone):
```python
print("Bill amount: $45.50")
print("Tip (18%): $", 45.50 * 0.18)
print("Total: $", 45.50 + (45.50 * 0.18))
```

**With Variables** (clear and maintainable):
```python
bill = 45.50
tip_rate = 0.18
tip = bill * tip_rate
total = bill + tip

print("Bill amount: $", bill)
print("Tip (18%): $", tip)
print("Total: $", total)
```

## Variable Assignment

### Basic Syntax
```python
variable_name = value
```

### Examples
```python
# Numbers
age = 25
height = 5.8
temperature = -10

# Text
name = "Alice Johnson"
city = "Seattle"

# True/False values
is_student = True
has_license = False
```

### Multiple Assignment
```python
# Assign same value to multiple variables
x = y = z = 0

# Assign different values at once
name, age, city = "Bob", 30, "Portland"
```

## Python Data Types

### 1. Integers (`int`)
Whole numbers, positive or negative

```python
age = 25
year = 2024
temperature = -5
score = 0

print(type(age))  # <class 'int'>
```

**Common Operations:**
```python
a = 10
b = 3

print(a + b)    # Addition: 13
print(a - b)    # Subtraction: 7
print(a * b)    # Multiplication: 30
print(a // b)   # Integer division: 3
print(a % b)    # Remainder: 1
print(a ** b)   # Exponentiation: 1000
```

### 2. Floating-Point Numbers (`float`)
Numbers with decimal points

```python
height = 5.8
price = 12.99
pi = 3.14159
percentage = 0.18

print(type(height))  # <class 'float'>
```

**Important Notes:**
```python
# Division always returns float
print(10 / 2)     # 5.0 (not 5)
print(type(10/2)) # <class 'float'>

# Mixing int and float gives float
print(5 + 2.5)    # 7.5
print(type(5 + 2.5))  # <class 'float'>
```

### 3. Strings (`str`)
Text data enclosed in quotes

```python
name = "Alice"
message = 'Hello, World!'
address = "123 Main St, Seattle, WA"

print(type(name))  # <class 'str'>
```

**Quote Options:**
```python
# Single quotes
greeting = 'Hello'

# Double quotes
greeting = "Hello"

# Triple quotes (for multi-line)
poem = """Roses are red,
Violets are blue,
Python is awesome,
And so are you!"""
```

**String Operations:**
```python
first_name = "Alice"
last_name = "Johnson"

# Concatenation (joining)
full_name = first_name + " " + last_name
print(full_name)  # Alice Johnson

# Repetition
laugh = "Ha" * 3
print(laugh)  # HaHaHa

# Length
print(len(full_name))  # 13
```

### 4. Booleans (`bool`)
True or False values

```python
is_sunny = True
is_raining = False
has_umbrella = True

print(type(is_sunny))  # <class 'bool'>
```

**Boolean Context:**
```python
# These values are considered False
print(bool(0))        # False
print(bool(""))       # False (empty string)
print(bool(None))     # False

# These values are considered True
print(bool(1))        # True
print(bool("hello"))  # True
print(bool(-5))       # True
```

## Type Conversion

### Checking Types
```python
age = 25
name = "Alice"
height = 5.8

print(type(age))    # <class 'int'>
print(type(name))   # <class 'str'>
print(type(height)) # <class 'float'>
```

### Converting Between Types
```python
# String to number
age_str = "25"
age_num = int(age_str)
print(age_num + 5)  # 30

price_str = "12.99"
price_num = float(price_str)
print(price_num * 2)  # 25.98

# Number to string
score = 95
score_str = str(score)
message = "Your score is " + score_str
print(message)  # Your score is 95

# Float to int (truncates decimal)
height = 5.8
height_int = int(height)
print(height_int)  # 5
```

### Common Conversion Errors
```python
# This will cause an error:
# age = int("twenty-five")  # ValueError: invalid literal

# This will also cause an error:
# result = "5" + 5  # TypeError: can only concatenate str to str
```

## Getting User Input

### The `input()` Function
```python
name = input("What's your name? ")
print("Hello, " + name + "!")
```

**Important**: `input()` always returns a string!

```python
# This won't work as expected:
age = input("How old are you? ")
next_year = age + 1  # Error! Can't add string and number

# Correct way:
age = int(input("How old are you? "))
next_year = age + 1
print("Next year you'll be", next_year)
```

### Input Validation Example
```python
# Get and convert user input
age_str = input("Enter your age: ")
age = int(age_str)

height_str = input("Enter your height in feet: ")
height = float(height_str)

print(f"You are {age} years old and {height} feet tall.")
```

## String Formatting

### Method 1: Concatenation
```python
name = "Alice"
age = 25
message = "Hello, " + name + "! You are " + str(age) + " years old."
print(message)
```

### Method 2: `.format()` Method
```python
name = "Alice"
age = 25
message = "Hello, {}! You are {} years old.".format(name, age)
print(message)
```

### Method 3: f-strings (Recommended)
```python
name = "Alice"
age = 25
message = f"Hello, {name}! You are {age} years old."
print(message)
```

**Advanced f-string Formatting:**
```python
price = 12.99
quantity = 3
total = price * quantity

print(f"Price: ${price:.2f}")           # Price: $12.99
print(f"Quantity: {quantity}")          # Quantity: 3
print(f"Total: ${total:.2f}")           # Total: $38.97
print(f"Per item: ${total/quantity:.2f}") # Per item: $12.99
```

## Creating Script Files

### Interactive vs. Script Mode

**Interactive Mode** (what we've been using):
- Type commands one at a time
- Good for testing and learning
- Results disappear when you close Python

**Script Mode** (saving programs):
- Write all commands in a `.py` file
- Run the entire file at once
- Can be shared and reused

### Creating Your First Script

1. **Create a new file** called `hello.py`
2. **Add this content**:
   ```python
   # My first Python script
   name = input("What's your name? ")
   age = int(input("How old are you? "))
   
   print(f"Hello, {name}!")
   print(f"You are {age} years old.")
   print(f"Next year you'll be {age + 1}!")
   ```
3. **Save the file**
4. **Run it** from Command Prompt:
   ```bash
   python hello.py
   ```

## Variable Naming Rules and Conventions

### Rules (Must Follow)
```python
# Valid names
name = "Alice"
age = 25
first_name = "Bob"
user_score = 100
_private = "secret"
name2 = "Charlie"

# Invalid names (will cause errors)
# 2name = "Invalid"      # Can't start with number
# first-name = "Invalid" # Can't use hyphens
# class = "Invalid"      # Can't use reserved words
```

### Conventions (Should Follow)
```python
# Good: descriptive and clear
student_name = "Alice"
total_score = 95
is_valid = True

# Bad: unclear abbreviations
sn = "Alice"
ts = 95
iv = True

# Good: snake_case for variables
user_email = "alice@email.com"
max_attempts = 3

# Bad: other naming styles (save for later)
userEmail = "alice@email.com"    # camelCase (used in other languages)
MaxAttempts = 3                  # PascalCase (used for classes)
```

### Reserved Words (Cannot Use)
```python
# These are reserved by Python:
# and, as, assert, break, class, continue, def, del, elif, else, 
# except, False, finally, for, from, global, if, import, in, 
# is, lambda, None, not, or, pass, raise, return, True, try, 
# while, with, yield
```

## Practical Examples

### Example 1: Personal Information Collector
```python
# Collect user information
print("=== Personal Information ===")
first_name = input("First name: ")
last_name = input("Last name: ")
age = int(input("Age: "))
height = float(input("Height in feet: "))
is_student = input("Are you a student? (yes/no): ").lower() == "yes"

# Display formatted information
print(f"\n=== Summary ===")
print(f"Name: {first_name} {last_name}")
print(f"Age: {age} years old")
print(f"Height: {height} feet")
print(f"Student status: {is_student}")
```

### Example 2: Simple Calculator
```python
# Get two numbers from user
print("=== Simple Calculator ===")
num1 = float(input("Enter first number: "))
num2 = float(input("Enter second number: "))

# Perform calculations
addition = num1 + num2
subtraction = num1 - num2
multiplication = num1 * num2
division = num1 / num2

# Display results
print(f"\n=== Results ===")
print(f"{num1} + {num2} = {addition}")
print(f"{num1} - {num2} = {subtraction}")
print(f"{num1} * {num2} = {multiplication}")
print(f"{num1} / {num2} = {division:.2f}")
```

### Example 3: Data Analysis Preparation
```python
# Collect data points for analysis
print("=== Daily Step Counter ===")
day1 = int(input("Steps on day 1: "))
day2 = int(input("Steps on day 2: "))
day3 = int(input("Steps on day 3: "))

# Basic analysis
total_steps = day1 + day2 + day3
average_steps = total_steps / 3
goal = 10000

print(f"\n=== Analysis ===")
print(f"Total steps: {total_steps}")
print(f"Average steps: {average_steps:.1f}")
print(f"Daily goal: {goal}")
print(f"Goal achievement: {average_steps/goal*100:.1f}%")
```

## Common Mistakes and How to Avoid Them

### 1. Forgetting to Convert Input
```python
# Wrong:
age = input("Age: ")
next_year = age + 1  # Error!

# Right:
age = int(input("Age: "))
next_year = age + 1
```

### 2. Mixing Data Types
```python
# Wrong:
result = "Score: " + 95  # Error!

# Right:
result = "Score: " + str(95)
# Or better:
result = f"Score: {95}"
```

### 3. Using Reserved Words
```python
# Wrong:
class = "Math"  # Error! 'class' is reserved

# Right:
subject = "Math"
```

### 4. Poor Variable Names
```python
# Unclear:
x = 25
y = "Alice"
z = x + 1

# Clear:
age = 25
name = "Alice"
next_age = age + 1
```

## Best Practices Summary

1. **Use descriptive variable names**: `student_count` not `sc`
2. **Follow naming conventions**: `snake_case` for variables
3. **Convert input appropriately**: Use `int()` or `float()` when needed
4. **Use f-strings for formatting**: More readable than concatenation
5. **Add comments**: Explain what your code does
6. **Test your code**: Run it to make sure it works
7. **Keep it simple**: Start with basic functionality, then improve

## Connection to Data Analysis

These basics are fundamental to data analysis:

**Variables** → **Data Storage**
```python
# Today: Simple variables
temperature = 72.5

# Later: Data collections
temperatures = [72.5, 68.2, 75.1, 69.8, 71.3]
```

**Data Types** → **Data Categories**
```python
# Today: Basic types
student_name = "Alice"    # Categorical data
test_score = 95          # Numerical data
passed = True            # Boolean data

# Later: Data analysis
import pandas as pd
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'score': [95, 87, 92],
    'passed': [True, True, True]
})
```

**Input/Output** → **Data Import/Export**
```python
# Today: User input
data = input("Enter value: ")

# Later: File input
data = pd.read_csv("dataset.csv")
```

## Looking Ahead

In our next lesson, we'll learn about:
- **Conditional statements**: Making decisions in code
- **Comparison operators**: Testing relationships between values
- **Logical operators**: Combining multiple conditions
- **Program flow**: Controlling what happens when

## Key Terminology Review

- **Variable**: A named storage location for data
- **Data Type**: The kind of data (int, float, str, bool)
- **Assignment**: Storing a value in a variable using `=`
- **Type Conversion**: Changing data from one type to another
- **String Formatting**: Creating formatted text output
- **Script File**: A `.py` file containing Python code
- **Input**: Getting data from the user
- **Reserved Word**: Words that Python uses for special purposes
