# Lesson 01: Introduction to Programming

## Learning Objectives
By the end of this lesson, you will be able to:
- Explain what programming is and why it's useful
- Understand basic programming concepts and terminology
- Identify the steps in problem-solving with code
- Write and run your first Python program
- Use Python as a calculator for basic operations

## What is Programming?

### Definition
**Programming** is the process of creating instructions for a computer to follow. Think of it like writing a very detailed recipe that a computer can understand and execute.

### Real-World Analogy: Cooking Recipe
```
Human Recipe:
1. Boil water
2. Add pasta
3. Cook until tender
4. Drain water

Computer Program:
1. Set temperature to 100°C
2. Wait until water reaches 100°C
3. Add 100g pasta
4. Set timer for 8 minutes
5. When timer ends, turn off heat
6. Pour contents through strainer
```

The computer needs much more specific instructions than humans do!

### Why Learn Programming?

**For Data Analysis Specifically:**
- **Automate repetitive tasks**: Instead of manually calculating averages in Excel, write code to do it instantly
- **Handle large datasets**: Work with millions of rows of data that would crash Excel
- **Reproducible analysis**: Others can run your exact same analysis
- **Advanced techniques**: Access to machine learning and statistical methods

**General Benefits:**
- **Problem-solving skills**: Break down complex problems into smaller steps
- **Logical thinking**: Learn to think systematically and precisely
- **Career opportunities**: Programming skills are valuable in many fields
- **Creativity**: Build tools, websites, games, and solutions to real problems

## Key Programming Concepts

### 1. Algorithm
An **algorithm** is a step-by-step procedure to solve a problem.

**Example: Making Coffee**
```
Algorithm: Make Coffee
1. Fill kettle with water
2. Turn on kettle
3. Wait for water to boil
4. Put coffee in cup
5. Pour hot water into cup
6. Stir
7. Enjoy!
```

**Programming Algorithm: Find Average**
```
Algorithm: Calculate Average of Numbers
1. Add all numbers together
2. Count how many numbers there are
3. Divide the sum by the count
4. Display the result
```

### 2. Program vs. Programming Language

**Program**: The actual instructions written for the computer
**Programming Language**: The "language" used to write those instructions

Think of it like:
- **Language**: English, Spanish, French
- **Document**: A letter, book, or email written in that language

- **Programming Language**: Python, Java, JavaScript
- **Program**: A calculator app, website, or data analysis script written in that language

### 3. Why Python?

Python is perfect for beginners and data analysis because:

**Readable**: Looks almost like English
```python
if temperature > 30:
    print("It's hot today!")
```

**Powerful**: Can handle complex data analysis
```python
average_sales = sales_data.mean()
```

**Popular**: Huge community and lots of help available
**Versatile**: Used for web development, data science, AI, automation

### 4. How Computers Execute Programs

```
Source Code (what you write)
        ↓
Python Interpreter (translates)
        ↓
Machine Code (what computer understands)
        ↓
Computer executes instructions
```

## Your First Python Program

### Setting Up
1. Open Command Prompt
2. Type `python` and press Enter
3. You should see something like:
   ```
   Python 3.11.0 (main, Oct 24 2022, 18:26:48)
   >>> 
   ```
4. The `>>>` is called the **Python prompt** - it's waiting for your instructions!

### Hello, World!
This is traditionally the first program every programmer writes:

```python
print("Hello, World!")
```

**Try it now:**
1. Type the line above at the Python prompt
2. Press Enter
3. You should see: `Hello, World!`

**What happened?**
- `print()` is a **function** that displays text
- `"Hello, World!"` is a **string** (text data)
- The parentheses `()` tell Python to execute the function

### Personalizing Your First Program

```python
print("Hello, my name is [Your Name]!")
print("I'm learning Python for data analysis.")
print("This is exciting!")
```

### Python as a Calculator

Python can perform mathematical operations:

```python
# Basic arithmetic
print(5 + 3)        # Addition: 8
print(10 - 4)       # Subtraction: 6
print(6 * 7)        # Multiplication: 42
print(15 / 3)       # Division: 5.0
print(2 ** 3)       # Exponentiation: 8
print(17 % 5)       # Modulus (remainder): 2
```

**Try these calculations:**
```python
# Calculate your age in days (approximately)
print(25 * 365)

# Calculate compound interest
print(1000 * (1.05 ** 10))

# Convert Celsius to Fahrenheit
celsius = 25
fahrenheit = (celsius * 9/5) + 32
print(fahrenheit)
```

## Problem-Solving Process

### The Programming Mindset
When faced with a problem, programmers think:

1. **Understand the Problem**: What exactly needs to be solved?
2. **Break It Down**: What are the smaller steps?
3. **Plan the Solution**: What's the algorithm?
4. **Write the Code**: Translate the plan into Python
5. **Test and Debug**: Does it work? If not, fix it
6. **Improve**: Can it be made better?

### Example: Calculate Restaurant Tip

**Problem**: Calculate a 18% tip on a $45.50 meal

**Step 1 - Understand**: Need to find 18% of $45.50 and add it to the bill

**Step 2 - Break Down**:
- Calculate tip amount (18% of bill)
- Add tip to original bill
- Display the total

**Step 3 - Plan**:
```
1. Set bill amount to 45.50
2. Set tip percentage to 0.18
3. Calculate tip = bill × tip_percentage
4. Calculate total = bill + tip
5. Display the results
```

**Step 4 - Write Code**:
```python
bill = 45.50
tip_percentage = 0.18
tip = bill * tip_percentage
total = bill + tip

print("Bill:", bill)
print("Tip:", tip)
print("Total:", total)
```

**Step 5 - Test**: Run it and check if the math is correct

## Understanding Errors

Errors are normal and helpful! They tell you what's wrong.

### Common First-Day Errors

**Syntax Error** - You wrote something Python doesn't understand:
```python
print("Hello World"  # Missing closing parenthesis
# SyntaxError: unexpected EOF while parsing
```

**Name Error** - You used something that doesn't exist:
```python
print(my_name)  # my_name was never defined
# NameError: name 'my_name' is not defined
```

**Type Error** - You tried to do something impossible:
```python
print("Hello" + 5)  # Can't add text and number
# TypeError: can only concatenate str (not "int") to str
```

### How to Read Error Messages
```
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'my_name' is not defined
```

- **Error Type**: `NameError`
- **Problem**: `name 'my_name' is not defined`
- **Location**: `line 1`

## Programming Best Practices (Starting Early)

### 1. Use Descriptive Names
```python
# Bad
x = 45.50
y = 0.18
z = x * y

# Good
bill_amount = 45.50
tip_percentage = 0.18
tip_amount = bill_amount * tip_percentage
```

### 2. Add Comments
```python
# Calculate restaurant tip
bill_amount = 45.50      # Original bill
tip_percentage = 0.18    # 18% tip
tip_amount = bill_amount * tip_percentage
```

### 3. Test Your Code
Always run your code to make sure it works as expected.

### 4. Start Simple
Begin with the simplest version that works, then improve it.

## Data Analysis Connection

Even this simple lesson connects to data analysis:

**Basic Calculations** → **Statistical Calculations**
```python
# Today: Simple math
print(10 + 20 + 30)

# Later: Analyzing data
print(sum([10, 20, 30, 40, 50]))  # Sum of a dataset
print(sum([10, 20, 30, 40, 50]) / 5)  # Average
```

**Problem Solving** → **Data Questions**
```python
# Today: Calculate tip
bill = 45.50
tip = bill * 0.18

# Later: Analyze sales data
sales_data = [100, 150, 200, 175, 225]
average_sales = sum(sales_data) / len(sales_data)
```

## Interactive Python vs. Script Files

### Interactive Mode (What we've been using)
- Type commands one at a time
- Great for testing and learning
- Results disappear when you close Python

### Script Files (What we'll learn next)
- Save commands in a `.py` file
- Run the entire file at once
- Can be shared and reused

**Preview of next lesson**: We'll create our first Python script file!

## Key Terminology Review

- **Algorithm**: Step-by-step procedure to solve a problem
- **Program**: Set of instructions for a computer
- **Programming Language**: The syntax and rules for writing programs
- **Python**: The programming language we're learning
- **Function**: A piece of code that performs a specific task (like `print()`)
- **String**: Text data enclosed in quotes
- **Syntax**: The rules for writing valid Python code
- **Error**: When Python can't understand or execute your code
- **Comment**: Notes in your code that Python ignores (start with `#`)

## Looking Ahead

In our next lesson, we'll learn about:
- **Variables**: Storing and reusing values
- **Data Types**: Different kinds of data (numbers, text, etc.)
- **Input**: Getting information from the user
- **Script Files**: Saving our programs to run later

## Reflection Questions

1. What's one thing about programming that surprised you today?
2. How might programming help you in your goal of data analysis?
3. What was the most challenging part of this lesson?
4. What questions do you have about programming or Python?

Remember: Every expert programmer started exactly where you are now. The key is practice and patience!
