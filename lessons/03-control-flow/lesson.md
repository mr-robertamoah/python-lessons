# Lesson 03: Control Flow - Making Decisions and Loops

## Learning Objectives
By the end of this lesson, you will be able to:
- Use conditional statements (if, elif, else) to make decisions
- Apply comparison and logical operators
- Create loops (for, while) to repeat code
- Control loop execution with break and continue
- Solve problems using decision-making logic

## What is Control Flow?

### Real-World Analogy: Daily Decisions
Every day you make decisions based on conditions:

```
If it's raining:
    Take an umbrella
Else if it's sunny:
    Wear sunglasses
Else:
    Just go outside normally

While I'm hungry:
    Look for food
    Eat food
    Check if still hungry
```

**Control flow** in programming works the same way - it determines which code runs based on conditions.

## Conditional Statements

### Basic if Statement
```python
temperature = 75

if temperature > 70:
    print("It's warm today!")
    print("Perfect weather for a walk.")
```

### if-else Statement
```python
age = 17

if age >= 18:
    print("You can vote!")
else:
    print("You're not old enough to vote yet.")
```

### if-elif-else Statement
```python
score = 85

if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
elif score >= 60:
    grade = "D"
else:
    grade = "F"

print(f"Your grade is: {grade}")
```

## Comparison Operators

### Basic Comparisons
```python
# Equal to
print(5 == 5)    # True
print(5 == 3)    # False

# Not equal to
print(5 != 3)    # True
print(5 != 5)    # False

# Greater than / Less than
print(10 > 5)    # True
print(3 < 8)     # True

# Greater than or equal / Less than or equal
print(5 >= 5)    # True
print(4 <= 3)    # False
```

### Comparing Strings
```python
name = "Alice"

if name == "Alice":
    print("Hello Alice!")

# Case-sensitive comparison
print("alice" == "Alice")  # False
print("alice".lower() == "Alice".lower())  # True
```

### Comparing with User Input
```python
password = input("Enter password: ")

if password == "secret123":
    print("Access granted!")
else:
    print("Access denied!")
```

## Logical Operators

### and Operator
Both conditions must be True
```python
age = 25
has_license = True

if age >= 18 and has_license:
    print("You can drive!")
else:
    print("You cannot drive.")
```

### or Operator
At least one condition must be True
```python
day = "Saturday"

if day == "Saturday" or day == "Sunday":
    print("It's the weekend!")
else:
    print("It's a weekday.")
```

### not Operator
Reverses the condition
```python
is_raining = False

if not is_raining:
    print("Great day for a picnic!")
else:
    print("Better stay inside.")
```

### Complex Conditions
```python
age = 25
income = 50000
credit_score = 720

if (age >= 21 and income >= 30000) or credit_score >= 700:
    print("Loan approved!")
else:
    print("Loan denied.")
```

## Loops

### for Loops
Repeat code a specific number of times

```python
# Loop through numbers
for i in range(5):
    print(f"Count: {i}")
# Output: 0, 1, 2, 3, 4

# Loop through a range with start and end
for num in range(1, 6):
    print(f"Number: {num}")
# Output: 1, 2, 3, 4, 5

# Loop through text
name = "Python"
for letter in name:
    print(letter)
# Output: P, y, t, h, o, n
```

### while Loops
Repeat code while a condition is True

```python
count = 0
while count < 5:
    print(f"Count is: {count}")
    count += 1  # Same as count = count + 1

print("Loop finished!")
```

### Practical Loop Examples
```python
# Calculate sum of numbers 1 to 10
total = 0
for i in range(1, 11):
    total += i
print(f"Sum: {total}")  # 55

# Countdown timer
countdown = 5
while countdown > 0:
    print(f"T-minus {countdown}")
    countdown -= 1
print("Blast off!")
```

## Loop Control

### break Statement
Exit the loop immediately
```python
# Find first number divisible by 7
for num in range(1, 100):
    if num % 7 == 0:
        print(f"First number divisible by 7: {num}")
        break
```

### continue Statement
Skip to next iteration
```python
# Print only odd numbers
for num in range(1, 11):
    if num % 2 == 0:  # If even
        continue      # Skip to next iteration
    print(num)        # Only prints odd numbers
```

## Nested Control Structures

### Nested if Statements
```python
weather = "sunny"
temperature = 75

if weather == "sunny":
    if temperature > 70:
        print("Perfect beach weather!")
    else:
        print("Sunny but a bit cool.")
else:
    print("Not sunny today.")
```

### Nested Loops
```python
# Multiplication table
for i in range(1, 4):
    for j in range(1, 4):
        result = i * j
        print(f"{i} x {j} = {result}")
    print()  # Empty line after each row
```

## Practical Examples

### Example 1: Grade Calculator
```python
print("=== Grade Calculator ===")

# Get scores
homework = float(input("Homework average (0-100): "))
midterm = float(input("Midterm score (0-100): "))
final = float(input("Final exam score (0-100): "))

# Calculate weighted average
total_score = (homework * 0.3) + (midterm * 0.3) + (final * 0.4)

# Determine letter grade
if total_score >= 90:
    letter = "A"
    message = "Excellent work!"
elif total_score >= 80:
    letter = "B"
    message = "Good job!"
elif total_score >= 70:
    letter = "C"
    message = "Satisfactory."
elif total_score >= 60:
    letter = "D"
    message = "Needs improvement."
else:
    letter = "F"
    message = "Please see instructor."

print(f"\nFinal Score: {total_score:.1f}%")
print(f"Letter Grade: {letter}")
print(f"Comment: {message}")
```

### Example 2: Number Guessing Game
```python
import random

print("=== Number Guessing Game ===")
secret_number = random.randint(1, 100)
attempts = 0
max_attempts = 7

print("I'm thinking of a number between 1 and 100.")
print(f"You have {max_attempts} attempts to guess it!")

while attempts < max_attempts:
    guess = int(input("Enter your guess: "))
    attempts += 1
    
    if guess == secret_number:
        print(f"Congratulations! You guessed it in {attempts} attempts!")
        break
    elif guess < secret_number:
        print("Too low!")
    else:
        print("Too high!")
    
    remaining = max_attempts - attempts
    if remaining > 0:
        print(f"You have {remaining} attempts left.")
    else:
        print(f"Game over! The number was {secret_number}.")
```

### Example 3: Data Validation
```python
print("=== User Registration ===")

# Validate age
while True:
    age = int(input("Enter your age: "))
    if 13 <= age <= 120:
        break
    else:
        print("Age must be between 13 and 120. Please try again.")

# Validate email
while True:
    email = input("Enter your email: ")
    if "@" in email and "." in email:
        break
    else:
        print("Please enter a valid email address.")

# Validate password
while True:
    password = input("Create a password (min 8 characters): ")
    if len(password) >= 8:
        break
    else:
        print("Password must be at least 8 characters long.")

print("Registration successful!")
print(f"Age: {age}")
print(f"Email: {email}")
```

## Common Patterns and Idioms

### Input Validation Pattern
```python
while True:
    try:
        value = int(input("Enter a number: "))
        if value > 0:
            break
        else:
            print("Please enter a positive number.")
    except ValueError:
        print("Please enter a valid number.")
```

### Menu System Pattern
```python
while True:
    print("\n=== Main Menu ===")
    print("1. Option A")
    print("2. Option B")
    print("3. Quit")
    
    choice = input("Choose an option: ")
    
    if choice == "1":
        print("You chose Option A")
    elif choice == "2":
        print("You chose Option B")
    elif choice == "3":
        print("Goodbye!")
        break
    else:
        print("Invalid choice. Please try again.")
```

### Accumulator Pattern
```python
# Sum all numbers from user until they enter 0
total = 0
count = 0

while True:
    num = float(input("Enter a number (0 to stop): "))
    if num == 0:
        break
    total += num
    count += 1

if count > 0:
    average = total / count
    print(f"Sum: {total}")
    print(f"Count: {count}")
    print(f"Average: {average:.2f}")
```

## Data Analysis Applications

### Filtering Data
```python
# Analyze test scores
scores = [85, 92, 78, 96, 88, 73, 91, 87]

# Count high scores
high_scores = 0
for score in scores:
    if score >= 90:
        high_scores += 1

print(f"High scores (90+): {high_scores}")

# Find failing grades
failing_scores = []
for score in scores:
    if score < 70:
        failing_scores.append(score)

print(f"Failing scores: {failing_scores}")
```

### Statistical Analysis
```python
# Calculate statistics
data = [12, 15, 18, 21, 19, 16, 14, 20, 17, 13]

# Find min and max
minimum = data[0]
maximum = data[0]

for value in data:
    if value < minimum:
        minimum = value
    if value > maximum:
        maximum = value

print(f"Minimum: {minimum}")
print(f"Maximum: {maximum}")
print(f"Range: {maximum - minimum}")
```

## Best Practices

### 1. Use Clear Conditions
```python
# Good: Clear and readable
if age >= 18:
    can_vote = True

# Avoid: Unclear logic
if not age < 18:
    can_vote = True
```

### 2. Avoid Deep Nesting
```python
# Good: Early return pattern
def check_eligibility(age, income):
    if age < 18:
        return "Too young"
    if income < 30000:
        return "Income too low"
    return "Eligible"

# Avoid: Deep nesting
def check_eligibility_bad(age, income):
    if age >= 18:
        if income >= 30000:
            return "Eligible"
        else:
            return "Income too low"
    else:
        return "Too young"
```

### 3. Use Meaningful Variable Names
```python
# Good
is_weekend = day in ["Saturday", "Sunday"]
if is_weekend:
    print("Sleep in!")

# Avoid
x = day in ["Saturday", "Sunday"]
if x:
    print("Sleep in!")
```

## Common Mistakes

### 1. Assignment vs. Comparison
```python
# Wrong: Assignment (=) instead of comparison (==)
if age = 18:  # SyntaxError
    print("Just turned 18!")

# Correct: Use == for comparison
if age == 18:
    print("Just turned 18!")
```

### 2. Infinite Loops
```python
# Wrong: Infinite loop
count = 0
while count < 10:
    print(count)
    # Forgot to increment count!

# Correct: Remember to update loop variable
count = 0
while count < 10:
    print(count)
    count += 1
```

### 3. Off-by-One Errors
```python
# Wrong: Misses last element
for i in range(len(data) - 1):
    print(data[i])

# Correct: Includes all elements
for i in range(len(data)):
    print(data[i])

# Even better: Direct iteration
for item in data:
    print(item)
```

## Key Terminology

- **Conditional Statement**: Code that runs based on a condition
- **Boolean Expression**: Expression that evaluates to True or False
- **Comparison Operator**: Operators like ==, !=, <, >, <=, >=
- **Logical Operator**: and, or, not operators
- **Loop**: Code structure that repeats instructions
- **Iteration**: One pass through a loop
- **Infinite Loop**: Loop that never ends (usually a bug)
- **Break**: Statement that exits a loop
- **Continue**: Statement that skips to next iteration

## Looking Ahead

In Lesson 04, we'll learn about:
- **Functions**: Creating reusable code blocks
- **Parameters and Arguments**: Passing data to functions
- **Return Values**: Getting results from functions
- **Scope**: Where variables can be accessed
- **Modules**: Organizing code into separate files
