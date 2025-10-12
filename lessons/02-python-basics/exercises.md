# Lesson 02 Exercises: Python Basics - Variables and Data Types

## Guided Exercises (Do with Instructor)

### Exercise 1: Variable Creation and Types
**Goal**: Practice creating variables and identifying their types

**Steps**:
1. Create variables for the following information about yourself:
   ```python
   # Your information here
   name = "Your Name"
   age = 25
   height = 5.8
   is_employed = True
   ```

2. Use the `type()` function to check each variable's type:
   ```python
   print(type(name))
   print(type(age))
   print(type(height))
   print(type(is_employed))
   ```

3. Print each variable with a descriptive label

**Expected Output**: You should see the values and their types displayed clearly

---

### Exercise 2: Type Conversion Practice
**Goal**: Learn to convert between different data types

**Tasks**:
1. Start with these string values:
   ```python
   age_str = "25"
   height_str = "5.8"
   score_str = "95"
   ```

2. Convert them to appropriate numeric types
3. Perform calculations with the converted values
4. Convert numbers back to strings for display

**Example**:
```python
age_str = "25"
age_num = int(age_str)
next_year = age_num + 1
result = f"Next year you'll be {next_year}"
```

---

### Exercise 3: User Input and Formatting
**Goal**: Get user input and display formatted output

**Create a program that**:
1. Asks for the user's name
2. Asks for their favorite number (convert to int)
3. Asks for their height in feet (convert to float)
4. Displays a formatted summary using f-strings

**Template**:
```python
name = input("What's your name? ")
# Add more input statements
# Display formatted results
```

---

## Independent Exercises (Try on Your Own)

### Exercise 4: Personal Profile Creator
**Goal**: Create a comprehensive personal profile program

**Requirements**:
Create a script that collects and displays:
- Full name (first and last)
- Age and birth year
- Height in feet and inches
- Favorite color
- Whether they have pets (yes/no)

**Display the information in a nicely formatted profile**

**Sample Output**:
```
=== Personal Profile ===
Name: Alice Johnson
Age: 28 (born in 1996)
Height: 5.6 feet (67.2 inches)
Favorite Color: Blue
Has Pets: Yes
```

---

### Exercise 5: Simple Calculator with Input
**Goal**: Build an interactive calculator

**Requirements**:
1. Ask user for two numbers
2. Perform all basic operations (+, -, *, /)
3. Display results with proper formatting
4. Handle decimal places appropriately

**Features to include**:
- Clear prompts for input
- Formatted output showing the operation
- Proper decimal formatting (2 decimal places)

**Sample Output**:
```
=== Simple Calculator ===
Enter first number: 15.5
Enter second number: 4.2

Results:
15.5 + 4.2 = 19.7
15.5 - 4.2 = 11.3
15.5 * 4.2 = 65.10
15.5 / 4.2 = 3.69
```

---

### Exercise 6: Unit Converter
**Goal**: Create a multi-unit conversion tool

**Create conversions for**:
1. **Temperature**: Fahrenheit to Celsius and vice versa
2. **Distance**: Miles to kilometers and vice versa
3. **Weight**: Pounds to kilograms and vice versa

**Requirements**:
- Ask user what type of conversion they want
- Get the value to convert
- Display both original and converted values
- Use appropriate formulas and formatting

**Formulas**:
- Celsius = (Fahrenheit - 32) × 5/9
- Fahrenheit = (Celsius × 9/5) + 32
- Kilometers = Miles × 1.60934
- Kilograms = Pounds × 0.453592

---

### Exercise 7: Shopping Cart Calculator
**Goal**: Calculate shopping totals with tax

**Scenario**: Create a program that calculates a shopping total

**Requirements**:
1. Ask for item name and price (3 items)
2. Ask for tax rate (as percentage)
3. Calculate subtotal, tax amount, and total
4. Display itemized receipt

**Sample Output**:
```
=== Shopping Receipt ===
Item 1: Shirt - $25.99
Item 2: Pants - $45.50
Item 3: Shoes - $89.99

Subtotal: $161.48
Tax (8.5%): $13.73
Total: $175.21
```

---

## Challenge Exercises (If You Finish Early)

### Challenge 1: BMI Calculator
**Goal**: Create a health-related calculator

**Requirements**:
1. Get user's weight (pounds) and height (feet and inches)
2. Convert to metric (kg and meters)
3. Calculate BMI using formula: BMI = weight(kg) / height(m)²
4. Display BMI with interpretation

**BMI Categories**:
- Underweight: < 18.5
- Normal: 18.5 - 24.9
- Overweight: 25 - 29.9
- Obese: ≥ 30

---

### Challenge 2: Time Zone Converter
**Goal**: Work with time calculations

**Create a program that**:
1. Gets current time in hours (24-hour format)
2. Asks for time zone difference
3. Calculates time in different zone
4. Handles day changes (0-23 hour range)

**Example**:
```
Current time: 15 (3 PM)
Time zone difference: +8 hours
Time in other zone: 23 (11 PM)
```

---

### Challenge 3: Grade Point Average Calculator
**Goal**: Handle multiple inputs and calculations

**Requirements**:
1. Get grades for 5 subjects (0-100 scale)
2. Get credit hours for each subject
3. Convert grades to GPA scale (A=4, B=3, C=2, D=1, F=0)
4. Calculate weighted GPA
5. Display detailed breakdown

**Grade Scale**:
- A: 90-100 (4.0)
- B: 80-89 (3.0)
- C: 70-79 (2.0)
- D: 60-69 (1.0)
- F: 0-59 (0.0)

---

## Debugging Exercises

### Debug Exercise 1: Find the Errors
**Goal**: Practice identifying and fixing common mistakes

**Fix these code snippets**:

```python
# Code 1 - Type error
age = input("Enter age: ")
next_year = age + 1
print("Next year:", next_year)

# Code 2 - String concatenation error
score = 95
message = "Your score is " + score

# Code 3 - Variable name error
first-name = "Alice"
print(first-name)

# Code 4 - Reserved word error
class = "Math"
print("Subject:", class)
```

### Debug Exercise 2: Logic Errors
**Goal**: Find errors in logic, not syntax

**Fix these calculations**:

```python
# Code 1 - Temperature conversion (wrong formula)
celsius = 25
fahrenheit = celsius * 9/5 - 32

# Code 2 - Percentage calculation (wrong approach)
score = 85
total = 100
percentage = score / total

# Code 3 - Area calculation (wrong formula)
length = 10
width = 5
area = length + width
```

---

## Self-Assessment Checklist

After completing these exercises, you should be able to:

- [ ] Create variables with appropriate names
- [ ] Use all four basic data types (int, float, str, bool)
- [ ] Convert between data types when needed
- [ ] Get input from users and convert it appropriately
- [ ] Format output using f-strings
- [ ] Create and run Python script files
- [ ] Debug common variable and type errors
- [ ] Follow Python naming conventions

## Troubleshooting Guide

### Common Issues and Solutions

**Problem**: `ValueError: invalid literal for int()`
- **Cause**: Trying to convert non-numeric string to int
- **Solution**: Make sure input is a valid number

**Problem**: `TypeError: can only concatenate str (not "int") to str`
- **Cause**: Mixing strings and numbers without conversion
- **Solution**: Convert number to string or use f-strings

**Problem**: `NameError: name 'variable_name' is not defined`
- **Cause**: Using variable before defining it or typo in name
- **Solution**: Check spelling and make sure variable is defined first

**Problem**: `SyntaxError: invalid syntax`
- **Cause**: Missing quotes, parentheses, or using reserved words
- **Solution**: Check syntax carefully, use proper variable names

## Git Reminder

Save your work to your repository:

1. Create folder `lesson-02-python-basics` in your repository
2. Save your exercise solutions as `.py` files
3. Add notes about what you learned
4. Commit and push:
   ```bash
   git add .
   git commit -m "Complete Lesson 02: Python Basics"
   git push
   ```

## Next Lesson Preview

In Lesson 03, we'll learn about:
- **Conditional statements**: Making decisions with if/elif/else
- **Comparison operators**: Testing relationships (>, <, ==, !=)
- **Logical operators**: Combining conditions (and, or, not)
- **Program flow**: Controlling what code runs when
