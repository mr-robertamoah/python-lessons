# Lesson 01 Exercises: Introduction to Programming

## Guided Exercises (Do with Instructor)

### Exercise 1: Your First Program
**Goal**: Get comfortable with the Python prompt and print function

**Steps**:
1. Open Command Prompt and start Python by typing `python`
2. Try these commands one at a time:
   ```python
   print("Hello, World!")
   print("My name is [Your Name]")
   print("Today is my first day programming!")
   ```
3. Create a personalized greeting with at least 3 print statements

**Expected Output**: Your custom greeting messages should appear

---

### Exercise 2: Python Calculator Practice
**Goal**: Use Python for basic mathematical operations

**Tasks**:
1. Calculate these expressions using Python:
   - 25 + 17
   - 100 - 37
   - 12 * 8
   - 144 / 12
   - 3 ** 4 (3 to the power of 4)
   - 23 % 7 (remainder when 23 is divided by 7)

2. Verify your answers with a regular calculator

**Learning Point**: Python can be used as a powerful calculator

---

### Exercise 3: Real-World Calculations
**Goal**: Apply programming to solve practical problems

**Problem**: You're planning a pizza party for 8 people. Each pizza costs $12.99 and serves 3 people.

**Calculate**:
1. How many pizzas do you need?
2. What's the total cost?
3. How much should each person pay?

**Use Python to solve this step by step**:
```python
people = 8
pizza_serves = 3
pizza_cost = 12.99

# Your calculations here
```

---

## Independent Exercises (Try on Your Own)

### Exercise 4: Personal Information Display
**Goal**: Practice using print statements and basic formatting

**Task**: Create a program that displays information about yourself:
- Your name
- Your age
- Your city
- Why you're learning Python
- One fun fact about yourself

**Requirements**:
- Use at least 5 print statements
- Make it look organized and readable

**Example Output**:
```
=== About Me ===
Name: Alex Johnson
Age: 28
City: Seattle
Learning Python because: I want to analyze data at work
Fun fact: I can juggle!
```

---

### Exercise 5: Unit Conversions
**Goal**: Practice mathematical operations with real-world applications

**Tasks**: Create calculations for these conversions:

1. **Distance**: Convert 50 miles to kilometers (1 mile = 1.60934 km)
2. **Weight**: Convert 150 pounds to kilograms (1 pound = 0.453592 kg)
3. **Temperature**: Convert 75°F to Celsius (formula: C = (F - 32) × 5/9)
4. **Time**: Convert 2.5 hours to minutes and seconds

**Format your output clearly**:
```python
# Example for miles to kilometers
miles = 50
kilometers = miles * 1.60934
print(f"{miles} miles = {kilometers} kilometers")
```

---

### Exercise 6: Budget Calculator
**Goal**: Solve a multi-step problem using programming logic

**Scenario**: You have a monthly budget of $3000. Calculate your expenses:
- Rent: $1200
- Food: $400
- Transportation: $200
- Entertainment: $150
- Savings goal: 20% of remaining money

**Calculate**:
1. Total fixed expenses
2. Money left after fixed expenses
3. Amount to save (20% of remaining)
4. Money left for other expenses

**Display your results clearly with labels**

---

### Exercise 7: Data Analysis Preview
**Goal**: Get a taste of what's coming in data analysis

**Task**: You collected daily step counts for a week:
- Monday: 8,432 steps
- Tuesday: 6,891 steps  
- Wednesday: 9,102 steps
- Thursday: 7,654 steps
- Friday: 5,432 steps
- Saturday: 12,098 steps
- Sunday: 9,876 steps

**Calculate**:
1. Total steps for the week
2. Average steps per day
3. Highest and lowest day (just by looking, we'll automate this later)

**Use Python to do the math**:
```python
monday = 8432
tuesday = 6891
# ... continue for all days

total_steps = monday + tuesday + # ... add all days
average_steps = total_steps / 7

print("Total steps:", total_steps)
print("Average steps per day:", average_steps)
```

---

## Challenge Exercises (If You Finish Early)

### Challenge 1: Compound Interest Calculator
**Goal**: Work with more complex mathematical formulas

**Task**: Calculate compound interest using the formula:
A = P(1 + r)^t

Where:
- P = Principal amount ($1000)
- r = Annual interest rate (5% = 0.05)
- t = Time in years (10 years)
- A = Final amount

**Calculate for different scenarios**:
1. $1000 at 5% for 10 years
2. $5000 at 3% for 20 years
3. $500 at 7% for 5 years

---

### Challenge 2: Grade Calculator
**Goal**: Practice with multiple calculations and formatting

**Scenario**: Calculate your course grade based on:
- Homework: 85% (worth 30% of grade)
- Midterm: 78% (worth 35% of grade)  
- Final: 92% (worth 35% of grade)

**Calculate**:
1. Weighted average
2. Letter grade (A: 90+, B: 80-89, C: 70-79, D: 60-69, F: <60)

**Note**: For the letter grade, just calculate the number - we'll learn how to automatically assign letters later!

---

### Challenge 3: Error Investigation
**Goal**: Learn to read and understand error messages

**Task**: Try these commands in Python and observe the errors:

1. `print("Hello World"`  (missing closing parenthesis)
2. `print(hello)`  (undefined variable)
3. `print("5" + 5)`  (mixing text and numbers)
4. `print(10 / 0)`  (division by zero)

**For each error**:
- Write down the error type
- Explain what went wrong
- Fix the code to make it work

---

## Troubleshooting Guide

### Common Issues and Solutions

**Problem**: Python prompt doesn't appear when typing `python`
- **Solution**: Python might not be installed or not in PATH. Try `py` instead of `python`

**Problem**: `SyntaxError: invalid syntax`
- **Solution**: Check for missing quotes, parentheses, or typos

**Problem**: `NameError: name 'X' is not defined`
- **Solution**: Make sure you've defined the variable or spelled it correctly

**Problem**: Numbers look weird (like 5.0 instead of 5)
- **Solution**: This is normal! Python shows decimal points for division results

**Problem**: Can't see previous commands
- **Solution**: Use the up arrow key to recall previous commands

---

## Self-Assessment Checklist

After completing these exercises, you should be able to:

- [ ] Start Python and use the interactive prompt
- [ ] Write and execute print statements
- [ ] Perform basic mathematical operations (+, -, *, /, **, %)
- [ ] Understand what an algorithm is
- [ ] Break down simple problems into steps
- [ ] Read and understand basic error messages
- [ ] Use Python as a calculator for real-world problems

## Reflection Questions

1. What was the most surprising thing you learned about programming today?
2. Which exercise was most challenging? Why?
3. How do you think these basic skills will help with data analysis?
4. What questions do you have about programming or Python?

## Next Lesson Preview

In Lesson 02, we'll learn about:
- **Variables**: Storing values to use later
- **Data Types**: Different kinds of data (numbers, text, true/false)
- **User Input**: Getting information from the person using your program
- **Script Files**: Saving your programs to run anytime

## Git Reminder

Don't forget to save your work to your Git repository:

1. Create a folder called `lesson-01-intro-programming` in your repository
2. Save any code you wrote as `.py` files in that folder
3. Add notes about what you learned
4. Commit and push your changes:
   ```bash
   git add .
   git commit -m "Complete Lesson 01: Introduction to Programming"
   git push
   ```
