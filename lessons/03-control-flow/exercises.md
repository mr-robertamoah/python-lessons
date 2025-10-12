# Lesson 03 Exercises: Control Flow

## Guided Exercises (Do with Instructor)

### Exercise 1: Age Category Classifier
**Goal**: Practice if-elif-else statements

**Create a program that**:
1. Asks for a person's age
2. Classifies them into categories:
   - 0-12: Child
   - 13-19: Teenager  
   - 20-64: Adult
   - 65+: Senior
3. Displays appropriate message for each category

**Template**:
```python
age = int(input("Enter your age: "))

if age <= 12:
    category = "Child"
    message = "Enjoy your childhood!"
# Continue with elif statements...
```

---

### Exercise 2: Simple Calculator with Conditions
**Goal**: Combine user input with conditional logic

**Requirements**:
1. Ask user for two numbers
2. Ask for operation (+, -, *, /)
3. Perform calculation based on operation
4. Handle division by zero
5. Display result with proper formatting

**Expected behavior**:
- If operation is invalid, show error message
- If dividing by zero, show warning
- Otherwise, show calculation result

---

### Exercise 3: Number Analysis Loop
**Goal**: Practice for loops with conditions

**Create a program that**:
1. Uses a for loop to check numbers 1 to 20
2. For each number, prints:
   - If it's even or odd
   - If it's divisible by 3
   - If it's a multiple of 5
3. Counts how many numbers meet each condition

**Sample Output**:
```
Number 1: Odd
Number 2: Even
Number 3: Odd, Divisible by 3
Number 5: Odd, Multiple of 5
...
Summary: 10 even, 10 odd, 6 divisible by 3, 4 multiples of 5
```

---

## Independent Exercises (Try on Your Own)

### Exercise 4: Password Strength Checker
**Goal**: Practice complex conditional logic

**Requirements**:
Create a program that evaluates password strength based on:
- Length (at least 8 characters)
- Contains uppercase letter
- Contains lowercase letter  
- Contains number
- Contains special character (!@#$%^&*)

**Scoring**:
- Each criterion met = 1 point
- 5 points = "Very Strong"
- 4 points = "Strong"  
- 3 points = "Medium"
- 2 points = "Weak"
- 0-1 points = "Very Weak"

**Sample Output**:
```
Enter password: MyPass123!
✓ Length requirement met
✓ Contains uppercase
✓ Contains lowercase  
✓ Contains number
✓ Contains special character
Password strength: Very Strong (5/5)
```

---

### Exercise 5: Grade Statistics Calculator
**Goal**: Combine loops with data analysis

**Create a program that**:
1. Asks how many students' grades to enter
2. Uses a loop to collect all grades
3. Calculates and displays:
   - Average grade
   - Highest grade
   - Lowest grade
   - Number of A's (90+), B's (80-89), C's (70-79), etc.
   - Pass rate (assuming 60+ is passing)

**Validation**: Ensure grades are between 0-100

---

### Exercise 6: Number Guessing Game
**Goal**: Practice while loops and game logic

**Requirements**:
1. Computer picks random number 1-100
2. User has 7 attempts to guess
3. After each guess, provide hint (too high/too low)
4. Track number of attempts
5. Offer to play again when game ends

**Features to include**:
- Input validation (ensure guess is 1-100)
- Congratulatory message based on attempts taken
- Option to play multiple rounds

---

### Exercise 7: Menu-Driven Calculator
**Goal**: Create a complete interactive program

**Create a calculator with menu**:
```
=== Calculator Menu ===
1. Addition
2. Subtraction  
3. Multiplication
4. Division
5. Power
6. Square Root
7. Quit

Choose an option: 
```

**Requirements**:
- Loop until user chooses quit
- Handle invalid menu choices
- Perform selected operation
- Handle mathematical errors (division by zero, negative square root)
- Show results clearly

---

## Challenge Exercises (If You Finish Early)

### Challenge 1: Prime Number Finder
**Goal**: Advanced loop logic and mathematical thinking

**Create a program that**:
1. Asks for a range (start and end numbers)
2. Finds all prime numbers in that range
3. Displays them in a formatted list
4. Shows count of primes found

**Prime number**: A number greater than 1 that has no positive divisors other than 1 and itself

**Hint**: Use nested loops to check if each number is divisible by any number from 2 to its square root

---

### Challenge 2: Text Analysis Tool
**Goal**: String processing with loops and conditions

**Create a program that analyzes text input**:
1. Ask user for a sentence or paragraph
2. Count and report:
   - Total characters (including spaces)
   - Total characters (excluding spaces)
   - Number of words
   - Number of vowels
   - Number of consonants
   - Most frequent character

**Sample Output**:
```
Text: "Hello World"
Total characters: 11
Characters (no spaces): 10
Words: 2
Vowels: 3
Consonants: 7
Most frequent: 'l' (appears 3 times)
```

---

### Challenge 3: Simple Banking System
**Goal**: Complex program with multiple features

**Create a banking simulation with**:
1. Account balance (starts at $1000)
2. Menu options:
   - Check balance
   - Deposit money
   - Withdraw money
   - View transaction history
   - Quit

**Requirements**:
- Validate all inputs (positive amounts, sufficient funds)
- Keep transaction history
- Show running balance after each transaction
- Handle overdraft attempts gracefully

---

## Debugging Exercises

### Debug Exercise 1: Fix the Logic Errors
**Goal**: Practice identifying logical mistakes

**Fix these code snippets**:

```python
# Problem 1: Grade assignment
score = 85
if score > 90:
    grade = "A"
if score > 80:
    grade = "B"  # This will always override A grades!
if score > 70:
    grade = "C"
print(f"Grade: {grade}")

# Problem 2: Infinite loop
count = 1
while count <= 10:
    print(count)
    # Missing increment!

# Problem 3: Wrong comparison
password = input("Enter password: ")
if password = "secret":  # Should be ==
    print("Access granted")
```

### Debug Exercise 2: Off-by-One Errors
**Goal**: Identify and fix common loop mistakes

```python
# Problem 1: Missing last element
numbers = [1, 2, 3, 4, 5]
for i in range(len(numbers) - 1):  # Should be len(numbers)
    print(numbers[i])

# Problem 2: Wrong range
# Want to print numbers 1 to 10 inclusive
for i in range(1, 10):  # Should be range(1, 11)
    print(i)
```

---

## Self-Assessment Checklist

After completing these exercises, you should be able to:

- [ ] Write if, elif, else statements for decision making
- [ ] Use comparison operators (==, !=, <, >, <=, >=)
- [ ] Combine conditions with logical operators (and, or, not)
- [ ] Create for loops to iterate over ranges and sequences
- [ ] Write while loops with proper termination conditions
- [ ] Use break and continue to control loop execution
- [ ] Validate user input with loops and conditions
- [ ] Debug common logical errors in conditional statements
- [ ] Create menu-driven programs with user interaction

## Common Mistakes to Avoid

### 1. Assignment vs Comparison
```python
# Wrong
if x = 5:  # SyntaxError

# Right  
if x == 5:
```

### 2. Infinite Loops
```python
# Wrong - infinite loop
while True:
    print("This will run forever!")

# Right - proper termination
count = 0
while count < 10:
    print(count)
    count += 1
```

### 3. Indentation Errors
```python
# Wrong - inconsistent indentation
if age >= 18:
print("Can vote")  # IndentationError

# Right - consistent indentation
if age >= 18:
    print("Can vote")
```

## Git Reminder

Save your work:
1. Create `lesson-03-control-flow` folder in your repository
2. Save exercise solutions as `.py` files
3. Commit and push:
   ```bash
   git add .
   git commit -m "Complete Lesson 03: Control Flow"
   git push
   ```

## Next Lesson Preview

In Lesson 04, we'll learn about:
- **Functions**: Creating reusable code blocks
- **Parameters**: Passing data to functions
- **Return values**: Getting results back
- **Scope**: Understanding where variables exist
- **Built-in functions**: Using Python's pre-made tools
