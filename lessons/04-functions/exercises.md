# Lesson 04 Exercises: Functions

## Guided Exercises (Do with Instructor)

### Exercise 1: Basic Function Creation
**Goal**: Practice defining and calling simple functions

**Create these functions**:
1. `say_hello()` - prints "Hello, World!"
2. `introduce(name, age)` - prints introduction with name and age
3. `add_two_numbers(a, b)` - returns sum of two numbers
4. `is_even(number)` - returns True if number is even, False otherwise

**Test each function**:
```python
say_hello()
introduce("Alice", 25)
result = add_two_numbers(10, 15)
print(f"Sum: {result}")
print(f"Is 8 even? {is_even(8)}")
```

---

### Exercise 2: Calculator Functions
**Goal**: Create a set of mathematical functions

**Requirements**:
Create functions for basic operations:
- `add(a, b)` - addition
- `subtract(a, b)` - subtraction  
- `multiply(a, b)` - multiplication
- `divide(a, b)` - division (handle division by zero)
- `power(base, exponent)` - exponentiation

**Test with a simple calculator program**:
```python
def simple_calculator():
    num1 = float(input("Enter first number: "))
    num2 = float(input("Enter second number: "))
    operation = input("Enter operation (+, -, *, /, **): ")
    
    # Use your functions based on operation
    # Display the result
```

---

### Exercise 3: String Processing Functions
**Goal**: Work with functions that process text

**Create these functions**:
1. `count_vowels(text)` - count vowels in a string
2. `reverse_string(text)` - return reversed string
3. `is_palindrome(text)` - check if text reads same forwards/backwards
4. `word_count(text)` - count words in text

**Test examples**:
```python
print(count_vowels("Hello World"))  # Should return 3
print(reverse_string("Python"))     # Should return "nohtyP"
print(is_palindrome("racecar"))     # Should return True
print(word_count("Hello world"))    # Should return 2
```

---

## Independent Exercises (Try on Your Own)

### Exercise 4: Temperature Conversion Suite
**Goal**: Create a comprehensive temperature converter

**Requirements**:
Create functions for all temperature conversions:
- `celsius_to_fahrenheit(c)`
- `fahrenheit_to_celsius(f)`
- `celsius_to_kelvin(c)`
- `kelvin_to_celsius(k)`
- `fahrenheit_to_kelvin(f)`
- `kelvin_to_fahrenheit(k)`

**Also create**:
- `temperature_converter()` - interactive menu system
- `format_temperature(temp, unit)` - format output nicely

**Sample interaction**:
```
=== Temperature Converter ===
1. Celsius to Fahrenheit
2. Fahrenheit to Celsius
3. Celsius to Kelvin
... (other options)
Choose conversion: 1
Enter temperature: 25
25.0°C = 77.0°F
```

---

### Exercise 5: Grade Analysis System
**Goal**: Build a complete grade management system using functions

**Create these functions**:
1. `letter_grade(percentage)` - convert percentage to letter grade
2. `gpa_points(letter)` - convert letter grade to GPA points
3. `calculate_average(scores)` - calculate average of score list
4. `find_highest_lowest(scores)` - return highest and lowest scores
5. `grade_distribution(scores)` - count A's, B's, C's, etc.
6. `analyze_student_performance(scores)` - comprehensive analysis

**Requirements**:
- Handle empty score lists gracefully
- Use proper grade scale (A: 90+, B: 80-89, etc.)
- Return formatted results

**Test with sample data**:
```python
student_scores = [85, 92, 78, 96, 88, 73, 91, 87]
analysis = analyze_student_performance(student_scores)
print(analysis)
```

---

### Exercise 6: Personal Finance Calculator
**Goal**: Create financial calculation functions

**Create functions for**:
1. `simple_interest(principal, rate, time)` - calculate simple interest
2. `compound_interest(principal, rate, time, compounds_per_year)` - compound interest
3. `monthly_payment(loan_amount, annual_rate, years)` - loan payment calculator
4. `savings_goal(target_amount, monthly_deposit, annual_rate)` - time to reach goal
5. `budget_analyzer(income, expenses_dict)` - analyze budget

**Example usage**:
```python
# Simple interest calculation
interest = simple_interest(1000, 0.05, 2)
print(f"Interest earned: ${interest}")

# Budget analysis
monthly_income = 5000
expenses = {
    'rent': 1500,
    'food': 600,
    'transportation': 300,
    'entertainment': 200,
    'utilities': 150
}
analysis = budget_analyzer(monthly_income, expenses)
```

---

### Exercise 7: Data Validation Functions
**Goal**: Create robust input validation functions

**Create validation functions**:
1. `validate_email(email)` - check email format
2. `validate_phone(phone)` - check phone number format
3. `validate_password(password)` - check password strength
4. `validate_age(age)` - check age is reasonable (0-150)
5. `validate_date(date_string)` - check date format (MM/DD/YYYY)
6. `get_valid_input(prompt, validation_function)` - generic validator

**Requirements**:
- Return True/False or error messages
- Handle edge cases gracefully
- Provide clear feedback for invalid input

**Example usage**:
```python
def register_user():
    email = get_valid_input("Enter email: ", validate_email)
    phone = get_valid_input("Enter phone: ", validate_phone)
    password = get_valid_input("Create password: ", validate_password)
    # ... continue registration
```

---

## Challenge Exercises (If You Finish Early)

### Challenge 1: Statistical Analysis Functions
**Goal**: Create advanced data analysis functions

**Create functions for**:
1. `calculate_mean(data)` - arithmetic mean
2. `calculate_median(data)` - middle value
3. `calculate_mode(data)` - most frequent value
4. `calculate_standard_deviation(data)` - measure of spread
5. `calculate_correlation(x_data, y_data)` - correlation coefficient
6. `remove_outliers(data, method='iqr')` - outlier detection and removal

**Requirements**:
- Handle edge cases (empty lists, single values)
- Implement proper statistical formulas
- Return meaningful error messages for invalid input

---

### Challenge 2: Text Analysis Suite
**Goal**: Advanced text processing functions

**Create functions for**:
1. `word_frequency(text)` - count frequency of each word
2. `reading_level(text)` - estimate reading difficulty
3. `extract_emails(text)` - find email addresses in text
4. `extract_phone_numbers(text)` - find phone numbers
5. `sentiment_score(text)` - basic sentiment analysis
6. `text_summary(text, num_sentences)` - extract key sentences

**Advanced features**:
- Handle punctuation and capitalization
- Use regular expressions for pattern matching
- Return structured data (dictionaries, lists)

---

### Challenge 3: Game Functions Library
**Goal**: Create reusable game components

**Create functions for**:
1. `roll_dice(num_dice, num_sides)` - simulate dice rolls
2. `shuffle_deck()` - create and shuffle card deck
3. `deal_cards(deck, num_cards)` - deal cards from deck
4. `calculate_blackjack_score(cards)` - blackjack hand value
5. `generate_lottery_numbers(count, max_number)` - lottery number generator
6. `rock_paper_scissors(player1, player2)` - determine winner

**Requirements**:
- Use appropriate randomization
- Handle invalid inputs gracefully
- Return structured results

---

## Debugging Exercises

### Debug Exercise 1: Function Errors
**Goal**: Fix common function mistakes

**Fix these functions**:
```python
# Problem 1: Missing return statement
def calculate_tip(bill, percentage):
    tip = bill * (percentage / 100)
    # Missing return!

# Problem 2: Incorrect parameter usage
def greet_user(name, greeting="Hello"):
    print(f"{name}, {greeting}!")  # Wrong order!

# Problem 3: Scope issue
def calculate_total():
    subtotal = 100
    tax = subtotal * 0.08
    return total  # 'total' is not defined!

# Problem 4: Mutable default argument
def add_to_list(item, my_list=[]):
    my_list.append(item)
    return my_list
```

### Debug Exercise 2: Logic Errors
**Goal**: Find and fix logical mistakes

```python
# Problem 1: Wrong calculation
def calculate_average(numbers):
    return sum(numbers) / len(numbers) + 1  # Why +1?

# Problem 2: Incorrect condition
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, n):  # Should be range(2, int(n**0.5) + 1)
        if n % i == 0:
            return False
    return True

# Problem 3: Off-by-one error
def get_grade(score):
    if score > 90:  # Should be >= 90
        return "A"
    elif score > 80:
        return "B"
    # ... etc
```

---

## Self-Assessment Checklist

After completing these exercises, you should be able to:

- [ ] Define functions with appropriate names and parameters
- [ ] Use return statements to send data back from functions
- [ ] Call functions with correct arguments
- [ ] Understand the difference between parameters and arguments
- [ ] Use default parameters effectively
- [ ] Handle variable scope (local vs global variables)
- [ ] Write functions that solve real-world problems
- [ ] Debug common function-related errors
- [ ] Document functions with docstrings

## Common Mistakes to Avoid

### 1. Forgetting Return Statements
```python
# Wrong
def add_numbers(a, b):
    result = a + b  # No return!

# Right
def add_numbers(a, b):
    result = a + b
    return result
```

### 2. Confusing Print vs Return
```python
# Wrong - prints but doesn't return
def calculate_area(length, width):
    area = length * width
    print(area)  # This prints but function returns None

# Right - returns the value
def calculate_area(length, width):
    area = length * width
    return area  # Now you can use the result
```

### 3. Modifying Global Variables Unnecessarily
```python
# Avoid when possible
total = 0
def add_to_total(value):
    global total
    total += value

# Better approach
def add_to_total(current_total, value):
    return current_total + value
```

## Git Reminder

Save your work:
1. Create `lesson-04-functions` folder in your repository
2. Save exercise solutions as `.py` files
3. Commit and push:
   ```bash
   git add .
   git commit -m "Complete Lesson 04: Functions"
   git push
   ```

## Next Lesson Preview

In Lesson 05, we'll learn about:
- **Lists**: Ordered collections for storing multiple items
- **Dictionaries**: Key-value pairs for structured data
- **Sets**: Collections of unique items
- **Tuples**: Immutable sequences
- **List comprehensions**: Elegant ways to create and modify lists
