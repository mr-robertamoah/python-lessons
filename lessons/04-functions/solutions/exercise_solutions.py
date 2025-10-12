"""
Lesson 04 Solutions: Functions Exercises
Complete solutions for all exercises
"""

print("=== LESSON 04 EXERCISE SOLUTIONS ===\n")

# ============================================================================
# GUIDED EXERCISES
# ============================================================================

print("GUIDED EXERCISE 1: Basic Function Creation")
print("=" * 50)

def say_hello():
    """Print hello message"""
    print("Hello, World!")

def introduce(name, age):
    """Print introduction with name and age"""
    print(f"Hi, I'm {name} and I'm {age} years old.")

def add_two_numbers(a, b):
    """Return sum of two numbers"""
    return a + b

def is_even(number):
    """Return True if number is even, False otherwise"""
    return number % 2 == 0

# Test the functions
print("Testing basic functions:")
say_hello()
introduce("Alice", 25)
result = add_two_numbers(10, 15)
print(f"Sum: {result}")
print(f"Is 8 even? {is_even(8)}")
print(f"Is 7 even? {is_even(7)}")
print()

print("GUIDED EXERCISE 2: Calculator Functions")
print("=" * 50)

def add(a, b):
    """Addition"""
    return a + b

def subtract(a, b):
    """Subtraction"""
    return a - b

def multiply(a, b):
    """Multiplication"""
    return a * b

def divide(a, b):
    """Division with zero check"""
    if b == 0:
        return "Error: Division by zero!"
    return a / b

def power(base, exponent):
    """Exponentiation"""
    return base ** exponent

def simple_calculator():
    """Interactive calculator using the functions"""
    try:
        num1 = float(input("Enter first number: "))
        num2 = float(input("Enter second number: "))
        operation = input("Enter operation (+, -, *, /, **): ")
        
        if operation == '+':
            result = add(num1, num2)
        elif operation == '-':
            result = subtract(num1, num2)
        elif operation == '*':
            result = multiply(num1, num2)
        elif operation == '/':
            result = divide(num1, num2)
        elif operation == '**':
            result = power(num1, num2)
        else:
            result = "Invalid operation!"
        
        print(f"Result: {result}")
    except ValueError:
        print("Please enter valid numbers!")

# Test calculator functions
print("Testing calculator functions:")
print(f"10 + 5 = {add(10, 5)}")
print(f"10 - 5 = {subtract(10, 5)}")
print(f"10 * 5 = {multiply(10, 5)}")
print(f"10 / 5 = {divide(10, 5)}")
print(f"10 / 0 = {divide(10, 0)}")
print(f"2 ** 3 = {power(2, 3)}")
print()

print("GUIDED EXERCISE 3: String Processing Functions")
print("=" * 50)

def count_vowels(text):
    """Count vowels in a string"""
    vowels = "aeiouAEIOU"
    return sum(1 for char in text if char in vowels)

def reverse_string(text):
    """Return reversed string"""
    return text[::-1]

def is_palindrome(text):
    """Check if text reads same forwards and backwards"""
    cleaned = ''.join(char.lower() for char in text if char.isalnum())
    return cleaned == cleaned[::-1]

def word_count(text):
    """Count words in text"""
    return len(text.split())

# Test string functions
print("Testing string functions:")
test_text = "Hello World"
print(f"Text: '{test_text}'")
print(f"Vowels: {count_vowels(test_text)}")
print(f"Reversed: '{reverse_string(test_text)}'")
print(f"Is palindrome: {is_palindrome(test_text)}")
print(f"Word count: {word_count(test_text)}")

print(f"\nTesting palindrome with 'racecar': {is_palindrome('racecar')}")
print(f"Testing palindrome with 'A man a plan a canal Panama': {is_palindrome('A man a plan a canal Panama')}")
print()

# ============================================================================
# INDEPENDENT EXERCISES
# ============================================================================

print("INDEPENDENT EXERCISE 4: Temperature Conversion Suite")
print("=" * 50)

def celsius_to_fahrenheit(c):
    """Convert Celsius to Fahrenheit"""
    return (c * 9/5) + 32

def fahrenheit_to_celsius(f):
    """Convert Fahrenheit to Celsius"""
    return (f - 32) * 5/9

def celsius_to_kelvin(c):
    """Convert Celsius to Kelvin"""
    return c + 273.15

def kelvin_to_celsius(k):
    """Convert Kelvin to Celsius"""
    return k - 273.15

def fahrenheit_to_kelvin(f):
    """Convert Fahrenheit to Kelvin"""
    return celsius_to_kelvin(fahrenheit_to_celsius(f))

def kelvin_to_fahrenheit(k):
    """Convert Kelvin to Fahrenheit"""
    return celsius_to_fahrenheit(kelvin_to_celsius(k))

def format_temperature(temp, unit):
    """Format temperature with unit"""
    return f"{temp:.1f}°{unit}"

def temperature_converter():
    """Interactive temperature converter menu"""
    print("\n=== Temperature Converter ===")
    print("1. Celsius to Fahrenheit")
    print("2. Fahrenheit to Celsius")
    print("3. Celsius to Kelvin")
    print("4. Kelvin to Celsius")
    print("5. Fahrenheit to Kelvin")
    print("6. Kelvin to Fahrenheit")
    
    try:
        choice = int(input("Choose conversion (1-6): "))
        temp = float(input("Enter temperature: "))
        
        conversions = {
            1: (celsius_to_fahrenheit(temp), 'F'),
            2: (fahrenheit_to_celsius(temp), 'C'),
            3: (celsius_to_kelvin(temp), 'K'),
            4: (kelvin_to_celsius(temp), 'C'),
            5: (fahrenheit_to_kelvin(temp), 'K'),
            6: (kelvin_to_fahrenheit(temp), 'F')
        }
        
        if choice in conversions:
            result, unit = conversions[choice]
            print(f"Result: {format_temperature(result, unit)}")
        else:
            print("Invalid choice!")
    
    except ValueError:
        print("Please enter valid numbers!")

# Test temperature conversions
print("Testing temperature conversions:")
print(f"25°C = {format_temperature(celsius_to_fahrenheit(25), 'F')}")
print(f"77°F = {format_temperature(fahrenheit_to_celsius(77), 'C')}")
print(f"0°C = {format_temperature(celsius_to_kelvin(0), 'K')}")
print()

print("INDEPENDENT EXERCISE 5: Grade Analysis System")
print("=" * 50)

def letter_grade(percentage):
    """Convert percentage to letter grade"""
    if percentage >= 97:
        return 'A+'
    elif percentage >= 93:
        return 'A'
    elif percentage >= 90:
        return 'A-'
    elif percentage >= 87:
        return 'B+'
    elif percentage >= 83:
        return 'B'
    elif percentage >= 80:
        return 'B-'
    elif percentage >= 77:
        return 'C+'
    elif percentage >= 73:
        return 'C'
    elif percentage >= 70:
        return 'C-'
    elif percentage >= 60:
        return 'D'
    else:
        return 'F'

def gpa_points(letter):
    """Convert letter grade to GPA points"""
    grade_points = {
        'A+': 4.0, 'A': 4.0, 'A-': 3.7,
        'B+': 3.3, 'B': 3.0, 'B-': 2.7,
        'C+': 2.3, 'C': 2.0, 'C-': 1.7,
        'D': 1.0, 'F': 0.0
    }
    return grade_points.get(letter, 0.0)

def calculate_average(scores):
    """Calculate average of score list"""
    if not scores:
        return 0
    return sum(scores) / len(scores)

def find_highest_lowest(scores):
    """Return highest and lowest scores"""
    if not scores:
        return None, None
    return max(scores), min(scores)

def grade_distribution(scores):
    """Count A's, B's, C's, etc."""
    distribution = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0}
    
    for score in scores:
        letter = letter_grade(score)
        grade_category = letter[0]  # Get first character (A, B, C, D, F)
        distribution[grade_category] += 1
    
    return distribution

def analyze_student_performance(scores):
    """Comprehensive analysis of student scores"""
    if not scores:
        return "No scores provided"
    
    avg = calculate_average(scores)
    highest, lowest = find_highest_lowest(scores)
    distribution = grade_distribution(scores)
    
    return {
        'count': len(scores),
        'average': round(avg, 2),
        'average_letter': letter_grade(avg),
        'highest': highest,
        'lowest': lowest,
        'range': highest - lowest,
        'distribution': distribution
    }

# Test grade analysis
print("Testing grade analysis system:")
student_scores = [85, 92, 78, 96, 88, 73, 91, 87]
analysis = analyze_student_performance(student_scores)

print(f"Scores: {student_scores}")
print("Analysis Results:")
for key, value in analysis.items():
    print(f"  {key.title()}: {value}")
print()

print("INDEPENDENT EXERCISE 6: Personal Finance Calculator")
print("=" * 50)

def simple_interest(principal, rate, time):
    """Calculate simple interest"""
    return principal * rate * time

def compound_interest(principal, rate, time, compounds_per_year=1):
    """Calculate compound interest"""
    return principal * (1 + rate/compounds_per_year) ** (compounds_per_year * time)

def monthly_payment(loan_amount, annual_rate, years):
    """Calculate monthly loan payment"""
    monthly_rate = annual_rate / 12
    num_payments = years * 12
    
    if monthly_rate == 0:
        return loan_amount / num_payments
    
    return loan_amount * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)

def savings_goal(target_amount, monthly_deposit, annual_rate):
    """Calculate time to reach savings goal"""
    if monthly_deposit <= 0:
        return float('inf')
    
    monthly_rate = annual_rate / 12
    if monthly_rate == 0:
        return target_amount / monthly_deposit
    
    # Formula for future value of annuity
    months = -1 * (1/12) * (annual_rate/100) * (target_amount / monthly_deposit - 1) / monthly_rate
    return months

def budget_analyzer(income, expenses_dict):
    """Analyze budget and return summary"""
    total_expenses = sum(expenses_dict.values())
    remaining = income - total_expenses
    savings_rate = (remaining / income) * 100 if income > 0 else 0
    
    return {
        'income': income,
        'total_expenses': total_expenses,
        'remaining': remaining,
        'savings_rate': round(savings_rate, 1),
        'largest_expense': max(expenses_dict, key=expenses_dict.get),
        'expense_breakdown': expenses_dict
    }

# Test financial functions
print("Testing financial calculations:")

# Simple vs compound interest
principal = 1000
rate = 0.05
years = 5

simple = simple_interest(principal, rate, years)
compound = compound_interest(principal, rate, years)

print(f"${principal} at {rate*100}% for {years} years:")
print(f"  Simple interest: ${simple:.2f}")
print(f"  Compound interest: ${compound:.2f}")
print(f"  Difference: ${compound - simple:.2f}")

# Loan payment
loan_payment = monthly_payment(200000, 0.04, 30)
print(f"\nMonthly payment for $200,000 loan at 4% for 30 years: ${loan_payment:.2f}")

# Budget analysis
monthly_income = 5000
expenses = {
    'rent': 1500,
    'food': 600,
    'transportation': 300,
    'entertainment': 200,
    'utilities': 150
}

budget_analysis = budget_analyzer(monthly_income, expenses)
print(f"\nBudget Analysis:")
for key, value in budget_analysis.items():
    if key != 'expense_breakdown':
        print(f"  {key.replace('_', ' ').title()}: {value}")
print()

print("=== CHALLENGE EXERCISES ===")
print()

print("CHALLENGE 1: Statistical Analysis Functions")
print("=" * 40)

def calculate_mean(data):
    """Calculate arithmetic mean"""
    return sum(data) / len(data) if data else 0

def calculate_median(data):
    """Calculate median value"""
    if not data:
        return 0
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n % 2 == 0:
        return (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
    return sorted_data[n//2]

def calculate_mode(data):
    """Calculate most frequent value"""
    if not data:
        return None
    frequency = {}
    for value in data:
        frequency[value] = frequency.get(value, 0) + 1
    return max(frequency, key=frequency.get)

def calculate_standard_deviation(data):
    """Calculate standard deviation"""
    if len(data) < 2:
        return 0
    mean = calculate_mean(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    return variance ** 0.5

def remove_outliers(data, method='iqr'):
    """Remove outliers using IQR method"""
    if len(data) < 4:
        return data
    
    sorted_data = sorted(data)
    n = len(sorted_data)
    q1 = sorted_data[n//4]
    q3 = sorted_data[3*n//4]
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    return [x for x in data if lower_bound <= x <= upper_bound]

# Test statistical functions
test_data = [1, 2, 3, 4, 5, 100]  # 100 is an outlier
print(f"Original data: {test_data}")
print(f"Mean: {calculate_mean(test_data):.2f}")
print(f"Median: {calculate_median(test_data):.2f}")
print(f"Mode: {calculate_mode(test_data)}")
print(f"Standard deviation: {calculate_standard_deviation(test_data):.2f}")

cleaned_data = remove_outliers(test_data)
print(f"After removing outliers: {cleaned_data}")
print(f"New mean: {calculate_mean(cleaned_data):.2f}")
print()

print("=== END OF SOLUTIONS ===")
