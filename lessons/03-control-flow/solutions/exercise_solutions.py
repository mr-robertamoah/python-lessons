"""
Lesson 03 Solutions: Control Flow Exercises
Complete solutions for all exercises
"""

print("=== LESSON 03 EXERCISE SOLUTIONS ===\n")

# ============================================================================
# GUIDED EXERCISES
# ============================================================================

print("GUIDED EXERCISE 1: Age Category Classifier")
print("=" * 50)

def classify_age(age):
    if age <= 12:
        return "Child", "Enjoy your childhood!"
    elif age <= 19:
        return "Teenager", "Great time to learn and explore!"
    elif age <= 64:
        return "Adult", "Make the most of your productive years!"
    else:
        return "Senior", "Wisdom comes with experience!"

# Test the function
test_ages = [8, 16, 35, 70]
for age in test_ages:
    category, message = classify_age(age)
    print(f"Age {age}: {category} - {message}")
print()

print("GUIDED EXERCISE 2: Simple Calculator with Conditions")
print("=" * 50)

def calculator(num1, num2, operation):
    if operation == '+':
        return num1 + num2
    elif operation == '-':
        return num1 - num2
    elif operation == '*':
        return num1 * num2
    elif operation == '/':
        if num2 == 0:
            return "Error: Division by zero!"
        return num1 / num2
    else:
        return "Error: Invalid operation!"

# Test the calculator
tests = [(10, 5, '+'), (10, 5, '-'), (10, 5, '*'), (10, 5, '/'), (10, 0, '/')]
for n1, n2, op in tests:
    result = calculator(n1, n2, op)
    print(f"{n1} {op} {n2} = {result}")
print()

print("GUIDED EXERCISE 3: Number Analysis Loop")
print("=" * 50)

even_count = odd_count = div_by_3 = mult_of_5 = 0

for num in range(1, 21):
    conditions = []
    
    if num % 2 == 0:
        conditions.append("Even")
        even_count += 1
    else:
        conditions.append("Odd")
        odd_count += 1
    
    if num % 3 == 0:
        conditions.append("Divisible by 3")
        div_by_3 += 1
    
    if num % 5 == 0:
        conditions.append("Multiple of 5")
        mult_of_5 += 1
    
    print(f"Number {num}: {', '.join(conditions)}")

print(f"\nSummary: {even_count} even, {odd_count} odd, {div_by_3} divisible by 3, {mult_of_5} multiples of 5")
print()

# ============================================================================
# INDEPENDENT EXERCISES
# ============================================================================

print("INDEPENDENT EXERCISE 4: Password Strength Checker")
print("=" * 50)

def check_password_strength(password):
    score = 0
    feedback = []
    
    if len(password) >= 8:
        score += 1
        feedback.append("✓ Length requirement met")
    else:
        feedback.append("✗ Password too short (min 8 characters)")
    
    if any(c.isupper() for c in password):
        score += 1
        feedback.append("✓ Contains uppercase")
    else:
        feedback.append("✗ Missing uppercase letter")
    
    if any(c.islower() for c in password):
        score += 1
        feedback.append("✓ Contains lowercase")
    else:
        feedback.append("✗ Missing lowercase letter")
    
    if any(c.isdigit() for c in password):
        score += 1
        feedback.append("✓ Contains number")
    else:
        feedback.append("✗ Missing number")
    
    if any(c in "!@#$%^&*" for c in password):
        score += 1
        feedback.append("✓ Contains special character")
    else:
        feedback.append("✗ Missing special character")
    
    strength_levels = {5: "Very Strong", 4: "Strong", 3: "Medium", 2: "Weak", 1: "Very Weak", 0: "Very Weak"}
    strength = strength_levels[score]
    
    return score, strength, feedback

# Test password checker
test_passwords = ["weak", "Better123", "MyPass123!", "abc", "STRONG123!"]
for pwd in test_passwords:
    score, strength, feedback = check_password_strength(pwd)
    print(f"\nPassword: {pwd}")
    for f in feedback:
        print(f"  {f}")
    print(f"  Strength: {strength} ({score}/5)")
print()

print("INDEPENDENT EXERCISE 5: Grade Statistics Calculator")
print("=" * 50)

def calculate_grade_statistics():
    num_students = int(input("How many students? "))
    grades = []
    
    for i in range(num_students):
        while True:
            try:
                grade = float(input(f"Enter grade for student {i+1}: "))
                if 0 <= grade <= 100:
                    grades.append(grade)
                    break
                else:
                    print("Grade must be between 0 and 100")
            except ValueError:
                print("Please enter a valid number")
    
    if not grades:
        return "No grades entered"
    
    # Calculate statistics
    average = sum(grades) / len(grades)
    highest = max(grades)
    lowest = min(grades)
    
    # Count letter grades
    grade_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0}
    for grade in grades:
        if grade >= 90:
            grade_counts['A'] += 1
        elif grade >= 80:
            grade_counts['B'] += 1
        elif grade >= 70:
            grade_counts['C'] += 1
        elif grade >= 60:
            grade_counts['D'] += 1
        else:
            grade_counts['F'] += 1
    
    pass_count = sum(1 for g in grades if g >= 60)
    pass_rate = (pass_count / len(grades)) * 100
    
    return {
        'average': round(average, 2),
        'highest': highest,
        'lowest': lowest,
        'grade_distribution': grade_counts,
        'pass_rate': round(pass_rate, 1)
    }

# Example with sample data
sample_grades = [85, 92, 78, 96, 88, 73, 91, 87, 65, 94]
print("Sample grade analysis:")
print(f"Grades: {sample_grades}")

average = sum(sample_grades) / len(sample_grades)
highest = max(sample_grades)
lowest = min(sample_grades)

grade_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0}
for grade in sample_grades:
    if grade >= 90:
        grade_counts['A'] += 1
    elif grade >= 80:
        grade_counts['B'] += 1
    elif grade >= 70:
        grade_counts['C'] += 1
    elif grade >= 60:
        grade_counts['D'] += 1
    else:
        grade_counts['F'] += 1

pass_count = sum(1 for g in sample_grades if g >= 60)
pass_rate = (pass_count / len(sample_grades)) * 100

print(f"Average: {average:.2f}")
print(f"Highest: {highest}")
print(f"Lowest: {lowest}")
print(f"Grade distribution: {grade_counts}")
print(f"Pass rate: {pass_rate:.1f}%")
print()

print("INDEPENDENT EXERCISE 6: Number Guessing Game")
print("=" * 50)

import random

def number_guessing_game():
    secret = random.randint(1, 100)
    attempts = 0
    max_attempts = 7
    
    print("I'm thinking of a number between 1 and 100!")
    
    while attempts < max_attempts:
        try:
            guess = int(input("Enter your guess: "))
            attempts += 1
            
            if guess == secret:
                print(f"Congratulations! You got it in {attempts} attempts!")
                return True
            elif guess < secret:
                print("Too low!")
            else:
                print("Too high!")
            
            remaining = max_attempts - attempts
            if remaining > 0:
                print(f"{remaining} attempts remaining")
        
        except ValueError:
            print("Please enter a valid number")
            attempts -= 1
    
    print(f"Game over! The number was {secret}")
    return False

# Simulate a game
print("Simulating number guessing game...")
secret_num = 42
guesses = [50, 25, 37, 43, 40, 41, 42]
for i, guess in enumerate(guesses, 1):
    print(f"Attempt {i}: Guess {guess}")
    if guess == secret_num:
        print(f"Correct! Found {secret_num} in {i} attempts!")
        break
    elif guess < secret_num:
        print("Too low!")
    else:
        print("Too high!")
print()

print("=== CHALLENGE EXERCISES ===")
print()

print("CHALLENGE 1: Prime Number Finder")
print("=" * 40)

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def find_primes(start, end):
    primes = []
    for num in range(start, end + 1):
        if is_prime(num):
            primes.append(num)
    return primes

# Find primes between 1 and 50
primes = find_primes(1, 50)
print(f"Prime numbers between 1 and 50: {primes}")
print(f"Count: {len(primes)} primes found")
print()

print("CHALLENGE 2: Text Analysis Tool")
print("=" * 40)

def analyze_text(text):
    # Count characters
    total_chars = len(text)
    chars_no_spaces = len(text.replace(' ', ''))
    
    # Count words
    words = text.split()
    word_count = len(words)
    
    # Count vowels and consonants
    vowels = "aeiouAEIOU"
    vowel_count = sum(1 for char in text if char in vowels)
    consonant_count = sum(1 for char in text if char.isalpha() and char not in vowels)
    
    # Find most frequent character
    char_freq = {}
    for char in text.lower():
        if char.isalpha():
            char_freq[char] = char_freq.get(char, 0) + 1
    
    most_frequent = max(char_freq, key=char_freq.get) if char_freq else None
    
    return {
        'total_chars': total_chars,
        'chars_no_spaces': chars_no_spaces,
        'words': word_count,
        'vowels': vowel_count,
        'consonants': consonant_count,
        'most_frequent': (most_frequent, char_freq.get(most_frequent, 0)) if most_frequent else None
    }

# Test text analysis
sample_text = "Hello World! This is a sample text for analysis."
analysis = analyze_text(sample_text)

print(f"Text: \"{sample_text}\"")
print(f"Total characters: {analysis['total_chars']}")
print(f"Characters (no spaces): {analysis['chars_no_spaces']}")
print(f"Words: {analysis['words']}")
print(f"Vowels: {analysis['vowels']}")
print(f"Consonants: {analysis['consonants']}")
if analysis['most_frequent']:
    char, count = analysis['most_frequent']
    print(f"Most frequent: '{char}' (appears {count} times)")
print()

print("=== END OF SOLUTIONS ===")
