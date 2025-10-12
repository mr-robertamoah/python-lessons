"""
Lesson 03 Example 1: Conditional Statements Demo
Demonstrates if, elif, else statements with practical examples
"""

print("=== Conditional Statements Demo ===\n")

# Example 1: Simple if statement
temperature = 75
print(f"Temperature: {temperature}°F")

if temperature > 70:
    print("It's warm today!")
    print("Perfect weather for a walk.")
print()

# Example 2: if-else statement
age = 17
print(f"Age: {age}")

if age >= 18:
    print("You can vote!")
else:
    print("You're not old enough to vote yet.")
print()

# Example 3: if-elif-else chain
score = 85
print(f"Test Score: {score}")

if score >= 90:
    grade = "A"
    message = "Excellent!"
elif score >= 80:
    grade = "B"
    message = "Good job!"
elif score >= 70:
    grade = "C"
    message = "Satisfactory"
elif score >= 60:
    grade = "D"
    message = "Needs improvement"
else:
    grade = "F"
    message = "Please see instructor"

print(f"Grade: {grade} - {message}")
print()

# Example 4: Logical operators
age = 25
has_license = True
print(f"Age: {age}, Has License: {has_license}")

if age >= 18 and has_license:
    print("You can drive!")
elif age >= 18 and not has_license:
    print("You need to get a license first.")
else:
    print("You're too young to drive.")
print()

# Example 5: String comparisons
day = "Saturday"
print(f"Today is: {day}")

if day == "Saturday" or day == "Sunday":
    print("It's the weekend! Time to relax.")
else:
    print("It's a weekday. Time to work.")
print()

# Example 6: Complex conditions
income = 50000
credit_score = 720
employment_years = 3

print(f"Income: ${income}")
print(f"Credit Score: {credit_score}")
print(f"Employment Years: {employment_years}")

if (income >= 40000 and credit_score >= 700) or employment_years >= 5:
    print("Loan approved!")
else:
    print("Loan application needs review.")
