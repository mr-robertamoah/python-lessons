"""
Lesson 01 Example 2: Python as a Calculator
Demonstrates basic mathematical operations
"""

print("=== Python Calculator Examples ===")

# Basic arithmetic operations
print("Addition: 5 + 3 =", 5 + 3)
print("Subtraction: 10 - 4 =", 10 - 4)
print("Multiplication: 6 * 7 =", 6 * 7)
print("Division: 15 / 3 =", 15 / 3)
print("Exponentiation: 2 ** 3 =", 2 ** 3)
print("Modulus (remainder): 17 % 5 =", 17 % 5)

print("\n=== Real-world Calculations ===")

# Age in days (approximate)
age_years = 25
age_days = age_years * 365
print(f"If you're {age_years} years old, you've lived approximately {age_days} days")

# Compound interest calculation
principal = 1000
rate = 0.05
years = 10
final_amount = principal * (1 + rate) ** years
print(f"${principal} invested at {rate*100}% for {years} years becomes ${final_amount:.2f}")

# Temperature conversion
celsius = 25
fahrenheit = (celsius * 9/5) + 32
print(f"{celsius}°C is equal to {fahrenheit}°F")
