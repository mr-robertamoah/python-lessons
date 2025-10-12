"""
Lesson 04 Example 2: Practical Functions
Real-world examples of function usage in data analysis
"""

print("=== Practical Functions Demo ===\n")

# Example 1: Temperature conversion functions
def celsius_to_fahrenheit(celsius):
    """Convert Celsius to Fahrenheit"""
    return (celsius * 9/5) + 32

def fahrenheit_to_celsius(fahrenheit):
    """Convert Fahrenheit to Celsius"""
    return (fahrenheit - 32) * 5/9

print("1. Temperature Conversions:")
temps_c = [0, 25, 37, 100]
for temp in temps_c:
    temp_f = celsius_to_fahrenheit(temp)
    print(f"{temp}°C = {temp_f:.1f}°F")
print()

# Example 2: Statistical functions
def calculate_mean(numbers):
    """Calculate arithmetic mean"""
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)

def calculate_median(numbers):
    """Calculate median value"""
    if not numbers:
        return 0
    sorted_nums = sorted(numbers)
    n = len(sorted_nums)
    if n % 2 == 0:
        return (sorted_nums[n//2 - 1] + sorted_nums[n//2]) / 2
    return sorted_nums[n//2]

def calculate_mode(numbers):
    """Find most frequent value"""
    if not numbers:
        return None
    frequency = {}
    for num in numbers:
        frequency[num] = frequency.get(num, 0) + 1
    return max(frequency, key=frequency.get)

print("2. Statistical Analysis:")
data = [1, 2, 2, 3, 4, 4, 4, 5]
print(f"Data: {data}")
print(f"Mean: {calculate_mean(data):.2f}")
print(f"Median: {calculate_median(data)}")
print(f"Mode: {calculate_mode(data)}")
print()

# Example 3: Data validation functions
def validate_email(email):
    """Basic email validation"""
    return "@" in email and "." in email and len(email) > 5

def validate_age(age):
    """Validate age is reasonable"""
    try:
        age_num = int(age)
        return 0 <= age_num <= 150
    except ValueError:
        return False

def validate_phone(phone):
    """Basic phone number validation"""
    digits = ''.join(filter(str.isdigit, phone))
    return len(digits) == 10

print("3. Data Validation:")
test_data = [
    ("alice@email.com", 25, "555-123-4567"),
    ("invalid-email", 200, "123"),
    ("bob@test.org", 30, "5551234567")
]

for email, age, phone in test_data:
    print(f"Email: {email} - Valid: {validate_email(email)}")
    print(f"Age: {age} - Valid: {validate_age(age)}")
    print(f"Phone: {phone} - Valid: {validate_phone(phone)}")
    print()

# Example 4: Financial calculations
def calculate_compound_interest(principal, rate, time, compounds_per_year=1):
    """Calculate compound interest"""
    return principal * (1 + rate/compounds_per_year) ** (compounds_per_year * time)

def calculate_monthly_payment(loan_amount, annual_rate, years):
    """Calculate monthly loan payment"""
    monthly_rate = annual_rate / 12
    num_payments = years * 12
    if monthly_rate == 0:
        return loan_amount / num_payments
    return loan_amount * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)

print("4. Financial Calculations:")
principal = 10000
rate = 0.05
years = 10

final_amount = calculate_compound_interest(principal, rate, years)
print(f"${principal} at {rate*100}% for {years} years = ${final_amount:.2f}")

loan_payment = calculate_monthly_payment(200000, 0.04, 30)
print(f"Monthly payment for $200,000 loan at 4% for 30 years: ${loan_payment:.2f}")
print()

# Example 5: Text processing functions
def word_count(text):
    """Count words in text"""
    return len(text.split())

def character_frequency(text):
    """Count frequency of each character"""
    freq = {}
    for char in text.lower():
        if char.isalpha():
            freq[char] = freq.get(char, 0) + 1
    return freq

def extract_numbers(text):
    """Extract all numbers from text"""
    import re
    return [float(match) for match in re.findall(r'-?\d+\.?\d*', text)]

print("5. Text Processing:")
sample_text = "The price is $25.99 and the discount is 15%"
print(f"Text: {sample_text}")
print(f"Word count: {word_count(sample_text)}")
print(f"Numbers found: {extract_numbers(sample_text)}")
char_freq = character_frequency(sample_text)
print(f"Most common letter: {max(char_freq, key=char_freq.get)}")
print()

print("=== Functions working together ===")

def analyze_dataset(data, name="Dataset"):
    """Comprehensive dataset analysis using multiple functions"""
    if not data:
        return f"{name}: No data provided"
    
    stats = {
        'name': name,
        'count': len(data),
        'mean': calculate_mean(data),
        'median': calculate_median(data),
        'mode': calculate_mode(data),
        'min': min(data),
        'max': max(data),
        'range': max(data) - min(data)
    }
    
    return stats

# Demonstrate function composition
sales_data = [1200, 1500, 1100, 1800, 1300, 1600, 1400, 1700, 1250, 1550]
analysis = analyze_dataset(sales_data, "Monthly Sales")

print("Dataset Analysis Results:")
for key, value in analysis.items():
    if isinstance(value, float):
        print(f"{key.title()}: {value:.2f}")
    else:
        print(f"{key.title()}: {value}")
