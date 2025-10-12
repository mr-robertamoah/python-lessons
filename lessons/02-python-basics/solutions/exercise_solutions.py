# Lesson 02 Solutions: Python Basics - Variables and Data Types

print("=== GUIDED EXERCISE SOLUTIONS ===\n")

# Exercise 1: Variable Creation and Types
print("Exercise 1: Variable Creation and Types")
print("-" * 40)

name = "Alice Johnson"
age = 25
height = 5.8
is_employed = True

print(f"Name: {name} (type: {type(name).__name__})")
print(f"Age: {age} (type: {type(age).__name__})")
print(f"Height: {height} (type: {type(height).__name__})")
print(f"Employed: {is_employed} (type: {type(is_employed).__name__})")
print()

# Exercise 2: Type Conversion Practice
print("Exercise 2: Type Conversion Practice")
print("-" * 40)

age_str = "25"
height_str = "5.8"
score_str = "95"

age_num = int(age_str)
height_num = float(height_str)
score_num = int(score_str)

next_year = age_num + 1
height_inches = height_num * 12
average_score = score_num / 100

print(f"Age next year: {next_year}")
print(f"Height in inches: {height_inches}")
print(f"Score as percentage: {average_score:.2%}")
print()

# Exercise 3: User Input and Formatting (simulated)
print("Exercise 3: User Input and Formatting")
print("-" * 40)

# Simulated user input
name = "Bob Smith"
favorite_number = 42
height = 6.1

print(f"Hello {name}!")
print(f"Your favorite number {favorite_number} is interesting.")
print(f"At {height} feet tall, you're quite tall!")
print()

print("=== INDEPENDENT EXERCISE SOLUTIONS ===\n")

# Exercise 4: Personal Profile Creator
print("Exercise 4: Personal Profile Creator")
print("-" * 40)

first_name = "Sarah"
last_name = "Wilson"
age = 28
birth_year = 2024 - age
height_feet = 5.6
height_inches = height_feet * 12
favorite_color = "Blue"
has_pets = True

print("=== Personal Profile ===")
print(f"Name: {first_name} {last_name}")
print(f"Age: {age} (born in {birth_year})")
print(f"Height: {height_feet} feet ({height_inches:.1f} inches)")
print(f"Favorite Color: {favorite_color}")
print(f"Has Pets: {'Yes' if has_pets else 'No'}")
print()

# Exercise 5: Simple Calculator
print("Exercise 5: Simple Calculator")
print("-" * 40)

num1 = 15.5
num2 = 4.2

print("=== Simple Calculator ===")
print(f"First number: {num1}")
print(f"Second number: {num2}")
print()
print("Results:")
print(f"{num1} + {num2} = {num1 + num2:.1f}")
print(f"{num1} - {num2} = {num1 - num2:.1f}")
print(f"{num1} * {num2} = {num1 * num2:.2f}")
print(f"{num1} / {num2} = {num1 / num2:.2f}")
print()

# Exercise 6: Unit Converter
print("Exercise 6: Unit Converter")
print("-" * 40)

# Temperature conversion
fahrenheit = 98.6
celsius = (fahrenheit - 32) * 5/9
print(f"Temperature: {fahrenheit}°F = {celsius:.1f}°C")

# Distance conversion
miles = 10
kilometers = miles * 1.60934
print(f"Distance: {miles} miles = {kilometers:.2f} km")

# Weight conversion
pounds = 150
kilograms = pounds * 0.453592
print(f"Weight: {pounds} lbs = {kilograms:.1f} kg")
print()

# Exercise 7: Shopping Cart Calculator
print("Exercise 7: Shopping Cart Calculator")
print("-" * 40)

item1_name, item1_price = "Shirt", 25.99
item2_name, item2_price = "Pants", 45.50
item3_name, item3_price = "Shoes", 89.99
tax_rate = 8.5

subtotal = item1_price + item2_price + item3_price
tax_amount = subtotal * (tax_rate / 100)
total = subtotal + tax_amount

print("=== Shopping Receipt ===")
print(f"Item 1: {item1_name} - ${item1_price:.2f}")
print(f"Item 2: {item2_name} - ${item2_price:.2f}")
print(f"Item 3: {item3_name} - ${item3_price:.2f}")
print()
print(f"Subtotal: ${subtotal:.2f}")
print(f"Tax ({tax_rate}%): ${tax_amount:.2f}")
print(f"Total: ${total:.2f}")
print()

print("=== CHALLENGE EXERCISE SOLUTIONS ===\n")

# Challenge 1: BMI Calculator
print("Challenge 1: BMI Calculator")
print("-" * 40)

weight_lbs = 150
height_ft = 5
height_in = 8

# Convert to metric
weight_kg = weight_lbs * 0.453592
height_total_in = (height_ft * 12) + height_in
height_m = height_total_in * 0.0254

# Calculate BMI
bmi = weight_kg / (height_m ** 2)

# Determine category
if bmi < 18.5:
    category = "Underweight"
elif bmi < 25:
    category = "Normal"
elif bmi < 30:
    category = "Overweight"
else:
    category = "Obese"

print(f"Weight: {weight_lbs} lbs ({weight_kg:.1f} kg)")
print(f"Height: {height_ft}'{height_in}\" ({height_m:.2f} m)")
print(f"BMI: {bmi:.1f} ({category})")
print()

# Challenge 2: Time Zone Converter
print("Challenge 2: Time Zone Converter")
print("-" * 40)

current_time = 15  # 3 PM
time_difference = 8

new_time = (current_time + time_difference) % 24

print(f"Current time: {current_time}:00 ({current_time % 12 or 12} {'PM' if current_time >= 12 else 'AM'})")
print(f"Time zone difference: +{time_difference} hours")
print(f"Time in other zone: {new_time}:00 ({new_time % 12 or 12} {'PM' if new_time >= 12 else 'AM'})")
print()

# Challenge 3: GPA Calculator
print("Challenge 3: GPA Calculator")
print("-" * 40)

# Sample grades and credit hours
subjects = ["Math", "English", "Science", "History", "Art"]
grades = [92, 87, 95, 78, 85]
credits = [4, 3, 4, 3, 2]

def grade_to_gpa(grade):
    if grade >= 90: return 4.0
    elif grade >= 80: return 3.0
    elif grade >= 70: return 2.0
    elif grade >= 60: return 1.0
    else: return 0.0

total_points = 0
total_credits = sum(credits)

print("=== Grade Breakdown ===")
for i, subject in enumerate(subjects):
    gpa_points = grade_to_gpa(grades[i])
    weighted_points = gpa_points * credits[i]
    total_points += weighted_points
    
    letter = "A" if grades[i] >= 90 else "B" if grades[i] >= 80 else "C" if grades[i] >= 70 else "D" if grades[i] >= 60 else "F"
    print(f"{subject}: {grades[i]}% ({letter}) - {credits[i]} credits - {gpa_points} GPA")

overall_gpa = total_points / total_credits
print(f"\nOverall GPA: {overall_gpa:.2f}")
print()

print("=== DEBUG EXERCISE SOLUTIONS ===\n")

print("Debug Exercise 1: Fixed Code")
print("-" * 40)

# Fixed Code 1
age = int(input("Enter age: "))  # Convert to int
next_year = age + 1
print("Next year:", next_year)

# Fixed Code 2
score = 95
message = "Your score is " + str(score)  # Convert to string
# Or better: message = f"Your score is {score}"

# Fixed Code 3
first_name = "Alice"  # Use underscore, not hyphen
print(first_name)

# Fixed Code 4
subject_class = "Math"  # Don't use reserved word 'class'
print("Subject:", subject_class)

print("\nDebug Exercise 2: Fixed Logic")
print("-" * 40)

# Fixed Code 1 - Temperature conversion
celsius = 25
fahrenheit = celsius * 9/5 + 32  # Add 32, don't subtract

# Fixed Code 2 - Percentage calculation
score = 85
total = 100
percentage = (score / total) * 100  # Multiply by 100 for percentage

# Fixed Code 3 - Area calculation
length = 10
width = 5
area = length * width  # Multiply, don't add

print(f"Temperature: {celsius}°C = {fahrenheit}°F")
print(f"Percentage: {percentage}%")
print(f"Area: {area} square units")
