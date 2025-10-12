"""
Lesson 02 Example 2: User Input and Type Conversion
Demonstrates getting input from users and converting types
"""

print("=== Personal Information Collector ===")

# Get user information
first_name = input("Enter your first name: ")
last_name = input("Enter your last name: ")
age = int(input("Enter your age: "))
height = float(input("Enter your height in feet: "))
is_student = input("Are you a student? (yes/no): ").lower() == "yes"

# Calculate additional information
birth_year = 2024 - age
height_inches = height * 12

# Display formatted results
print(f"\n=== Your Information ===")
print(f"Full Name: {first_name} {last_name}")
print(f"Age: {age} years old")
print(f"Birth Year: {birth_year}")
print(f"Height: {height} feet ({height_inches:.1f} inches)")
print(f"Student Status: {is_student}")

# Demonstrate type conversion
print(f"\n=== Type Information ===")
print(f"first_name type: {type(first_name)}")
print(f"age type: {type(age)}")
print(f"height type: {type(height)}")
print(f"is_student type: {type(is_student)}")

# String formatting examples
print(f"\n=== Formatting Examples ===")
print(f"In 5 years, {first_name} will be {age + 5} years old.")
print(f"Height in centimeters: {height * 30.48:.1f} cm")
print(f"Student status: {'Yes' if is_student else 'No'}")
