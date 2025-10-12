"""
Lesson 02 Example 1: Variables and Data Types Demo
Demonstrates basic variable usage and data types
"""

print("=== Variables and Data Types Demo ===\n")

# Integer variables
age = 25
year = 2024
score = 100

print("Integer Examples:")
print(f"Age: {age} (type: {type(age)})")
print(f"Year: {year} (type: {type(year)})")
print(f"Score: {score} (type: {type(score)})")
print()

# Float variables
height = 5.8
price = 12.99
temperature = 98.6

print("Float Examples:")
print(f"Height: {height} feet (type: {type(height)})")
print(f"Price: ${price} (type: {type(price)})")
print(f"Temperature: {temperature}°F (type: {type(temperature)})")
print()

# String variables
name = "Alice Johnson"
city = "Seattle"
message = 'Hello, World!'

print("String Examples:")
print(f"Name: {name} (type: {type(name)})")
print(f"City: {city} (type: {type(city)})")
print(f"Message: {message} (type: {type(message)})")
print()

# Boolean variables
is_student = True
has_license = False
is_sunny = True

print("Boolean Examples:")
print(f"Is student: {is_student} (type: {type(is_student)})")
print(f"Has license: {has_license} (type: {type(has_license)})")
print(f"Is sunny: {is_sunny} (type: {type(is_sunny)})")
print()

# Variable operations
print("=== Variable Operations ===")
full_name = name  # Copy variable
birth_year = year - age  # Calculate using variables
greeting = f"Hello, {name}!"  # String formatting

print(f"Full name: {full_name}")
print(f"Birth year: {birth_year}")
print(f"Greeting: {greeting}")
