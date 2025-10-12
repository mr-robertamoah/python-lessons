"""
Lesson 04 Example 1: Basic Functions Demo
Demonstrates function definition, parameters, and return values
"""

print("=== Basic Functions Demo ===\n")

# Example 1: Simple function with no parameters
def greet():
    """Simple greeting function"""
    print("Hello from the function!")

print("1. Function with no parameters:")
greet()
greet()  # Can call multiple times
print()

# Example 2: Function with parameters
def greet_person(name, greeting="Hello"):
    """Greet a person with customizable greeting"""
    return f"{greeting}, {name}!"

print("2. Function with parameters:")
message1 = greet_person("Alice")
message2 = greet_person("Bob", "Hi")
print(message1)  # Hello, Alice!
print(message2)  # Hi, Bob!
print()

# Example 3: Function with calculations
def calculate_rectangle_area(length, width):
    """Calculate area of a rectangle"""
    area = length * width
    return area

def calculate_circle_area(radius):
    """Calculate area of a circle"""
    pi = 3.14159
    area = pi * radius ** 2
    return round(area, 2)

print("3. Mathematical functions:")
rect_area = calculate_rectangle_area(10, 5)
circle_area = calculate_circle_area(3)
print(f"Rectangle area (10x5): {rect_area}")
print(f"Circle area (radius 3): {circle_area}")
print()

# Example 4: Function with multiple return values
def analyze_number(num):
    """Analyze properties of a number"""
    is_positive = num > 0
    is_even = num % 2 == 0
    absolute_value = abs(num)
    return is_positive, is_even, absolute_value

print("4. Multiple return values:")
positive, even, abs_val = analyze_number(-8)
print(f"Number: -8")
print(f"Positive: {positive}, Even: {even}, Absolute: {abs_val}")
print()

# Example 5: Function with variable arguments
def calculate_average(*numbers):
    """Calculate average of any number of values"""
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)

print("5. Variable arguments:")
avg1 = calculate_average(10, 20, 30)
avg2 = calculate_average(5, 15, 25, 35, 45)
print(f"Average of (10, 20, 30): {avg1}")
print(f"Average of (5, 15, 25, 35, 45): {avg2}")
print()

# Example 6: Function scope demonstration
global_var = "I'm global"

def scope_demo():
    """Demonstrate variable scope"""
    local_var = "I'm local"
    print(f"Inside function - Global: {global_var}")
    print(f"Inside function - Local: {local_var}")

print("6. Variable scope:")
scope_demo()
print(f"Outside function - Global: {global_var}")
# print(f"Outside function - Local: {local_var}")  # This would cause an error
