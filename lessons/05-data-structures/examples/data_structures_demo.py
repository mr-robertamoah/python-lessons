"""
Lesson 05 Example 1: Data Structures Demo
Comprehensive demonstration of lists, dictionaries, sets, and tuples
"""

print("=== Data Structures Demo ===\n")

# ============================================================================
# LISTS - Ordered, mutable collections
# ============================================================================

print("1. LISTS - Ordered Collections")
print("=" * 40)

# Creating and manipulating lists
fruits = ["apple", "banana", "orange"]
numbers = [1, 2, 3, 4, 5]
mixed = ["Alice", 25, True, 3.14]

print(f"Fruits: {fruits}")
print(f"Numbers: {numbers}")
print(f"Mixed data: {mixed}")

# List operations
fruits.append("grape")
fruits.insert(1, "cherry")
print(f"After adding items: {fruits}")

# List slicing and indexing
print(f"First fruit: {fruits[0]}")
print(f"Last fruit: {fruits[-1]}")
print(f"First three: {fruits[:3]}")

# List methods
numbers.extend([6, 7, 8])
print(f"Extended numbers: {numbers}")
print(f"Sum of numbers: {sum(numbers)}")
print(f"Length: {len(numbers)}")
print()

# ============================================================================
# DICTIONARIES - Key-value pairs
# ============================================================================

print("2. DICTIONARIES - Key-Value Pairs")
print("=" * 40)

# Creating dictionaries
student = {
    "name": "Alice Johnson",
    "age": 20,
    "major": "Computer Science",
    "gpa": 3.8,
    "courses": ["Python", "Statistics", "Calculus"]
}

print("Student information:")
for key, value in student.items():
    print(f"  {key}: {value}")

# Dictionary operations
student["year"] = "Junior"
student["gpa"] = 3.9  # Update existing value
print(f"\nUpdated GPA: {student['gpa']}")

# Safe access with get()
print(f"Email: {student.get('email', 'Not provided')}")

# Dictionary methods
print(f"Keys: {list(student.keys())}")
print(f"Has 'name' key: {'name' in student}")
print()

# ============================================================================
# SETS - Unique collections
# ============================================================================

print("3. SETS - Unique Collections")
print("=" * 40)

# Creating sets
colors1 = {"red", "green", "blue"}
colors2 = {"blue", "yellow", "purple"}
numbers_set = set([1, 2, 3, 2, 1])  # Duplicates removed

print(f"Colors1: {colors1}")
print(f"Colors2: {colors2}")
print(f"Numbers (duplicates removed): {numbers_set}")

# Set operations
union = colors1 | colors2
intersection = colors1 & colors2
difference = colors1 - colors2

print(f"Union: {union}")
print(f"Intersection: {intersection}")
print(f"Difference (colors1 - colors2): {difference}")

# Set methods
colors1.add("orange")
colors1.discard("red")
print(f"Modified colors1: {colors1}")
print()

# ============================================================================
# TUPLES - Immutable sequences
# ============================================================================

print("4. TUPLES - Immutable Sequences")
print("=" * 40)

# Creating tuples
coordinates = (10, 20)
rgb_color = (255, 128, 0)
person_info = ("Bob", 30, "Engineer")

print(f"Coordinates: {coordinates}")
print(f"RGB Color: {rgb_color}")
print(f"Person: {person_info}")

# Tuple unpacking
x, y = coordinates
name, age, job = person_info

print(f"X: {x}, Y: {y}")
print(f"Name: {name}, Age: {age}, Job: {job}")

# Tuple methods
numbers_tuple = (1, 2, 3, 2, 1)
print(f"Count of 2: {numbers_tuple.count(2)}")
print(f"Index of 3: {numbers_tuple.index(3)}")
print()

# ============================================================================
# PRACTICAL EXAMPLES
# ============================================================================

print("5. PRACTICAL DATA ANALYSIS EXAMPLES")
print("=" * 40)

# Example 1: Student grades management
students_grades = {
    "Alice": [85, 92, 78, 96],
    "Bob": [88, 76, 91, 83],
    "Charlie": [92, 95, 89, 94]
}

print("Grade Analysis:")
for student, grades in students_grades.items():
    average = sum(grades) / len(grades)
    print(f"{student}: {grades} -> Average: {average:.1f}")

# Example 2: Inventory tracking with sets
current_inventory = {"apples", "bananas", "oranges", "grapes"}
sold_items = {"bananas", "oranges"}
new_arrivals = {"strawberries", "blueberries", "apples"}

remaining = current_inventory - sold_items
updated_inventory = remaining | new_arrivals

print(f"\nInventory Management:")
print(f"Current: {current_inventory}")
print(f"Sold: {sold_items}")
print(f"New arrivals: {new_arrivals}")
print(f"Updated inventory: {updated_inventory}")

# Example 3: Coordinate system with tuples
points = [(0, 0), (3, 4), (6, 8), (9, 12)]

print(f"\nDistance calculations:")
for i, point in enumerate(points):
    x, y = point
    distance = (x**2 + y**2)**0.5
    print(f"Point {i+1} {point}: Distance from origin = {distance:.2f}")
print()

# ============================================================================
# LIST COMPREHENSIONS
# ============================================================================

print("6. LIST COMPREHENSIONS")
print("=" * 40)

# Basic list comprehensions
squares = [x**2 for x in range(1, 6)]
even_numbers = [x for x in range(20) if x % 2 == 0]
word_lengths = [len(word) for word in ["hello", "world", "python"]]

print(f"Squares: {squares}")
print(f"Even numbers: {even_numbers}")
print(f"Word lengths: {word_lengths}")

# Advanced comprehensions
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened = [num for row in matrix for num in row]
print(f"Flattened matrix: {flattened}")

# Dictionary comprehension
word_list = ["hello", "world", "python"]
word_lengths_dict = {word: len(word) for word in word_list}
print(f"Word lengths dict: {word_lengths_dict}")

# Set comprehension
unique_squares = {x**2 for x in [-2, -1, 0, 1, 2]}
print(f"Unique squares: {unique_squares}")
print()

# ============================================================================
# CHOOSING THE RIGHT DATA STRUCTURE
# ============================================================================

print("7. CHOOSING THE RIGHT DATA STRUCTURE")
print("=" * 40)

# Shopping list (order matters, duplicates allowed) -> LIST
shopping_list = ["milk", "bread", "eggs", "milk"]
print(f"Shopping list (LIST): {shopping_list}")

# Phone book (key-value lookup) -> DICTIONARY
phone_book = {"Alice": "555-1234", "Bob": "555-5678"}
print(f"Phone book (DICT): {phone_book}")

# Unique visitors (no duplicates, fast membership) -> SET
unique_visitors = {"user1", "user2", "user3", "user1"}
print(f"Unique visitors (SET): {unique_visitors}")

# GPS coordinates (immutable, paired data) -> TUPLE
gps_location = (40.7128, -74.0060)
print(f"GPS location (TUPLE): {gps_location}")
print()

print("=== Data Structure Performance Tips ===")
print("• Lists: Use for ordered data, frequent appending")
print("• Dictionaries: Use for key-based lookups, O(1) access")
print("• Sets: Use for uniqueness, fast membership testing")
print("• Tuples: Use for immutable data, dictionary keys")
