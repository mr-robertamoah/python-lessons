# Lesson 05: Data Structures - Lists, Dictionaries, Sets, and Tuples

## Learning Objectives
By the end of this lesson, you will be able to:
- Create and manipulate lists for ordered data collections
- Use dictionaries to store key-value pairs
- Work with sets for unique collections
- Understand when to use tuples for immutable data
- Apply list comprehensions for elegant data processing
- Choose the right data structure for different problems

## What are Data Structures?

### Real-World Analogy: Organization Systems
Think of data structures like different ways to organize things:

- **List**: Like a shopping list - ordered, can have duplicates, items can be changed
- **Dictionary**: Like a phone book - look up information by name (key)
- **Set**: Like a collection of unique trading cards - no duplicates allowed
- **Tuple**: Like coordinates (x, y) - ordered but can't be changed

```python
# Shopping list (List)
shopping_list = ["apples", "bread", "milk", "apples"]  # Can have duplicates

# Phone book (Dictionary)
phone_book = {"Alice": "555-1234", "Bob": "555-5678"}

# Unique cards (Set)
card_collection = {"Pikachu", "Charizard", "Blastoise"}  # No duplicates

# Coordinates (Tuple)
location = (40.7128, -74.0060)  # NYC coordinates - shouldn't change
```

## Lists - Ordered Collections

### Creating Lists
```python
# Empty list
empty_list = []
numbers = list()  # Alternative way

# List with initial values
fruits = ["apple", "banana", "orange"]
numbers = [1, 2, 3, 4, 5]
mixed = ["Alice", 25, True, 3.14]  # Can mix data types

# List from range
countdown = list(range(10, 0, -1))  # [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
```

### Accessing List Elements
```python
fruits = ["apple", "banana", "orange", "grape"]

# Positive indexing (from start)
print(fruits[0])   # "apple" (first item)
print(fruits[1])   # "banana" (second item)
print(fruits[3])   # "grape" (fourth item)

# Negative indexing (from end)
print(fruits[-1])  # "grape" (last item)
print(fruits[-2])  # "orange" (second to last)

# Slicing
print(fruits[1:3])   # ["banana", "orange"] (items 1 and 2)
print(fruits[:2])    # ["apple", "banana"] (first 2 items)
print(fruits[2:])    # ["orange", "grape"] (from item 2 to end)
print(fruits[::2])   # ["apple", "orange"] (every 2nd item)
```

### Modifying Lists
```python
fruits = ["apple", "banana", "orange"]

# Change individual items
fruits[1] = "blueberry"
print(fruits)  # ["apple", "blueberry", "orange"]

# Add items
fruits.append("grape")           # Add to end
fruits.insert(1, "cherry")      # Insert at position 1
fruits.extend(["kiwi", "mango"]) # Add multiple items

# Remove items
fruits.remove("apple")    # Remove first occurrence
last_fruit = fruits.pop() # Remove and return last item
second_fruit = fruits.pop(1)  # Remove and return item at index 1
del fruits[0]            # Delete item at index 0
```

### List Methods and Operations
```python
numbers = [3, 1, 4, 1, 5, 9, 2, 6]

# Information about list
print(len(numbers))        # 8 (length)
print(numbers.count(1))    # 2 (how many 1's)
print(numbers.index(4))    # 2 (position of first 4)

# Sorting and reversing
numbers.sort()             # Sort in place: [1, 1, 2, 3, 4, 5, 6, 9]
numbers.reverse()          # Reverse in place: [9, 6, 5, 4, 3, 2, 1, 1]

# Non-destructive operations
original = [3, 1, 4, 1, 5]
sorted_copy = sorted(original)     # Returns new sorted list
reversed_copy = list(reversed(original))  # Returns new reversed list

# List operations
list1 = [1, 2, 3]
list2 = [4, 5, 6]
combined = list1 + list2   # [1, 2, 3, 4, 5, 6]
repeated = list1 * 3       # [1, 2, 3, 1, 2, 3, 1, 2, 3]

# Check membership
print(2 in list1)          # True
print(7 not in list1)      # True
```

## Dictionaries - Key-Value Pairs

### Creating Dictionaries
```python
# Empty dictionary
empty_dict = {}
student = dict()  # Alternative way

# Dictionary with initial values
student = {
    "name": "Alice Johnson",
    "age": 20,
    "major": "Computer Science",
    "gpa": 3.8
}

# Dictionary from lists
keys = ["name", "age", "city"]
values = ["Bob", 25, "Seattle"]
person = dict(zip(keys, values))
```

### Accessing Dictionary Values
```python
student = {"name": "Alice", "age": 20, "major": "CS"}

# Access values by key
print(student["name"])     # "Alice"
print(student.get("age"))  # 20

# Safe access with default
print(student.get("gpa", 0.0))  # 0.0 (default if key doesn't exist)

# Check if key exists
if "major" in student:
    print(f"Major: {student['major']}")
```

### Modifying Dictionaries
```python
student = {"name": "Alice", "age": 20}

# Add or update values
student["major"] = "Computer Science"  # Add new key-value pair
student["age"] = 21                    # Update existing value

# Update multiple values
student.update({"gpa": 3.8, "year": "Junior"})

# Remove items
del student["age"]              # Remove specific key
gpa = student.pop("gpa", 0.0)  # Remove and return value (with default)
last_item = student.popitem()  # Remove and return arbitrary item
student.clear()                # Remove all items
```

### Dictionary Methods
```python
student = {"name": "Alice", "age": 20, "major": "CS", "gpa": 3.8}

# Get all keys, values, or items
print(student.keys())    # dict_keys(['name', 'age', 'major', 'gpa'])
print(student.values())  # dict_values(['Alice', 20, 'CS', 3.8])
print(student.items())   # dict_items([('name', 'Alice'), ('age', 20), ...])

# Iterate through dictionary
for key in student:
    print(f"{key}: {student[key]}")

for key, value in student.items():
    print(f"{key}: {value}")

for value in student.values():
    print(value)
```

## Sets - Unique Collections

### Creating Sets
```python
# Empty set
empty_set = set()  # Note: {} creates empty dict, not set

# Set with initial values
colors = {"red", "green", "blue"}
numbers = set([1, 2, 3, 2, 1])  # {1, 2, 3} - duplicates removed

# Set from string
letters = set("hello")  # {'h', 'e', 'l', 'o'} - duplicates removed
```

### Set Operations
```python
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}

# Add and remove items
set1.add(5)        # Add single item
set1.update([6, 7]) # Add multiple items
set1.remove(1)     # Remove item (raises error if not found)
set1.discard(10)   # Remove item (no error if not found)

# Set mathematics
union = set1 | set2           # {2, 3, 4, 5, 6, 7} - all items
intersection = set1 & set2    # {3, 4, 5, 6} - common items
difference = set1 - set2      # {2, 7} - items in set1 but not set2
symmetric_diff = set1 ^ set2  # {2, 7} - items in either but not both

# Set relationships
print(set1.issubset(set2))    # False
print(set1.issuperset(set2))  # False
print(set1.isdisjoint(set2))  # False (they share common elements)
```

## Tuples - Immutable Sequences

### Creating Tuples
```python
# Empty tuple
empty_tuple = ()
empty_tuple = tuple()

# Tuple with values
coordinates = (10, 20)
rgb_color = (255, 128, 0)
mixed_tuple = ("Alice", 25, True)

# Single item tuple (note the comma!)
single_item = (42,)  # Without comma, it's just parentheses around 42

# Tuple from list
numbers_list = [1, 2, 3]
numbers_tuple = tuple(numbers_list)
```

### Working with Tuples
```python
point = (10, 20, 30)

# Access items (like lists)
x = point[0]  # 10
y = point[1]  # 20
z = point[2]  # 30

# Slicing works too
first_two = point[:2]  # (10, 20)

# Tuple unpacking
x, y, z = point
print(f"x={x}, y={y}, z={z}")

# Tuple methods
numbers = (1, 2, 3, 2, 1)
print(numbers.count(2))  # 2 (how many 2's)
print(numbers.index(3))  # 2 (position of first 3)

# Tuples are immutable
# point[0] = 15  # This would cause an error!
```

## List Comprehensions

### Basic List Comprehensions
```python
# Traditional way
squares = []
for x in range(10):
    squares.append(x ** 2)

# List comprehension way
squares = [x ** 2 for x in range(10)]
# [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# More examples
even_numbers = [x for x in range(20) if x % 2 == 0]
# [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

words = ["hello", "world", "python"]
lengths = [len(word) for word in words]
# [5, 5, 6]

uppercase = [word.upper() for word in words]
# ['HELLO', 'WORLD', 'PYTHON']
```

### Advanced List Comprehensions
```python
# Conditional expressions
numbers = [1, -2, 3, -4, 5]
abs_numbers = [x if x >= 0 else -x for x in numbers]
# [1, 2, 3, 4, 5]

# Nested loops
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened = [num for row in matrix for num in row]
# [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Dictionary comprehension
words = ["hello", "world", "python"]
word_lengths = {word: len(word) for word in words}
# {'hello': 5, 'world': 5, 'python': 6}

# Set comprehension
numbers = [1, 2, 2, 3, 3, 3, 4]
unique_squares = {x ** 2 for x in numbers}
# {1, 4, 9, 16}
```

## Practical Examples

### Example 1: Student Grade Management
```python
# Using different data structures for student management
students = [
    {"name": "Alice", "grades": [85, 92, 78, 96]},
    {"name": "Bob", "grades": [88, 76, 91, 83]},
    {"name": "Charlie", "grades": [92, 95, 89, 94]}
]

def analyze_class_performance(students):
    """Analyze performance of all students"""
    all_grades = []
    student_averages = {}
    
    for student in students:
        name = student["name"]
        grades = student["grades"]
        average = sum(grades) / len(grades)
        
        student_averages[name] = round(average, 2)
        all_grades.extend(grades)  # Add all grades to master list
    
    class_average = sum(all_grades) / len(all_grades)
    
    return {
        "student_averages": student_averages,
        "class_average": round(class_average, 2),
        "highest_average": max(student_averages.values()),
        "lowest_average": min(student_averages.values())
    }

# Analyze the class
results = analyze_class_performance(students)
print("Class Analysis:")
for key, value in results.items():
    print(f"  {key.replace('_', ' ').title()}: {value}")
```

### Example 2: Inventory Management
```python
# Using sets and dictionaries for inventory
inventory = {
    "apples": {"quantity": 50, "price": 0.75, "category": "fruit"},
    "bread": {"quantity": 20, "price": 2.50, "category": "bakery"},
    "milk": {"quantity": 15, "price": 3.25, "category": "dairy"},
    "bananas": {"quantity": 30, "price": 0.60, "category": "fruit"}
}

def low_stock_items(inventory, threshold=25):
    """Find items with low stock"""
    return [item for item, details in inventory.items() 
            if details["quantity"] < threshold]

def items_by_category(inventory, category):
    """Get all items in a specific category"""
    return {item: details for item, details in inventory.items() 
            if details["category"] == category}

def calculate_inventory_value(inventory):
    """Calculate total value of inventory"""
    return sum(details["quantity"] * details["price"] 
              for details in inventory.values())

# Use the functions
print("Low stock items:", low_stock_items(inventory))
print("Fruit items:", items_by_category(inventory, "fruit"))
print(f"Total inventory value: ${calculate_inventory_value(inventory):.2f}")
```

### Example 3: Data Analysis with Lists
```python
# Analyzing survey data
survey_responses = [
    {"age": 25, "satisfaction": 8, "department": "Engineering"},
    {"age": 32, "satisfaction": 6, "department": "Marketing"},
    {"age": 28, "satisfaction": 9, "department": "Engineering"},
    {"age": 45, "satisfaction": 7, "department": "Sales"},
    {"age": 29, "satisfaction": 8, "department": "Marketing"},
    {"age": 38, "satisfaction": 5, "department": "Sales"}
]

def analyze_survey_data(responses):
    """Comprehensive survey analysis"""
    # Extract data using list comprehensions
    ages = [r["age"] for r in responses]
    satisfactions = [r["satisfaction"] for r in responses]
    departments = [r["department"] for r in responses]
    
    # Calculate statistics
    avg_age = sum(ages) / len(ages)
    avg_satisfaction = sum(satisfactions) / len(satisfactions)
    
    # Department analysis
    dept_satisfaction = {}
    for response in responses:
        dept = response["department"]
        if dept not in dept_satisfaction:
            dept_satisfaction[dept] = []
        dept_satisfaction[dept].append(response["satisfaction"])
    
    # Calculate department averages
    dept_averages = {dept: sum(scores) / len(scores) 
                    for dept, scores in dept_satisfaction.items()}
    
    return {
        "total_responses": len(responses),
        "average_age": round(avg_age, 1),
        "average_satisfaction": round(avg_satisfaction, 1),
        "department_satisfaction": dept_averages,
        "age_range": (min(ages), max(ages))
    }

analysis = analyze_survey_data(survey_responses)
print("Survey Analysis:")
for key, value in analysis.items():
    print(f"  {key.replace('_', ' ').title()}: {value}")
```

## Choosing the Right Data Structure

### Decision Guide
```python
# Use LIST when:
# - Order matters
# - You need to allow duplicates
# - You need to modify items frequently
shopping_list = ["apples", "bread", "milk", "apples"]

# Use DICTIONARY when:
# - You need to look up values by key
# - You have paired data (key-value relationships)
# - You need fast lookups
student_grades = {"Alice": 95, "Bob": 87, "Charlie": 92}

# Use SET when:
# - You need unique items only
# - You need fast membership testing
# - You need set operations (union, intersection)
unique_visitors = {"user1", "user2", "user3"}

# Use TUPLE when:
# - Data shouldn't change (immutable)
# - You need to use as dictionary key
# - You're returning multiple values from function
coordinates = (40.7128, -74.0060)  # NYC latitude, longitude
```

## Performance Considerations

### Time Complexity (Big O)
```python
# List operations
my_list = [1, 2, 3, 4, 5]
# Access by index: O(1)
item = my_list[2]
# Search for item: O(n)
index = my_list.index(3)
# Append to end: O(1)
my_list.append(6)
# Insert at beginning: O(n)
my_list.insert(0, 0)

# Dictionary operations
my_dict = {"a": 1, "b": 2, "c": 3}
# Access by key: O(1) average
value = my_dict["a"]
# Check if key exists: O(1) average
exists = "b" in my_dict

# Set operations
my_set = {1, 2, 3, 4, 5}
# Check membership: O(1) average
exists = 3 in my_set
# Add item: O(1) average
my_set.add(6)
```

## Common Patterns and Idioms

### Data Processing Patterns
```python
# Filter and transform data
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Get even numbers squared
even_squares = [x**2 for x in numbers if x % 2 == 0]

# Group data by condition
def group_by_condition(data, condition_func):
    groups = {"true": [], "false": []}
    for item in data:
        key = "true" if condition_func(item) else "false"
        groups[key].append(item)
    return groups

# Count occurrences
def count_items(items):
    counts = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return counts

# Example usage
words = ["apple", "banana", "apple", "cherry", "banana", "apple"]
word_counts = count_items(words)
print(word_counts)  # {'apple': 3, 'banana': 2, 'cherry': 1}
```

## Best Practices

### 1. Choose Appropriate Data Structure
```python
# Good: Use set for unique items
unique_ids = set()
unique_ids.add(user_id)

# Avoid: Use list when you need uniqueness
unique_ids = []
if user_id not in unique_ids:  # Slow for large lists
    unique_ids.append(user_id)
```

### 2. Use List Comprehensions Wisely
```python
# Good: Simple, readable comprehensions
squares = [x**2 for x in range(10)]

# Avoid: Complex comprehensions that hurt readability
# This is hard to read:
result = [x**2 for x in range(100) if x % 2 == 0 if x > 10 if x < 50]

# Better: Break into steps or use regular loop
numbers = range(100)
filtered = [x for x in numbers if x % 2 == 0 and 10 < x < 50]
squares = [x**2 for x in filtered]
```

### 3. Handle Empty Collections
```python
def safe_average(numbers):
    """Calculate average, handling empty list"""
    if not numbers:  # Empty list is falsy
        return 0
    return sum(numbers) / len(numbers)

def safe_max(numbers):
    """Find maximum, handling empty list"""
    if not numbers:
        return None
    return max(numbers)
```

## Key Terminology

- **List**: Ordered, mutable collection that allows duplicates
- **Dictionary**: Unordered collection of key-value pairs
- **Set**: Unordered collection of unique items
- **Tuple**: Ordered, immutable collection
- **Index**: Position of an item in a sequence
- **Key**: Identifier used to access values in a dictionary
- **Mutable**: Can be changed after creation
- **Immutable**: Cannot be changed after creation
- **Comprehension**: Concise way to create collections
- **Unpacking**: Extracting values from a collection into variables

## Looking Ahead

In Lesson 06, we'll learn about:
- **File Input/Output**: Reading and writing files
- **CSV files**: Working with structured data
- **Exception handling**: Dealing with errors gracefully
- **Context managers**: Proper resource management
