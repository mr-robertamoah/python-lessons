# Lesson 05 Exercises: Data Structures - Lists, Tuples, Dictionaries, Sets

## Guided Exercises (Do with Instructor)

### Exercise 1: List Operations
**Goal**: Practice creating and manipulating lists

**Tasks**:
1. Create a list of your top 5 favorite movies
2. Add a new movie to the end
3. Insert a movie at position 2
4. Remove a specific movie by name
5. Print the final list with numbering

```python
movies = ["Movie 1", "Movie 2", "Movie 3", "Movie 4", "Movie 5"]
# Complete the operations
```

---

### Exercise 2: Dictionary Basics
**Goal**: Store and access structured data

**Create a dictionary for a student with**:
- Name, age, grade, subjects (list), GPA
- Add a new subject
- Update the GPA
- Print formatted student info

```python
student = {
    "name": "Alice",
    # Add other fields
}
```

---

### Exercise 3: Tuple Practice
**Goal**: Work with immutable data

**Tasks**:
1. Create tuples for coordinates (x, y)
2. Store multiple coordinates in a list
3. Calculate distances between points
4. Try to modify a tuple (see what happens)

---

## Independent Exercises (Try on Your Own)

### Exercise 4: Shopping List Manager
**Goal**: Build a practical list application

**Requirements**:
Create a program that:
1. Starts with an empty shopping list
2. Adds items with quantities
3. Removes completed items
4. Shows the current list
5. Counts total items

**Features**:
- Use a list of dictionaries: `[{"item": "milk", "quantity": 2}]`
- Functions to add, remove, and display items

---

### Exercise 5: Grade Book System
**Goal**: Manage student grades with dictionaries

**Create a system that**:
1. Stores multiple students and their grades
2. Calculates average grade per student
3. Finds the highest and lowest grades
4. Lists students by grade range

**Data Structure**:
```python
gradebook = {
    "Alice": [85, 92, 78, 90],
    "Bob": [76, 88, 82, 85],
    # Add more students
}
```

---

### Exercise 6: Inventory Management
**Goal**: Track product inventory using multiple data structures

**Requirements**:
1. Use dictionaries for product info (name, price, stock)
2. Use sets to track categories
3. Use lists for transaction history
4. Implement add/remove stock functions

**Sample Structure**:
```python
inventory = {
    "P001": {"name": "Laptop", "price": 999.99, "stock": 5, "category": "Electronics"},
    # Add more products
}
```

---

### Exercise 7: Contact Book
**Goal**: Create a comprehensive contact management system

**Features**:
1. Store contacts with multiple phone numbers
2. Group contacts by categories (family, work, friends)
3. Search contacts by name or phone
4. Export contact list

**Data Structure**:
```python
contacts = {
    "Alice Johnson": {
        "phones": ["555-1234", "555-5678"],
        "email": "alice@email.com",
        "category": "friend",
        "address": "123 Main St"
    }
}
```

---

## Challenge Exercises (If You Finish Early)

### Challenge 1: Text Analysis Tool
**Goal**: Analyze text using various data structures

**Requirements**:
1. Count word frequency (dictionary)
2. Find unique words (set)
3. Track word positions (list of tuples)
4. Generate statistics report

**Input**: A paragraph of text
**Output**: Detailed analysis with counts, unique words, most common words

---

### Challenge 2: Library Management System
**Goal**: Complex data structure relationships

**Features**:
1. Books with multiple authors
2. Member checkout history
3. Available vs checked out books
4. Due date tracking
5. Fine calculations

**Data Structures**:
- Books: dictionary with lists of authors
- Members: dictionary with checkout history
- Transactions: list of tuples with dates

---

### Challenge 3: Social Network Graph
**Goal**: Represent relationships using data structures

**Create a system that**:
1. Stores users and their connections
2. Finds mutual friends
3. Suggests new connections
4. Calculates network statistics

**Use nested dictionaries and sets to represent the network**

---

## Data Structure Comparison Exercises

### Exercise 8: Performance Testing
**Goal**: Compare data structure performance

**Tasks**:
1. Create large datasets (1000+ items)
2. Time operations: search, add, remove
3. Compare list vs set vs dictionary performance
4. Document findings

```python
import time

# Test searching in list vs set
large_list = list(range(10000))
large_set = set(range(10000))

# Time the searches
```

---

### Exercise 9: Memory Usage Analysis
**Goal**: Understand memory implications

**Compare memory usage of**:
1. List vs tuple for same data
2. Dictionary vs list of tuples
3. Set vs list for unique items

Use `sys.getsizeof()` to measure memory

---

## Real-World Application Exercises

### Exercise 10: Restaurant Menu System
**Goal**: Model a restaurant's menu and orders

**Requirements**:
1. Menu items with prices and categories
2. Customer orders (list of items)
3. Calculate totals with tax
4. Track popular items

**Data Structure**:
```python
menu = {
    "appetizers": [
        {"name": "Wings", "price": 8.99},
        {"name": "Nachos", "price": 7.99}
    ],
    "entrees": [
        {"name": "Burger", "price": 12.99},
        {"name": "Pizza", "price": 14.99}
    ]
}
```

---

### Exercise 11: Weather Data Analysis
**Goal**: Process weather data using appropriate structures

**Tasks**:
1. Store daily weather data (temperature, humidity, conditions)
2. Calculate monthly averages
3. Find extreme weather days
4. Group by weather conditions

**Data Format**:
```python
weather_data = [
    {"date": "2024-01-01", "temp": 32, "humidity": 65, "condition": "sunny"},
    # More daily data
]
```

---

### Exercise 12: Student Course Registration
**Goal**: Model course registration system

**Features**:
1. Students can register for multiple courses
2. Courses have enrollment limits
3. Prerequisites tracking
4. Schedule conflict detection

**Data Structures**:
- Students: dictionary with enrolled courses
- Courses: dictionary with student lists and details
- Prerequisites: dictionary mapping courses to requirements

---

## Debugging Exercises

### Debug Exercise 1: List Issues
**Fix these common list problems**:

```python
# Problem 1: Index error
numbers = [1, 2, 3, 4, 5]
print(numbers[5])  # Error!

# Problem 2: Modifying list while iterating
items = ["a", "b", "c", "d"]
for item in items:
    if item == "b":
        items.remove(item)

# Problem 3: Shallow copy issue
original = [[1, 2], [3, 4]]
copy = original
copy[0][0] = 99
print(original)  # Unexpected change!
```

---

### Debug Exercise 2: Dictionary Problems
**Fix these dictionary issues**:

```python
# Problem 1: KeyError
student = {"name": "Alice", "age": 20}
print(student["grade"])  # Error!

# Problem 2: Modifying dict during iteration
scores = {"Alice": 85, "Bob": 92, "Charlie": 78}
for name in scores:
    if scores[name] < 80:
        del scores[name]  # Error!

# Problem 3: Mutable default argument
def add_item(item, inventory=[]):
    inventory.append(item)
    return inventory
```

---

## Self-Assessment Checklist

After completing these exercises, you should be able to:

- [ ] Create and manipulate lists (add, remove, modify elements)
- [ ] Use list methods (append, insert, remove, pop, sort)
- [ ] Create and access dictionary data
- [ ] Add, update, and remove dictionary items
- [ ] Use dictionary methods (keys, values, items)
- [ ] Create and use tuples appropriately
- [ ] Understand tuple immutability
- [ ] Create and use sets for unique data
- [ ] Perform set operations (union, intersection, difference)
- [ ] Choose appropriate data structure for different tasks
- [ ] Nest data structures (lists in dictionaries, etc.)
- [ ] Iterate through different data structures
- [ ] Handle common data structure errors

## Common Patterns and Best Practices

### When to Use Each Data Structure

**Lists**: When you need:
- Ordered data that can change
- Duplicate values
- Index-based access
- Example: Shopping cart items, game scores

**Tuples**: When you need:
- Ordered data that won't change
- Coordinates, database records
- Dictionary keys (if hashable)
- Example: GPS coordinates, RGB colors

**Dictionaries**: When you need:
- Key-value relationships
- Fast lookups by key
- Structured data
- Example: User profiles, configuration settings

**Sets**: When you need:
- Unique values only
- Fast membership testing
- Mathematical set operations
- Example: Unique visitors, tags, categories

### Performance Tips

1. **Use sets for membership testing**: `item in my_set` is faster than `item in my_list`
2. **Use dictionaries for lookups**: Faster than searching through lists
3. **Use list comprehensions**: More efficient than loops for creating lists
4. **Consider tuples for immutable data**: Slightly more memory efficient than lists

## Git Reminder

Save your work:

1. Create folder `lesson-05-data-structures` in your repository
2. Save exercise solutions as `.py` files
3. Add comments explaining your approach
4. Commit and push:
   ```bash
   git add .
   git commit -m "Complete Lesson 05: Data Structures"
   git push
   ```

## Next Lesson Preview

In Lesson 06, we'll learn about:
- **File Input/Output**: Reading and writing files
- **CSV file handling**: Working with structured data files
- **Error handling**: Managing file-related errors
- **File paths**: Working with different file locations
