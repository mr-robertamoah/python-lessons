# Lesson 05 Solutions: Data Structures - Lists, Tuples, Dictionaries, Sets

print("=== GUIDED EXERCISE SOLUTIONS ===\n")

# Exercise 1: List Operations
print("Exercise 1: List Operations")
print("-" * 40)

movies = ["The Matrix", "Inception", "Interstellar", "The Dark Knight", "Pulp Fiction"]
print("Original movies:", movies)

# Add movie to end
movies.append("Dune")
print("After append:", movies)

# Insert at position 2
movies.insert(2, "Blade Runner")
print("After insert:", movies)

# Remove specific movie
movies.remove("Pulp Fiction")
print("After remove:", movies)

# Print with numbering
print("\nFinal movie list:")
for i, movie in enumerate(movies, 1):
    print(f"{i}. {movie}")
print()

# Exercise 2: Dictionary Basics
print("Exercise 2: Dictionary Basics")
print("-" * 40)

student = {
    "name": "Alice Johnson",
    "age": 20,
    "grade": "Junior",
    "subjects": ["Math", "Physics", "Computer Science"],
    "gpa": 3.7
}

print("Original student:", student)

# Add new subject
student["subjects"].append("Chemistry")
print("After adding subject:", student["subjects"])

# Update GPA
student["gpa"] = 3.8
print("Updated GPA:", student["gpa"])

# Print formatted info
print(f"\nStudent Profile:")
print(f"Name: {student['name']}")
print(f"Age: {student['age']}")
print(f"Grade: {student['grade']}")
print(f"GPA: {student['gpa']}")
print(f"Subjects: {', '.join(student['subjects'])}")
print()

# Exercise 3: Tuple Practice
print("Exercise 3: Tuple Practice")
print("-" * 40)

# Create coordinate tuples
point1 = (0, 0)
point2 = (3, 4)
point3 = (6, 8)

coordinates = [point1, point2, point3]
print("Coordinates:", coordinates)

# Calculate distance between point1 and point2
import math
distance = math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
print(f"Distance between {point1} and {point2}: {distance}")

# Try to modify tuple (will cause error if uncommented)
# point1[0] = 5  # TypeError: 'tuple' object does not support item assignment
print("Tuples are immutable - cannot modify elements")
print()

print("=== INDEPENDENT EXERCISE SOLUTIONS ===\n")

# Exercise 4: Shopping List Manager
print("Exercise 4: Shopping List Manager")
print("-" * 40)

shopping_list = []

def add_item(item, quantity):
    shopping_list.append({"item": item, "quantity": quantity})
    print(f"Added {quantity} {item}(s) to list")

def remove_item(item):
    for i, list_item in enumerate(shopping_list):
        if list_item["item"] == item:
            removed = shopping_list.pop(i)
            print(f"Removed {removed['item']} from list")
            return
    print(f"{item} not found in list")

def display_list():
    if not shopping_list:
        print("Shopping list is empty")
        return
    
    print("Current Shopping List:")
    total_items = 0
    for item in shopping_list:
        print(f"- {item['item']}: {item['quantity']}")
        total_items += item['quantity']
    print(f"Total items: {total_items}")

# Demo the shopping list
add_item("milk", 2)
add_item("bread", 1)
add_item("eggs", 12)
display_list()
remove_item("bread")
display_list()
print()

# Exercise 5: Grade Book System
print("Exercise 5: Grade Book System")
print("-" * 40)

gradebook = {
    "Alice": [85, 92, 78, 90],
    "Bob": [76, 88, 82, 85],
    "Charlie": [95, 87, 91, 89],
    "Diana": [68, 75, 72, 80]
}

def calculate_average(grades):
    return sum(grades) / len(grades)

def find_highest_lowest():
    all_grades = []
    for grades in gradebook.values():
        all_grades.extend(grades)
    return max(all_grades), min(all_grades)

def students_by_grade_range(min_avg, max_avg):
    result = []
    for student, grades in gradebook.items():
        avg = calculate_average(grades)
        if min_avg <= avg <= max_avg:
            result.append((student, avg))
    return result

# Calculate and display results
print("Student Averages:")
for student, grades in gradebook.items():
    avg = calculate_average(grades)
    print(f"{student}: {avg:.1f}")

highest, lowest = find_highest_lowest()
print(f"\nHighest grade: {highest}")
print(f"Lowest grade: {lowest}")

a_students = students_by_grade_range(90, 100)
print(f"\nA students (90-100): {[f'{name} ({avg:.1f})' for name, avg in a_students]}")
print()

# Exercise 6: Inventory Management
print("Exercise 6: Inventory Management")
print("-" * 40)

inventory = {
    "P001": {"name": "Laptop", "price": 999.99, "stock": 5, "category": "Electronics"},
    "P002": {"name": "Mouse", "price": 29.99, "stock": 20, "category": "Electronics"},
    "P003": {"name": "Desk", "price": 199.99, "stock": 3, "category": "Furniture"},
    "P004": {"name": "Chair", "price": 149.99, "stock": 8, "category": "Furniture"}
}

categories = set()
transaction_history = []

def add_stock(product_id, quantity):
    if product_id in inventory:
        inventory[product_id]["stock"] += quantity
        transaction_history.append(("add", product_id, quantity))
        print(f"Added {quantity} units of {inventory[product_id]['name']}")
    else:
        print("Product not found")

def remove_stock(product_id, quantity):
    if product_id in inventory:
        if inventory[product_id]["stock"] >= quantity:
            inventory[product_id]["stock"] -= quantity
            transaction_history.append(("remove", product_id, quantity))
            print(f"Removed {quantity} units of {inventory[product_id]['name']}")
        else:
            print("Insufficient stock")
    else:
        print("Product not found")

# Collect categories
for product in inventory.values():
    categories.add(product["category"])

print("Categories:", categories)
print("\nInventory Status:")
for pid, product in inventory.items():
    print(f"{pid}: {product['name']} - ${product['price']:.2f} - Stock: {product['stock']}")

# Demo transactions
add_stock("P001", 2)
remove_stock("P002", 5)
print(f"\nTransaction History: {transaction_history}")
print()

# Exercise 7: Contact Book
print("Exercise 7: Contact Book")
print("-" * 40)

contacts = {
    "Alice Johnson": {
        "phones": ["555-1234", "555-5678"],
        "email": "alice@email.com",
        "category": "friend",
        "address": "123 Main St"
    },
    "Bob Smith": {
        "phones": ["555-9999"],
        "email": "bob@work.com",
        "category": "work",
        "address": "456 Oak Ave"
    },
    "Carol Wilson": {
        "phones": ["555-1111", "555-2222", "555-3333"],
        "email": "carol@family.com",
        "category": "family",
        "address": "789 Pine Rd"
    }
}

def search_contact(query):
    results = []
    for name, info in contacts.items():
        if query.lower() in name.lower():
            results.append(name)
        elif query in info["phones"]:
            results.append(name)
    return results

def contacts_by_category(category):
    return [name for name, info in contacts.items() if info["category"] == category]

def export_contacts():
    print("Contact Export:")
    for name, info in contacts.items():
        print(f"Name: {name}")
        print(f"Phones: {', '.join(info['phones'])}")
        print(f"Email: {info['email']}")
        print(f"Category: {info['category']}")
        print(f"Address: {info['address']}")
        print("-" * 20)

# Demo contact operations
search_results = search_contact("Alice")
print(f"Search for 'Alice': {search_results}")

family_contacts = contacts_by_category("family")
print(f"Family contacts: {family_contacts}")

export_contacts()
print()

print("=== CHALLENGE EXERCISE SOLUTIONS ===\n")

# Challenge 1: Text Analysis Tool
print("Challenge 1: Text Analysis Tool")
print("-" * 40)

text = "The quick brown fox jumps over the lazy dog. The dog was very lazy."

def analyze_text(text):
    # Clean and split text
    words = text.lower().replace(".", "").replace(",", "").split()
    
    # Word frequency
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Unique words
    unique_words = set(words)
    
    # Word positions
    word_positions = []
    for i, word in enumerate(words):
        word_positions.append((word, i))
    
    return word_freq, unique_words, word_positions

word_freq, unique_words, word_positions = analyze_text(text)

print(f"Text: {text}")
print(f"Total words: {len(text.split())}")
print(f"Unique words: {len(unique_words)}")
print(f"Word frequency: {dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True))}")
print(f"Most common word: {max(word_freq, key=word_freq.get)} ({word_freq[max(word_freq, key=word_freq.get)]} times)")
print()

# Challenge 2: Library Management System
print("Challenge 2: Library Management System")
print("-" * 40)

from datetime import datetime, timedelta

books = {
    "B001": {
        "title": "Python Programming",
        "authors": ["John Smith", "Jane Doe"],
        "available": True,
        "checked_out_by": None,
        "due_date": None
    },
    "B002": {
        "title": "Data Science Handbook",
        "authors": ["Alice Johnson"],
        "available": False,
        "checked_out_by": "M001",
        "due_date": datetime.now() + timedelta(days=14)
    }
}

members = {
    "M001": {
        "name": "Bob Wilson",
        "checkout_history": ["B002"],
        "fines": 0.0
    },
    "M002": {
        "name": "Carol Brown",
        "checkout_history": [],
        "fines": 5.50
    }
}

def checkout_book(book_id, member_id):
    if book_id in books and books[book_id]["available"]:
        books[book_id]["available"] = False
        books[book_id]["checked_out_by"] = member_id
        books[book_id]["due_date"] = datetime.now() + timedelta(days=14)
        members[member_id]["checkout_history"].append(book_id)
        print(f"Book {book_id} checked out to {members[member_id]['name']}")
    else:
        print("Book not available")

def return_book(book_id):
    if book_id in books and not books[book_id]["available"]:
        books[book_id]["available"] = True
        books[book_id]["checked_out_by"] = None
        books[book_id]["due_date"] = None
        print(f"Book {book_id} returned")
    else:
        print("Book not checked out")

print("Library System Demo:")
print(f"Available books: {[bid for bid, book in books.items() if book['available']]}")
checkout_book("B001", "M002")
return_book("B002")
print()

print("=== DEBUG EXERCISE SOLUTIONS ===\n")

print("Debug Exercise 1: List Issues - Fixed")
print("-" * 40)

# Fixed Problem 1: Index error
numbers = [1, 2, 3, 4, 5]
if len(numbers) > 5:
    print(numbers[5])
else:
    print("Index 5 is out of range")

# Fixed Problem 2: Modifying list while iterating
items = ["a", "b", "c", "d"]
items_to_remove = []
for item in items:
    if item == "b":
        items_to_remove.append(item)
for item in items_to_remove:
    items.remove(item)
print("Items after removal:", items)

# Fixed Problem 3: Shallow copy issue
original = [[1, 2], [3, 4]]
import copy
deep_copy = copy.deepcopy(original)
deep_copy[0][0] = 99
print("Original:", original)
print("Deep copy:", deep_copy)
print()

print("Debug Exercise 2: Dictionary Problems - Fixed")
print("-" * 40)

# Fixed Problem 1: KeyError
student = {"name": "Alice", "age": 20}
grade = student.get("grade", "Not assigned")
print("Grade:", grade)

# Fixed Problem 2: Modifying dict during iteration
scores = {"Alice": 85, "Bob": 92, "Charlie": 78}
to_remove = []
for name, score in scores.items():
    if score < 80:
        to_remove.append(name)
for name in to_remove:
    del scores[name]
print("Scores after removal:", scores)

# Fixed Problem 3: Mutable default argument
def add_item(item, inventory=None):
    if inventory is None:
        inventory = []
    inventory.append(item)
    return inventory

list1 = add_item("apple")
list2 = add_item("banana")
print("List 1:", list1)
print("List 2:", list2)
print()

print("=== PERFORMANCE COMPARISON ===\n")

import time
import sys

# Performance testing
print("Performance Testing: List vs Set vs Dictionary")
print("-" * 50)

# Create test data
test_data = list(range(10000))
test_list = test_data
test_set = set(test_data)
test_dict = {i: i for i in test_data}

search_item = 9999

# Time list search
start = time.time()
result = search_item in test_list
list_time = time.time() - start

# Time set search
start = time.time()
result = search_item in test_set
set_time = time.time() - start

# Time dict search
start = time.time()
result = search_item in test_dict
dict_time = time.time() - start

print(f"List search time: {list_time:.6f} seconds")
print(f"Set search time: {set_time:.6f} seconds")
print(f"Dict search time: {dict_time:.6f} seconds")

# Memory usage comparison
print(f"\nMemory Usage:")
print(f"List: {sys.getsizeof(test_list)} bytes")
print(f"Set: {sys.getsizeof(test_set)} bytes")
print(f"Dict: {sys.getsizeof(test_dict)} bytes")

# Tuple vs List memory
test_tuple = tuple(test_data)
print(f"Tuple: {sys.getsizeof(test_tuple)} bytes")
