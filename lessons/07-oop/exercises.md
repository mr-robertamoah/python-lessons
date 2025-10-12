# Lesson 05 Exercises: Object-Oriented Programming

## Guided Exercises (Do with Instructor)

### Exercise 1: Creating Your First Class
**Goal**: Understand basic class creation and object instantiation

**Tasks**:
1. Create a `Student` class with name, age, and grade attributes
2. Add methods to display student info and update grade
3. Create multiple student objects and test methods

```python
class Student:
    def __init__(self, name, age, grade):
        # Initialize attributes
        pass
    
    def display_info(self):
        # Display student information
        pass
    
    def update_grade(self, new_grade):
        # Update the student's grade
        pass

# Create and test student objects
```

---

### Exercise 2: Bank Account System
**Goal**: Practice encapsulation and data validation

**Tasks**:
1. Create a `BankAccount` class with account number and balance
2. Implement deposit, withdraw, and balance inquiry methods
3. Add validation to prevent negative balances
4. Use private attributes where appropriate

```python
class BankAccount:
    def __init__(self, account_number, initial_balance=0):
        # Initialize account
        pass
    
    def deposit(self, amount):
        # Add money to account
        pass
    
    def withdraw(self, amount):
        # Remove money from account
        pass
    
    def get_balance(self):
        # Return current balance
        pass
```

---

### Exercise 3: Inheritance with Vehicles
**Goal**: Implement inheritance and method overriding

**Tasks**:
1. Create a base `Vehicle` class
2. Create `Car` and `Motorcycle` subclasses
3. Override methods in subclasses
4. Demonstrate polymorphism

```python
class Vehicle:
    def __init__(self, brand, model, year):
        # Base vehicle initialization
        pass
    
    def start(self):
        # Start the vehicle
        pass

class Car(Vehicle):
    def __init__(self, brand, model, year, doors):
        # Car-specific initialization
        pass

class Motorcycle(Vehicle):
    def __init__(self, brand, model, year, engine_size):
        # Motorcycle-specific initialization
        pass
```

---

## Independent Exercises (Try on Your Own)

### Exercise 4: Library Management System
**Goal**: Build a comprehensive class hierarchy

**Requirements**:
1. Create `Book` class with title, author, ISBN, and availability
2. Create `Member` class with name, member ID, and borrowed books
3. Create `Library` class to manage books and members
4. Implement borrowing and returning functionality

**Features to implement**:
- Add/remove books from library
- Register/remove members
- Borrow/return books with due dates
- Search books by title or author
- Display member borrowing history

---

### Exercise 5: E-commerce Product System
**Goal**: Practice composition and inheritance

**Requirements**:
1. Create base `Product` class
2. Create specific product types (Electronics, Clothing, Books)
3. Implement shopping cart functionality
4. Add discount and tax calculations

**Classes to create**:
- `Product` (base class)
- `Electronics`, `Clothing`, `Book` (inherited classes)
- `ShoppingCart` (composition)
- `Customer` (has shopping cart)

---

### Exercise 6: Game Character System
**Goal**: Advanced OOP with multiple inheritance concepts

**Requirements**:
1. Create character classes with different abilities
2. Implement leveling and experience systems
3. Add inventory management
4. Create different character types (Warrior, Mage, Archer)

**Features**:
- Character stats (health, mana, strength, etc.)
- Skill systems and abilities
- Equipment and inventory
- Battle mechanics

---

## Challenge Exercises (If You Finish Early)

### Challenge 1: Restaurant Management System
**Goal**: Complex system with multiple interacting classes

**Requirements**:
1. Menu items with prices and categories
2. Order management with multiple items
3. Table reservation system
4. Staff management (waiters, chefs, managers)
5. Bill calculation with taxes and tips

### Challenge 2: Social Media Platform
**Goal**: Practice advanced OOP patterns

**Requirements**:
1. User profiles with friends/followers
2. Post creation and interaction (likes, comments)
3. Privacy settings and permissions
4. Notification system
5. Content moderation

---

## Design Pattern Exercises

### Exercise 7: Factory Pattern
**Goal**: Implement the Factory design pattern

**Tasks**:
1. Create a shape factory that produces different shapes
2. Implement Circle, Rectangle, and Triangle classes
3. Use factory to create shapes based on user input

### Exercise 8: Observer Pattern
**Goal**: Implement the Observer design pattern

**Tasks**:
1. Create a news publisher that notifies subscribers
2. Implement different types of subscribers
3. Add/remove subscribers dynamically

---

## Real-World Application Exercises

### Exercise 9: Employee Management System
**Goal**: Model a real business scenario

**Requirements**:
1. Different employee types (Full-time, Part-time, Contractor)
2. Salary calculation based on employee type
3. Department management
4. Performance tracking and reviews

### Exercise 10: Hotel Booking System
**Goal**: Complex business logic with OOP

**Requirements**:
1. Room types with different amenities and prices
2. Booking management with dates and availability
3. Customer profiles and booking history
4. Payment processing and billing

---

## Testing Your Classes

### Exercise 11: Unit Testing for OOP
**Goal**: Learn to test object-oriented code

**Tasks**:
1. Write unit tests for your classes
2. Test edge cases and error conditions
3. Use setUp and tearDown methods
4. Test inheritance and polymorphism

```python
import unittest

class TestBankAccount(unittest.TestCase):
    def setUp(self):
        self.account = BankAccount("12345", 100)
    
    def test_deposit(self):
        # Test deposit functionality
        pass
    
    def test_withdraw(self):
        # Test withdraw functionality
        pass
```

---

## Self-Assessment Checklist

After completing these exercises, you should be able to:

- [ ] Create classes with attributes and methods
- [ ] Use constructors (`__init__`) properly
- [ ] Implement encapsulation with private/protected attributes
- [ ] Create inheritance hierarchies
- [ ] Override methods in subclasses
- [ ] Use polymorphism effectively
- [ ] Implement composition relationships
- [ ] Use class methods and static methods
- [ ] Implement properties with getters and setters
- [ ] Use magic methods for operator overloading
- [ ] Apply common design patterns
- [ ] Test object-oriented code

## Common OOP Mistakes to Avoid

### 1. Overusing Inheritance
```python
# Bad: Forced inheritance
class Rectangle(Color):
    pass

# Good: Use composition
class Rectangle:
    def __init__(self, color):
        self.color = color
```

### 2. Breaking Encapsulation
```python
# Bad: Direct attribute access
account.balance = -1000

# Good: Use methods
account.withdraw(amount)
```

### 3. God Objects
```python
# Bad: Class doing too much
class GameManager:
    def render_graphics(self): pass
    def handle_input(self): pass
    def manage_network(self): pass
    def play_sound(self): pass

# Good: Separate responsibilities
class GraphicsRenderer: pass
class InputHandler: pass
class NetworkManager: pass
class SoundManager: pass
```

## Git Reminder

Save your work:
1. Create `lesson-05-oop` folder in your repository
2. Save your class implementations as `.py` files
3. Include test files for your classes
4. Document your design decisions
5. Commit and push:
   ```bash
   git add .
   git commit -m "Complete Lesson 05: Object-Oriented Programming"
   git push
   ```

## Next Lesson Preview

In Lesson 06, we'll learn about:
- **Data Structures**: Lists, dictionaries, sets, and tuples
- **When to use each data structure**
- **Performance considerations**
- **Combining OOP with data structures**
