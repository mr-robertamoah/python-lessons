# Lesson 05 Solutions: Object-Oriented Programming

print("=== GUIDED EXERCISE SOLUTIONS ===\n")

# Exercise 1: Creating Your First Class
print("Exercise 1: Creating Your First Class")
print("-" * 50)

class Student:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade
    
    def display_info(self):
        return f"Student: {self.name}, Age: {self.age}, Grade: {self.grade}"
    
    def update_grade(self, new_grade):
        old_grade = self.grade
        self.grade = new_grade
        return f"Grade updated from {old_grade} to {new_grade}"
    
    def __str__(self):
        return f"{self.name} (Age: {self.age}, Grade: {self.grade})"

# Test the Student class
student1 = Student("Alice Johnson", 16, "A")
student2 = Student("Bob Smith", 17, "B")

print(student1.display_info())
print(student2.display_info())
print(student1.update_grade("A+"))
print(f"Updated student: {student1}")

# Exercise 2: Bank Account System
print("\n" + "="*50)
print("Exercise 2: Bank Account System")
print("-" * 50)

class BankAccount:
    def __init__(self, account_number, initial_balance=0):
        self.account_number = account_number
        self.__balance = max(0, initial_balance)  # Private attribute
        self.__transaction_history = []
    
    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount
            self.__transaction_history.append(f"Deposited: ${amount:.2f}")
            return f"Deposited ${amount:.2f}. New balance: ${self.__balance:.2f}"
        return "Invalid deposit amount"
    
    def withdraw(self, amount):
        if amount <= 0:
            return "Invalid withdrawal amount"
        
        if amount > self.__balance:
            return "Insufficient funds"
        
        self.__balance -= amount
        self.__transaction_history.append(f"Withdrew: ${amount:.2f}")
        return f"Withdrew ${amount:.2f}. New balance: ${self.__balance:.2f}"
    
    def get_balance(self):
        return self.__balance
    
    def get_transaction_history(self):
        return self.__transaction_history.copy()
    
    def __str__(self):
        return f"Account {self.account_number}: ${self.__balance:.2f}"

# Test the BankAccount class
account = BankAccount("12345", 1000)
print(account)
print(account.deposit(500))
print(account.withdraw(200))
print(account.withdraw(2000))  # Should fail
print(f"Final balance: ${account.get_balance():.2f}")

# Exercise 3: Inheritance with Vehicles
print("\n" + "="*50)
print("Exercise 3: Inheritance with Vehicles")
print("-" * 50)

class Vehicle:
    def __init__(self, brand, model, year):
        self.brand = brand
        self.model = model
        self.year = year
        self.is_running = False
    
    def start(self):
        self.is_running = True
        return f"{self.brand} {self.model} started"
    
    def stop(self):
        self.is_running = False
        return f"{self.brand} {self.model} stopped"
    
    def get_info(self):
        return f"{self.year} {self.brand} {self.model}"
    
    def __str__(self):
        return self.get_info()

class Car(Vehicle):
    def __init__(self, brand, model, year, doors):
        super().__init__(brand, model, year)
        self.doors = doors
        self.speed = 0
    
    def accelerate(self, amount):
        if self.is_running:
            self.speed += amount
            return f"Car accelerated to {self.speed} mph"
        return "Start the car first!"
    
    def start(self):
        self.is_running = True
        return f"{self.brand} {self.model} car started with {self.doors} doors"

class Motorcycle(Vehicle):
    def __init__(self, brand, model, year, engine_size):
        super().__init__(brand, model, year)
        self.engine_size = engine_size
    
    def wheelie(self):
        if self.is_running:
            return f"{self.brand} {self.model} is doing a wheelie!"
        return "Start the motorcycle first!"
    
    def start(self):
        self.is_running = True
        return f"{self.brand} {self.model} motorcycle roars to life! ({self.engine_size}cc)"

# Test inheritance and polymorphism
vehicles = [
    Car("Toyota", "Camry", 2022, 4),
    Motorcycle("Harley", "Sportster", 2021, 883),
    Vehicle("Generic", "Vehicle", 2020)
]

print("Testing polymorphism:")
for vehicle in vehicles:
    print(f"  {vehicle.start()}")

# Test specific methods
car = vehicles[0]
motorcycle = vehicles[1]

print(f"\nCar specific: {car.accelerate(30)}")
print(f"Motorcycle specific: {motorcycle.wheelie()}")

print("\n=== INDEPENDENT EXERCISE SOLUTIONS ===\n")

# Exercise 4: Library Management System
print("Exercise 4: Library Management System")
print("-" * 50)

from datetime import datetime, timedelta

class Book:
    def __init__(self, title, author, isbn):
        self.title = title
        self.author = author
        self.isbn = isbn
        self.is_available = True
        self.borrowed_by = None
        self.due_date = None
    
    def __str__(self):
        status = "Available" if self.is_available else f"Borrowed (due: {self.due_date})"
        return f"'{self.title}' by {self.author} - {status}"

class Member:
    def __init__(self, name, member_id):
        self.name = name
        self.member_id = member_id
        self.borrowed_books = []
        self.borrowing_history = []
    
    def __str__(self):
        return f"Member: {self.name} (ID: {self.member_id}) - {len(self.borrowed_books)} books borrowed"

class Library:
    def __init__(self, name):
        self.name = name
        self.books = {}  # ISBN -> Book
        self.members = {}  # member_id -> Member
    
    def add_book(self, book):
        self.books[book.isbn] = book
        return f"Added book: {book.title}"
    
    def register_member(self, member):
        self.members[member.member_id] = member
        return f"Registered member: {member.name}"
    
    def borrow_book(self, member_id, isbn):
        if member_id not in self.members:
            return "Member not found"
        
        if isbn not in self.books:
            return "Book not found"
        
        book = self.books[isbn]
        member = self.members[member_id]
        
        if not book.is_available:
            return f"Book '{book.title}' is not available"
        
        # Borrow the book
        book.is_available = False
        book.borrowed_by = member_id
        book.due_date = datetime.now() + timedelta(days=14)
        
        member.borrowed_books.append(isbn)
        member.borrowing_history.append(f"Borrowed '{book.title}' on {datetime.now().strftime('%Y-%m-%d')}")
        
        return f"Book '{book.title}' borrowed by {member.name}. Due: {book.due_date.strftime('%Y-%m-%d')}"
    
    def return_book(self, member_id, isbn):
        if isbn not in self.books:
            return "Book not found"
        
        book = self.books[isbn]
        
        if book.is_available:
            return "Book is not currently borrowed"
        
        if book.borrowed_by != member_id:
            return "This book was not borrowed by this member"
        
        # Return the book
        book.is_available = True
        book.borrowed_by = None
        book.due_date = None
        
        member = self.members[member_id]
        member.borrowed_books.remove(isbn)
        member.borrowing_history.append(f"Returned '{book.title}' on {datetime.now().strftime('%Y-%m-%d')}")
        
        return f"Book '{book.title}' returned by {member.name}"
    
    def search_books(self, query):
        results = []
        query_lower = query.lower()
        
        for book in self.books.values():
            if (query_lower in book.title.lower() or 
                query_lower in book.author.lower()):
                results.append(book)
        
        return results
    
    def get_member_history(self, member_id):
        if member_id not in self.members:
            return "Member not found"
        
        member = self.members[member_id]
        return member.borrowing_history

# Test the Library Management System
library = Library("City Library")

# Add books
book1 = Book("Python Programming", "John Smith", "978-1234567890")
book2 = Book("Data Science Handbook", "Jane Doe", "978-0987654321")
book3 = Book("Machine Learning", "Alice Johnson", "978-1122334455")

library.add_book(book1)
library.add_book(book2)
library.add_book(book3)

# Register members
member1 = Member("Bob Wilson", "M001")
member2 = Member("Carol Brown", "M002")

library.register_member(member1)
library.register_member(member2)

# Test borrowing and returning
print(library.borrow_book("M001", "978-1234567890"))
print(library.borrow_book("M002", "978-0987654321"))
print(library.borrow_book("M001", "978-1234567890"))  # Should fail

print(f"\nCurrent library status:")
for book in library.books.values():
    print(f"  {book}")

print(f"\nMembers:")
for member in library.members.values():
    print(f"  {member}")

# Search functionality
search_results = library.search_books("Python")
print(f"\nSearch results for 'Python':")
for book in search_results:
    print(f"  {book}")

# Return a book
print(f"\n{library.return_book('M001', '978-1234567890')}")

# Exercise 5: E-commerce Product System
print("\n" + "="*50)
print("Exercise 5: E-commerce Product System")
print("-" * 50)

class Product:
    def __init__(self, name, price, category):
        self.name = name
        self.price = price
        self.category = category
    
    def get_price(self):
        return self.price
    
    def apply_discount(self, discount_percent):
        self.price *= (1 - discount_percent / 100)
    
    def __str__(self):
        return f"{self.name} - ${self.price:.2f} ({self.category})"

class Electronics(Product):
    def __init__(self, name, price, brand, warranty_years):
        super().__init__(name, price, "Electronics")
        self.brand = brand
        self.warranty_years = warranty_years
    
    def get_warranty_info(self):
        return f"{self.warranty_years} year warranty"
    
    def __str__(self):
        return f"{self.brand} {self.name} - ${self.price:.2f} ({self.get_warranty_info()})"

class Clothing(Product):
    def __init__(self, name, price, size, material):
        super().__init__(name, price, "Clothing")
        self.size = size
        self.material = material
    
    def __str__(self):
        return f"{self.name} (Size {self.size}, {self.material}) - ${self.price:.2f}"

class ShoppingCart:
    def __init__(self):
        self.items = []
    
    def add_item(self, product, quantity=1):
        self.items.append({"product": product, "quantity": quantity})
        return f"Added {quantity}x {product.name} to cart"
    
    def remove_item(self, product_name):
        for i, item in enumerate(self.items):
            if item["product"].name == product_name:
                removed = self.items.pop(i)
                return f"Removed {removed['product'].name} from cart"
        return "Item not found in cart"
    
    def calculate_total(self):
        total = 0
        for item in self.items:
            total += item["product"].get_price() * item["quantity"]
        return total
    
    def apply_tax(self, tax_rate):
        return self.calculate_total() * (1 + tax_rate / 100)
    
    def __str__(self):
        if not self.items:
            return "Cart is empty"
        
        result = "Shopping Cart:\n"
        for item in self.items:
            result += f"  {item['quantity']}x {item['product']}\n"
        result += f"Total: ${self.calculate_total():.2f}"
        return result

class Customer:
    def __init__(self, name, email):
        self.name = name
        self.email = email
        self.cart = ShoppingCart()
    
    def add_to_cart(self, product, quantity=1):
        return self.cart.add_item(product, quantity)
    
    def checkout(self, tax_rate=8.5):
        total = self.cart.apply_tax(tax_rate)
        items_count = len(self.cart.items)
        self.cart = ShoppingCart()  # Clear cart after checkout
        return f"Checkout complete! Total: ${total:.2f} for {items_count} items"

# Test the E-commerce system
laptop = Electronics("MacBook Pro", 1999.99, "Apple", 1)
shirt = Clothing("Cotton T-Shirt", 29.99, "L", "Cotton")
phone = Electronics("iPhone", 999.99, "Apple", 1)

customer = Customer("Alice Johnson", "alice@email.com")

print(customer.add_to_cart(laptop))
print(customer.add_to_cart(shirt, 2))
print(customer.add_to_cart(phone))

print(f"\n{customer.cart}")
print(f"\nWith tax: ${customer.cart.apply_tax(8.5):.2f}")
print(customer.checkout())

print("\n" + "="*50)
print("Object-Oriented Programming exercise solutions complete!")
print("Key concepts demonstrated:")
print("- Class creation with attributes and methods")
print("- Encapsulation with private attributes")
print("- Inheritance and method overriding")
print("- Polymorphism with different object types")
print("- Composition and complex object relationships")
print("- Real-world application design patterns")
