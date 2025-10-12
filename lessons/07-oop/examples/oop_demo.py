#!/usr/bin/env python3
"""
Object-Oriented Programming Demo
Comprehensive examples of OOP concepts in Python
"""

# Basic Class Example
class Car:
    """A simple car class demonstrating basic OOP concepts"""
    
    # Class attribute (shared by all instances)
    wheels = 4
    
    def __init__(self, brand, model, year, color="white"):
        """Constructor method"""
        self.brand = brand
        self.model = model
        self.year = year
        self.color = color
        self.speed = 0
        self.is_running = False
    
    def start(self):
        """Start the car"""
        self.is_running = True
        return f"The {self.brand} {self.model} is now running!"
    
    def accelerate(self, amount):
        """Increase speed"""
        if self.is_running:
            self.speed += amount
            return f"Speed increased to {self.speed} mph"
        return "Start the car first!"
    
    def brake(self, amount):
        """Decrease speed"""
        self.speed = max(0, self.speed - amount)
        return f"Speed decreased to {self.speed} mph"
    
    def stop(self):
        """Stop the car"""
        self.speed = 0
        self.is_running = False
        return f"The {self.brand} {self.model} has stopped"
    
    def __str__(self):
        """String representation"""
        return f"{self.year} {self.brand} {self.model} ({self.color})"
    
    def __repr__(self):
        """Developer representation"""
        return f"Car('{self.brand}', '{self.model}', {self.year}, '{self.color}')"

# Encapsulation Example
class BankAccount:
    """Bank account demonstrating encapsulation"""
    
    def __init__(self, account_number, owner, initial_balance=0):
        self.account_number = account_number
        self.owner = owner
        self._balance = initial_balance  # Protected attribute
        self.__pin = "1234"  # Private attribute
        self._transaction_history = []
    
    def deposit(self, amount):
        """Deposit money"""
        if amount > 0:
            self._balance += amount
            self._transaction_history.append(f"Deposited ${amount:.2f}")
            return True
        return False
    
    def withdraw(self, amount):
        """Withdraw money"""
        if 0 < amount <= self._balance:
            self._balance -= amount
            self._transaction_history.append(f"Withdrew ${amount:.2f}")
            return True
        return False
    
    def get_balance(self):
        """Get current balance"""
        return self._balance
    
    def get_transaction_history(self):
        """Get transaction history"""
        return self._transaction_history.copy()
    
    def __str__(self):
        return f"Account {self.account_number} ({self.owner}): ${self._balance:.2f}"

# Inheritance Example
class Vehicle:
    """Base vehicle class"""
    
    def __init__(self, brand, model, year):
        self.brand = brand
        self.model = model
        self.year = year
        self.is_running = False
    
    def start(self):
        """Start the vehicle"""
        self.is_running = True
        return f"{self.brand} {self.model} started"
    
    def stop(self):
        """Stop the vehicle"""
        self.is_running = False
        return f"{self.brand} {self.model} stopped"
    
    def get_info(self):
        """Get vehicle information"""
        return f"{self.year} {self.brand} {self.model}"

class ElectricCar(Vehicle):
    """Electric car inheriting from Vehicle"""
    
    def __init__(self, brand, model, year, battery_capacity):
        super().__init__(brand, model, year)
        self.battery_capacity = battery_capacity
        self.charge_level = 100
    
    def charge(self, amount):
        """Charge the battery"""
        self.charge_level = min(100, self.charge_level + amount)
        return f"Battery charged to {self.charge_level}%"
    
    def get_range(self):
        """Calculate driving range"""
        return (self.charge_level / 100) * self.battery_capacity * 3  # 3 miles per kWh
    
    def start(self):
        """Override start method"""
        if self.charge_level > 0:
            self.is_running = True
            return f"Electric {self.brand} {self.model} started silently"
        return "Battery is empty! Please charge first."

class Motorcycle(Vehicle):
    """Motorcycle inheriting from Vehicle"""
    
    def __init__(self, brand, model, year, engine_size):
        super().__init__(brand, model, year)
        self.engine_size = engine_size
    
    def wheelie(self):
        """Perform a wheelie"""
        if self.is_running:
            return "Performing a wheelie! 🏍️"
        return "Start the motorcycle first!"
    
    def start(self):
        """Override start method"""
        self.is_running = True
        return f"{self.brand} {self.model} roars to life! 🏍️"

# Polymorphism Example
def start_vehicle(vehicle):
    """Function that works with any Vehicle object"""
    return vehicle.start()

# Composition Example
class Engine:
    """Engine class for composition example"""
    
    def __init__(self, horsepower, fuel_type="gasoline"):
        self.horsepower = horsepower
        self.fuel_type = fuel_type
        self.is_running = False
    
    def start(self):
        self.is_running = True
        return f"{self.horsepower}HP {self.fuel_type} engine started"
    
    def stop(self):
        self.is_running = False
        return "Engine stopped"

class CompositionCar:
    """Car using composition instead of inheritance"""
    
    def __init__(self, brand, model, engine_hp):
        self.brand = brand
        self.model = model
        self.engine = Engine(engine_hp)  # Composition: Car HAS an Engine
        self.speed = 0
    
    def start(self):
        """Start the car by starting its engine"""
        result = self.engine.start()
        return f"{self.brand} {self.model}: {result}"
    
    def accelerate(self, amount):
        """Accelerate if engine is running"""
        if self.engine.is_running:
            self.speed += amount
            return f"Accelerating to {self.speed} mph"
        return "Start the engine first!"

# Properties Example
class Temperature:
    """Temperature class demonstrating properties"""
    
    def __init__(self, celsius=0):
        self._celsius = celsius
    
    @property
    def celsius(self):
        """Get temperature in Celsius"""
        return self._celsius
    
    @celsius.setter
    def celsius(self, value):
        """Set temperature in Celsius with validation"""
        if value < -273.15:
            raise ValueError("Temperature cannot be below absolute zero")
        self._celsius = value
    
    @property
    def fahrenheit(self):
        """Get temperature in Fahrenheit"""
        return (self._celsius * 9/5) + 32
    
    @fahrenheit.setter
    def fahrenheit(self, value):
        """Set temperature using Fahrenheit"""
        self.celsius = (value - 32) * 5/9
    
    @property
    def kelvin(self):
        """Get temperature in Kelvin"""
        return self._celsius + 273.15
    
    def __str__(self):
        return f"{self._celsius:.1f}°C ({self.fahrenheit:.1f}°F)"

# Magic Methods Example
class Point:
    """Point class demonstrating magic methods"""
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __str__(self):
        """String representation for users"""
        return f"({self.x}, {self.y})"
    
    def __repr__(self):
        """String representation for developers"""
        return f"Point({self.x}, {self.y})"
    
    def __add__(self, other):
        """Addition operator"""
        return Point(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        """Subtraction operator"""
        return Point(self.x - other.x, self.y - other.y)
    
    def __eq__(self, other):
        """Equality operator"""
        return self.x == other.x and self.y == other.y
    
    def __len__(self):
        """Distance from origin"""
        return int((self.x ** 2 + self.y ** 2) ** 0.5)
    
    def distance_to(self, other):
        """Calculate distance to another point"""
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

# Class Methods and Static Methods Example
class MathUtils:
    """Utility class demonstrating class and static methods"""
    
    pi = 3.14159  # Class attribute
    
    def __init__(self, name):
        self.name = name
    
    @classmethod
    def create_default(cls):
        """Class method to create default instance"""
        return cls("Default Calculator")
    
    @staticmethod
    def add(a, b):
        """Static method for addition"""
        return a + b
    
    @staticmethod
    def multiply(a, b):
        """Static method for multiplication"""
        return a * b
    
    def circle_area(self, radius):
        """Instance method to calculate circle area"""
        return self.pi * radius ** 2
    
    def __str__(self):
        return f"MathUtils Calculator: {self.name}"

# Demo Functions
def basic_class_demo():
    """Demonstrate basic class usage"""
    print("=== Basic Class Demo ===")
    
    # Create car objects
    my_car = Car("Toyota", "Camry", 2022, "blue")
    your_car = Car("Honda", "Civic", 2021)
    
    print(f"My car: {my_car}")
    print(f"Your car: {your_car}")
    
    # Use methods
    print(my_car.start())
    print(my_car.accelerate(30))
    print(my_car.brake(10))
    print(my_car.stop())

def encapsulation_demo():
    """Demonstrate encapsulation"""
    print("\n=== Encapsulation Demo ===")
    
    account = BankAccount("12345", "Alice Johnson", 1000)
    print(account)
    
    # Use public methods
    account.deposit(500)
    account.withdraw(200)
    print(f"Balance: ${account.get_balance():.2f}")
    
    # Show transaction history
    print("Transaction History:")
    for transaction in account.get_transaction_history():
        print(f"  {transaction}")

def inheritance_demo():
    """Demonstrate inheritance"""
    print("\n=== Inheritance Demo ===")
    
    # Create different vehicle types
    regular_car = Vehicle("Ford", "Focus", 2020)
    electric_car = ElectricCar("Tesla", "Model 3", 2023, 75)
    motorcycle = Motorcycle("Harley", "Sportster", 2022, 883)
    
    vehicles = [regular_car, electric_car, motorcycle]
    
    # Demonstrate polymorphism
    print("Starting all vehicles:")
    for vehicle in vehicles:
        print(f"  {start_vehicle(vehicle)}")
    
    # Electric car specific methods
    print(f"\nElectric car range: {electric_car.get_range():.1f} miles")
    print(electric_car.charge(20))
    
    # Motorcycle specific methods
    print(motorcycle.wheelie())

def composition_demo():
    """Demonstrate composition"""
    print("\n=== Composition Demo ===")
    
    car = CompositionCar("BMW", "X5", 300)
    print(car.start())
    print(car.accelerate(25))

def properties_demo():
    """Demonstrate properties"""
    print("\n=== Properties Demo ===")
    
    temp = Temperature(25)
    print(f"Temperature: {temp}")
    
    # Set using Fahrenheit
    temp.fahrenheit = 100
    print(f"After setting to 100°F: {temp}")
    print(f"Kelvin: {temp.kelvin:.1f}K")

def magic_methods_demo():
    """Demonstrate magic methods"""
    print("\n=== Magic Methods Demo ===")
    
    p1 = Point(3, 4)
    p2 = Point(1, 2)
    
    print(f"Point 1: {p1}")
    print(f"Point 2: {p2}")
    print(f"Addition: {p1 + p2}")
    print(f"Subtraction: {p1 - p2}")
    print(f"Are equal? {p1 == p2}")
    print(f"Distance from origin: {len(p1)}")
    print(f"Distance between points: {p1.distance_to(p2):.2f}")

def class_static_methods_demo():
    """Demonstrate class and static methods"""
    print("\n=== Class and Static Methods Demo ===")
    
    # Create instances
    calc1 = MathUtils("Scientific Calculator")
    calc2 = MathUtils.create_default()  # Using class method
    
    print(calc1)
    print(calc2)
    
    # Use static methods
    print(f"Addition: {MathUtils.add(5, 3)}")
    print(f"Multiplication: {MathUtils.multiply(4, 7)}")
    
    # Use instance method
    print(f"Circle area (radius=5): {calc1.circle_area(5):.2f}")

def main():
    """Run all demonstrations"""
    print("Object-Oriented Programming Demonstration")
    print("=" * 50)
    
    basic_class_demo()
    encapsulation_demo()
    inheritance_demo()
    composition_demo()
    properties_demo()
    magic_methods_demo()
    class_static_methods_demo()
    
    print("\n" + "=" * 50)
    print("OOP demonstration complete!")

if __name__ == "__main__":
    main()
