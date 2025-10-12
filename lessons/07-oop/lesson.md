# Lesson 05: Object-Oriented Programming

## Learning Objectives

By the end of this lesson, you will be able to:
- Understand the concepts of classes and objects
- Create and use classes with attributes and methods
- Implement inheritance and polymorphism
- Use encapsulation and data hiding
- Apply OOP principles to solve real-world problems

## What is Object-Oriented Programming?

Object-Oriented Programming (OOP) is a programming paradigm that organizes code around objects rather than functions. Think of objects as real-world entities that have:
- **Attributes** (characteristics/properties)
- **Methods** (behaviors/actions)

### Real-World Analogy

Consider a **Car**:
- **Attributes**: color, brand, model, year, speed
- **Methods**: start(), stop(), accelerate(), brake()

In Python, we can model this car as a class and create specific car objects from it.

## Classes and Objects

### Defining a Class

```python
class Car:
    """A simple car class"""
    
    def __init__(self, brand, model, year):
        """Constructor method - called when creating a new car"""
        self.brand = brand
        self.model = model
        self.year = year
        self.speed = 0
        self.is_running = False
    
    def start(self):
        """Start the car"""
        self.is_running = True
        print(f"The {self.brand} {self.model} is now running!")
    
    def accelerate(self, amount):
        """Increase speed"""
        if self.is_running:
            self.speed += amount
            print(f"Speed increased to {self.speed} mph")
        else:
            print("Start the car first!")
    
    def stop(self):
        """Stop the car"""
        self.speed = 0
        self.is_running = False
        print(f"The {self.brand} {self.model} has stopped")
```

### Creating Objects

```python
# Create car objects
my_car = Car("Toyota", "Camry", 2022)
your_car = Car("Honda", "Civic", 2021)

# Use the objects
my_car.start()
my_car.accelerate(30)
my_car.stop()
```

## Key OOP Concepts

### 1. Encapsulation

Encapsulation means bundling data and methods together and controlling access to them.

```python
class BankAccount:
    def __init__(self, account_number, initial_balance=0):
        self.account_number = account_number
        self._balance = initial_balance  # Protected attribute
        self.__pin = "1234"  # Private attribute
    
    def deposit(self, amount):
        if amount > 0:
            self._balance += amount
            return True
        return False
    
    def withdraw(self, amount):
        if 0 < amount <= self._balance:
            self._balance -= amount
            return True
        return False
    
    def get_balance(self):
        return self._balance
    
    def __str__(self):
        return f"Account {self.account_number}: ${self._balance:.2f}"
```

### 2. Inheritance

Inheritance allows a class to inherit attributes and methods from another class.

```python
class Vehicle:
    """Base class for all vehicles"""
    
    def __init__(self, brand, model, year):
        self.brand = brand
        self.model = model
        self.year = year
        self.is_running = False
    
    def start(self):
        self.is_running = True
        print(f"{self.brand} {self.model} started")
    
    def stop(self):
        self.is_running = False
        print(f"{self.brand} {self.model} stopped")

class Car(Vehicle):
    """Car class inherits from Vehicle"""
    
    def __init__(self, brand, model, year, doors):
        super().__init__(brand, model, year)  # Call parent constructor
        self.doors = doors
        self.speed = 0
    
    def accelerate(self, amount):
        if self.is_running:
            self.speed += amount
            print(f"Car speed: {self.speed} mph")

class Motorcycle(Vehicle):
    """Motorcycle class inherits from Vehicle"""
    
    def __init__(self, brand, model, year, engine_size):
        super().__init__(brand, model, year)
        self.engine_size = engine_size
    
    def wheelie(self):
        if self.is_running:
            print("Doing a wheelie!")
        else:
            print("Start the motorcycle first!")
```

### 3. Polymorphism

Polymorphism allows objects of different classes to be treated as objects of a common base class.

```python
def start_vehicle(vehicle):
    """This function works with any Vehicle object"""
    vehicle.start()

# Create different vehicles
car = Car("Toyota", "Camry", 2022, 4)
motorcycle = Motorcycle("Harley", "Sportster", 2021, 883)

# Polymorphism in action
vehicles = [car, motorcycle]
for vehicle in vehicles:
    start_vehicle(vehicle)  # Same method call, different behavior
```

## Class Methods and Static Methods

```python
class MathUtils:
    pi = 3.14159  # Class attribute
    
    def __init__(self, name):
        self.name = name  # Instance attribute
    
    @classmethod
    def create_default(cls):
        """Class method - creates instance with default values"""
        return cls("Default Calculator")
    
    @staticmethod
    def add(a, b):
        """Static method - doesn't need class or instance"""
        return a + b
    
    def circle_area(self, radius):
        """Instance method"""
        return self.pi * radius ** 2

# Usage
calc1 = MathUtils("My Calculator")
calc2 = MathUtils.create_default()  # Using class method

result = MathUtils.add(5, 3)  # Using static method
area = calc1.circle_area(5)   # Using instance method
```

## Properties and Getters/Setters

```python
class Temperature:
    def __init__(self, celsius=0):
        self._celsius = celsius
    
    @property
    def celsius(self):
        """Getter for celsius"""
        return self._celsius
    
    @celsius.setter
    def celsius(self, value):
        """Setter for celsius with validation"""
        if value < -273.15:
            raise ValueError("Temperature cannot be below absolute zero")
        self._celsius = value
    
    @property
    def fahrenheit(self):
        """Computed property"""
        return (self._celsius * 9/5) + 32
    
    @fahrenheit.setter
    def fahrenheit(self, value):
        """Set celsius from fahrenheit"""
        self.celsius = (value - 32) * 5/9

# Usage
temp = Temperature(25)
print(f"Celsius: {temp.celsius}")
print(f"Fahrenheit: {temp.fahrenheit}")

temp.fahrenheit = 100  # Sets celsius automatically
print(f"New Celsius: {temp.celsius}")
```

## Magic Methods (Dunder Methods)

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __str__(self):
        """String representation for users"""
        return f"Point({self.x}, {self.y})"
    
    def __repr__(self):
        """String representation for developers"""
        return f"Point(x={self.x}, y={self.y})"
    
    def __add__(self, other):
        """Addition operator"""
        return Point(self.x + other.x, self.y + other.y)
    
    def __eq__(self, other):
        """Equality operator"""
        return self.x == other.x and self.y == other.y
    
    def __len__(self):
        """Length (distance from origin)"""
        return (self.x ** 2 + self.y ** 2) ** 0.5

# Usage
p1 = Point(3, 4)
p2 = Point(1, 2)

print(p1)           # Uses __str__
print(repr(p1))     # Uses __repr__
p3 = p1 + p2        # Uses __add__
print(p1 == p2)     # Uses __eq__
print(len(p1))      # Uses __len__
```

## Composition vs Inheritance

Sometimes composition (has-a relationship) is better than inheritance (is-a relationship):

```python
class Engine:
    def __init__(self, horsepower):
        self.horsepower = horsepower
        self.is_running = False
    
    def start(self):
        self.is_running = True
        print("Engine started")
    
    def stop(self):
        self.is_running = False
        print("Engine stopped")

class Car:
    def __init__(self, brand, model):
        self.brand = brand
        self.model = model
        self.engine = Engine(200)  # Composition: Car HAS an Engine
    
    def start(self):
        print(f"Starting {self.brand} {self.model}")
        self.engine.start()
    
    def stop(self):
        print(f"Stopping {self.brand} {self.model}")
        self.engine.stop()
```

## Best Practices

### 1. Use Clear Class Names
```python
# Good
class CustomerAccount:
    pass

# Avoid
class CA:
    pass
```

### 2. Keep Methods Focused
```python
class Order:
    def calculate_total(self):
        """Single responsibility: calculate total"""
        return sum(item.price * item.quantity for item in self.items)
    
    def apply_discount(self, discount_percent):
        """Single responsibility: apply discount"""
        self.discount = discount_percent
```

### 3. Use Inheritance Appropriately
```python
# Good: Clear is-a relationship
class Dog(Animal):
    pass

# Questionable: Forced inheritance
class Rectangle(Color):  # Rectangle is not a Color
    pass
```

## Common Patterns

### 1. Builder Pattern
```python
class Pizza:
    def __init__(self):
        self.size = None
        self.toppings = []
    
    def set_size(self, size):
        self.size = size
        return self  # Return self for chaining
    
    def add_topping(self, topping):
        self.toppings.append(topping)
        return self
    
    def build(self):
        return f"{self.size} pizza with {', '.join(self.toppings)}"

# Usage
pizza = (Pizza()
         .set_size("Large")
         .add_topping("pepperoni")
         .add_topping("mushrooms")
         .build())
```

### 2. Factory Pattern
```python
class ShapeFactory:
    @staticmethod
    def create_shape(shape_type, **kwargs):
        if shape_type == "circle":
            return Circle(kwargs.get("radius", 1))
        elif shape_type == "rectangle":
            return Rectangle(kwargs.get("width", 1), kwargs.get("height", 1))
        else:
            raise ValueError(f"Unknown shape type: {shape_type}")

# Usage
circle = ShapeFactory.create_shape("circle", radius=5)
rectangle = ShapeFactory.create_shape("rectangle", width=3, height=4)
```

## When to Use OOP

**Use OOP when:**
- You have complex data with related behaviors
- You need to model real-world entities
- You want to reuse code through inheritance
- You need to organize large codebases

**Consider alternatives when:**
- You have simple, procedural tasks
- You're doing mathematical computations
- You're writing small scripts
- Functional programming fits better

## Summary

Object-Oriented Programming provides a powerful way to organize and structure your code by modeling real-world entities as objects. Key concepts include:

- **Classes and Objects**: Templates and instances
- **Encapsulation**: Bundling data and methods
- **Inheritance**: Reusing code through parent-child relationships
- **Polymorphism**: Same interface, different implementations
- **Composition**: Building complex objects from simpler ones

Understanding OOP is crucial for writing maintainable, scalable Python applications and working with many Python libraries and frameworks.

## Next Steps

In the next lesson, we'll explore Python's built-in data structures (lists, dictionaries, sets, tuples) and see how they can be used effectively in object-oriented designs.
