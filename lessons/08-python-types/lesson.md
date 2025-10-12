# Lesson 22: Python Types and Advanced OOP

## Learning Objectives

By the end of this lesson, you will be able to:
- Use type hints to improve code clarity and catch errors
- Understand Python's type system and built-in types
- Implement advanced OOP patterns with proper typing
- Use dataclasses for cleaner data structures
- Apply protocols and abstract base classes effectively
- Debug type-related issues in complex applications

## Introduction to Type Hints

Type hints were introduced in Python 3.5 to help developers write more maintainable code. They don't affect runtime behavior but provide valuable information for IDEs, linters, and other developers.

### Basic Type Hints

```python
def greet(name: str) -> str:
    """Function with type hints"""
    return f"Hello, {name}!"

def calculate_area(length: float, width: float) -> float:
    """Calculate rectangle area"""
    return length * width

# Variable annotations
age: int = 25
price: float = 99.99
is_active: bool = True
```

### Collection Types

```python
from typing import List, Dict, Set, Tuple, Optional, Union

# Lists and dictionaries
numbers: List[int] = [1, 2, 3, 4, 5]
scores: Dict[str, float] = {"Alice": 95.5, "Bob": 87.2}
unique_ids: Set[int] = {1, 2, 3, 4}

# Tuples (fixed size)
coordinates: Tuple[float, float] = (10.5, 20.3)
rgb_color: Tuple[int, int, int] = (255, 128, 0)

# Optional and Union types
def find_user(user_id: int) -> Optional[str]:
    """Returns username or None if not found"""
    users = {1: "Alice", 2: "Bob"}
    return users.get(user_id)

def process_id(user_id: Union[int, str]) -> str:
    """Accept either int or string ID"""
    return str(user_id)
```

## Advanced Type Hints

### Generic Types

```python
from typing import TypeVar, Generic, List

T = TypeVar('T')

class Stack(Generic[T]):
    """Generic stack implementation"""
    
    def __init__(self) -> None:
        self._items: List[T] = []
    
    def push(self, item: T) -> None:
        self._items.append(item)
    
    def pop(self) -> T:
        if not self._items:
            raise IndexError("Stack is empty")
        return self._items.pop()
    
    def peek(self) -> T:
        if not self._items:
            raise IndexError("Stack is empty")
        return self._items[-1]
    
    def is_empty(self) -> bool:
        return len(self._items) == 0

# Usage with specific types
int_stack: Stack[int] = Stack()
str_stack: Stack[str] = Stack()
```

### Callable Types

```python
from typing import Callable

def apply_operation(x: int, y: int, operation: Callable[[int, int], int]) -> int:
    """Apply a binary operation to two integers"""
    return operation(x, y)

def add(a: int, b: int) -> int:
    return a + b

def multiply(a: int, b: int) -> int:
    return a * b

# Usage
result1 = apply_operation(5, 3, add)       # 8
result2 = apply_operation(5, 3, multiply)  # 15
```

## Dataclasses

Dataclasses provide a decorator to automatically generate special methods for classes that primarily store data.

```python
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime

@dataclass
class Person:
    """Simple person dataclass"""
    name: str
    age: int
    email: str
    is_active: bool = True  # Default value

@dataclass
class Product:
    """Product with more complex fields"""
    name: str
    price: float
    category: str
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Called after initialization"""
        if self.price < 0:
            raise ValueError("Price cannot be negative")

@dataclass(frozen=True)  # Immutable dataclass
class Point:
    """Immutable point"""
    x: float
    y: float
    
    def distance_to(self, other: 'Point') -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

# Usage
person = Person("Alice", 30, "alice@email.com")
product = Product("Laptop", 999.99, "Electronics", ["computer", "portable"])
point = Point(3.0, 4.0)
```

## Protocols and Structural Typing

Protocols define interfaces that classes can implement without explicit inheritance.

```python
from typing import Protocol

class Drawable(Protocol):
    """Protocol for drawable objects"""
    
    def draw(self) -> str:
        """Draw the object"""
        ...
    
    def get_area(self) -> float:
        """Get the area of the object"""
        ...

class Circle:
    """Circle class implementing Drawable protocol"""
    
    def __init__(self, radius: float):
        self.radius = radius
    
    def draw(self) -> str:
        return f"Drawing circle with radius {self.radius}"
    
    def get_area(self) -> float:
        return 3.14159 * self.radius ** 2

class Rectangle:
    """Rectangle class implementing Drawable protocol"""
    
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
    
    def draw(self) -> str:
        return f"Drawing rectangle {self.width}x{self.height}"
    
    def get_area(self) -> float:
        return self.width * self.height

def render_shape(shape: Drawable) -> str:
    """Render any drawable shape"""
    return f"{shape.draw()} - Area: {shape.get_area():.2f}"

# Usage - no explicit inheritance needed
circle = Circle(5)
rectangle = Rectangle(4, 6)

print(render_shape(circle))
print(render_shape(rectangle))
```

## Abstract Base Classes

```python
from abc import ABC, abstractmethod
from typing import List

class Animal(ABC):
    """Abstract base class for animals"""
    
    def __init__(self, name: str, species: str):
        self.name = name
        self.species = species
    
    @abstractmethod
    def make_sound(self) -> str:
        """Abstract method - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def move(self) -> str:
        """Abstract method for movement"""
        pass
    
    def get_info(self) -> str:
        """Concrete method available to all subclasses"""
        return f"{self.name} is a {self.species}"

class Dog(Animal):
    """Concrete implementation of Animal"""
    
    def __init__(self, name: str, breed: str):
        super().__init__(name, "Dog")
        self.breed = breed
    
    def make_sound(self) -> str:
        return "Woof!"
    
    def move(self) -> str:
        return "Running on four legs"
    
    def fetch(self) -> str:
        return f"{self.name} is fetching the ball!"

class Bird(Animal):
    """Another concrete implementation"""
    
    def __init__(self, name: str, can_fly: bool = True):
        super().__init__(name, "Bird")
        self.can_fly = can_fly
    
    def make_sound(self) -> str:
        return "Tweet!"
    
    def move(self) -> str:
        return "Flying" if self.can_fly else "Walking"

# Usage
dog = Dog("Buddy", "Golden Retriever")
bird = Bird("Tweety")

animals: List[Animal] = [dog, bird]
for animal in animals:
    print(f"{animal.get_info()}: {animal.make_sound()} - {animal.move()}")
```

## Advanced Class Features

### Metaclasses

```python
class SingletonMeta(type):
    """Metaclass for singleton pattern"""
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class DatabaseConnection(metaclass=SingletonMeta):
    """Singleton database connection"""
    
    def __init__(self, host: str = "localhost"):
        self.host = host
        self.connected = False
    
    def connect(self) -> str:
        self.connected = True
        return f"Connected to {self.host}"

# Usage - both variables reference the same instance
db1 = DatabaseConnection("server1")
db2 = DatabaseConnection("server2")  # host is ignored
print(db1 is db2)  # True
```

### Descriptors

```python
class ValidatedAttribute:
    """Descriptor for validated attributes"""
    
    def __init__(self, validator: Callable[[any], bool], error_message: str):
        self.validator = validator
        self.error_message = error_message
        self.name = None
    
    def __set_name__(self, owner, name):
        self.name = name
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj.__dict__.get(self.name)
    
    def __set__(self, obj, value):
        if not self.validator(value):
            raise ValueError(f"{self.error_message}: {value}")
        obj.__dict__[self.name] = value

class Person:
    """Person class with validated attributes"""
    
    name = ValidatedAttribute(
        lambda x: isinstance(x, str) and len(x) > 0,
        "Name must be a non-empty string"
    )
    
    age = ValidatedAttribute(
        lambda x: isinstance(x, int) and 0 <= x <= 150,
        "Age must be an integer between 0 and 150"
    )
    
    email = ValidatedAttribute(
        lambda x: isinstance(x, str) and "@" in x,
        "Email must contain @"
    )
    
    def __init__(self, name: str, age: int, email: str):
        self.name = name
        self.age = age
        self.email = email
```

## Type Checking Tools

### Using mypy

```python
# Install: pip install mypy
# Run: mypy your_file.py

def process_numbers(numbers: List[int]) -> float:
    """Calculate average of numbers"""
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)

# This will cause a mypy error:
# result = process_numbers(["1", "2", "3"])  # Error: List[str] not List[int]

# This is correct:
result = process_numbers([1, 2, 3])
```

### Runtime Type Checking

```python
from typing import get_type_hints
import inspect

def validate_types(func):
    """Decorator to validate function argument types at runtime"""
    def wrapper(*args, **kwargs):
        # Get type hints
        hints = get_type_hints(func)
        
        # Get function signature
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        # Validate each argument
        for name, value in bound_args.arguments.items():
            if name in hints:
                expected_type = hints[name]
                if not isinstance(value, expected_type):
                    raise TypeError(
                        f"Argument '{name}' must be {expected_type.__name__}, "
                        f"got {type(value).__name__}"
                    )
        
        return func(*args, **kwargs)
    return wrapper

@validate_types
def divide(a: float, b: float) -> float:
    """Divide two numbers with runtime type checking"""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

# This will work:
result = divide(10.0, 2.0)

# This will raise TypeError:
# result = divide("10", 2)
```

## Advanced Patterns

### Builder Pattern with Types

```python
from typing import Optional, Self

class QueryBuilder:
    """SQL query builder with method chaining"""
    
    def __init__(self):
        self._select: List[str] = []
        self._from: Optional[str] = None
        self._where: List[str] = []
        self._order_by: List[str] = []
    
    def select(self, *columns: str) -> Self:
        """Add columns to SELECT clause"""
        self._select.extend(columns)
        return self
    
    def from_table(self, table: str) -> Self:
        """Set FROM clause"""
        self._from = table
        return self
    
    def where(self, condition: str) -> Self:
        """Add WHERE condition"""
        self._where.append(condition)
        return self
    
    def order_by(self, column: str, desc: bool = False) -> Self:
        """Add ORDER BY clause"""
        order = f"{column} DESC" if desc else column
        self._order_by.append(order)
        return self
    
    def build(self) -> str:
        """Build the final SQL query"""
        if not self._from:
            raise ValueError("FROM clause is required")
        
        query_parts = []
        
        # SELECT
        select_clause = "SELECT " + (", ".join(self._select) if self._select else "*")
        query_parts.append(select_clause)
        
        # FROM
        query_parts.append(f"FROM {self._from}")
        
        # WHERE
        if self._where:
            query_parts.append("WHERE " + " AND ".join(self._where))
        
        # ORDER BY
        if self._order_by:
            query_parts.append("ORDER BY " + ", ".join(self._order_by))
        
        return " ".join(query_parts)

# Usage with method chaining
query = (QueryBuilder()
         .select("name", "email", "age")
         .from_table("users")
         .where("age > 18")
         .where("is_active = true")
         .order_by("name")
         .build())

print(query)
```

### Factory Pattern with Generics

```python
from typing import TypeVar, Type, Dict, Callable
from abc import ABC, abstractmethod

T = TypeVar('T', bound='Shape')

class Shape(ABC):
    """Abstract shape class"""
    
    @abstractmethod
    def area(self) -> float:
        pass
    
    @abstractmethod
    def perimeter(self) -> float:
        pass

class Circle(Shape):
    def __init__(self, radius: float):
        self.radius = radius
    
    def area(self) -> float:
        return 3.14159 * self.radius ** 2
    
    def perimeter(self) -> float:
        return 2 * 3.14159 * self.radius

class Rectangle(Shape):
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
    
    def area(self) -> float:
        return self.width * self.height
    
    def perimeter(self) -> float:
        return 2 * (self.width + self.height)

class ShapeFactory:
    """Type-safe shape factory"""
    
    _creators: Dict[str, Callable[..., Shape]] = {}
    
    @classmethod
    def register(cls, shape_type: str, creator: Callable[..., T]) -> None:
        """Register a shape creator"""
        cls._creators[shape_type] = creator
    
    @classmethod
    def create(cls, shape_type: str, **kwargs) -> Shape:
        """Create a shape of the specified type"""
        if shape_type not in cls._creators:
            raise ValueError(f"Unknown shape type: {shape_type}")
        
        return cls._creators[shape_type](**kwargs)

# Register shape creators
ShapeFactory.register("circle", lambda radius: Circle(radius))
ShapeFactory.register("rectangle", lambda width, height: Rectangle(width, height))

# Usage
circle = ShapeFactory.create("circle", radius=5)
rectangle = ShapeFactory.create("rectangle", width=4, height=6)
```

## Best Practices

### 1. Use Type Hints Consistently
```python
# Good: Consistent typing
def process_user_data(
    user_id: int,
    data: Dict[str, Any],
    validate: bool = True
) -> Optional[User]:
    pass

# Avoid: Inconsistent typing
def process_user_data(user_id, data: Dict[str, Any], validate=True):
    pass
```

### 2. Use Union Types Sparingly
```python
# Good: Specific types
def get_user_by_id(user_id: int) -> Optional[User]:
    pass

def get_user_by_email(email: str) -> Optional[User]:
    pass

# Avoid: Overuse of Union
def get_user(identifier: Union[int, str, UUID, Email]) -> Optional[User]:
    pass
```

### 3. Leverage Protocols for Flexibility
```python
# Good: Protocol-based design
class Serializable(Protocol):
    def to_dict(self) -> Dict[str, Any]: ...

def save_to_file(obj: Serializable, filename: str) -> None:
    with open(filename, 'w') as f:
        json.dump(obj.to_dict(), f)

# Any class with to_dict() method works
```

## Summary

Python's type system and advanced OOP features provide powerful tools for building robust, maintainable applications:

- **Type Hints**: Improve code clarity and catch errors early
- **Dataclasses**: Simplify data-focused class creation
- **Protocols**: Enable structural typing and flexible interfaces
- **Abstract Base Classes**: Enforce contracts in inheritance hierarchies
- **Advanced Patterns**: Implement sophisticated design patterns with type safety
- **Type Checking**: Use tools like mypy for static analysis

These features help create more professional, maintainable Python code that scales well in team environments and complex applications.

## Next Steps

This completes our comprehensive Python course! You now have the skills to:
- Build complex applications with proper OOP design
- Use type hints for better code quality
- Implement advanced patterns and architectures
- Work with data analysis libraries professionally
- Create maintainable, scalable Python projects

Continue practicing these concepts in real projects and explore specialized libraries for your domain of interest!
