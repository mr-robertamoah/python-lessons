# Lesson 22 Exercises: Python Types and Advanced OOP

## Guided Exercises (Do with Instructor)

### Exercise 1: Basic Type Hints
**Goal**: Add type hints to improve code clarity

**Tasks**:
1. Add type hints to function parameters and return types
2. Use Optional and Union types appropriately
3. Add type hints to variables and collections

```python
# Add type hints to these functions
def calculate_discount(price, discount_percent):
    return price * (1 - discount_percent / 100)

def find_max_value(numbers):
    if not numbers:
        return None
    return max(numbers)

def format_user_info(user_data):
    name = user_data.get("name", "Unknown")
    age = user_data.get("age", 0)
    return f"{name} ({age} years old)"
```

---

### Exercise 2: Dataclasses
**Goal**: Create clean data structures using dataclasses

**Tasks**:
1. Convert regular classes to dataclasses
2. Use field() for complex default values
3. Implement frozen dataclasses for immutable data

```python
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime

# Convert this regular class to a dataclass
class Product:
    def __init__(self, name, price, category, tags=None):
        self.name = name
        self.price = price
        self.category = category
        self.tags = tags or []
        self.created_at = datetime.now()
    
    def __str__(self):
        return f"{self.name} - ${self.price}"
```

---

### Exercise 3: Protocols and Structural Typing
**Goal**: Define interfaces using protocols

**Tasks**:
1. Create protocols for common interfaces
2. Implement classes that satisfy protocols
3. Use protocols in function parameters

```python
from typing import Protocol

# Define protocols for these interfaces
class Drawable(Protocol):
    # Define methods that drawable objects should have
    pass

class Saveable(Protocol):
    # Define methods for objects that can be saved
    pass

# Implement classes that satisfy these protocols
```

---

## Independent Exercises (Try on Your Own)

### Exercise 4: Generic Data Structures
**Goal**: Create type-safe generic data structures

**Requirements**:
1. Implement a generic Stack class
2. Create a generic Repository pattern
3. Build a type-safe Cache system
4. Add proper type hints throughout

**Features to implement**:
- Generic Stack with push, pop, peek operations
- Repository with CRUD operations for any type
- LRU Cache with generic key-value pairs
- Type-safe error handling

---

### Exercise 5: Advanced Class Patterns
**Goal**: Implement sophisticated OOP patterns with types

**Requirements**:
1. Create a Builder pattern with method chaining
2. Implement Factory pattern with type safety
3. Build Observer pattern with proper typing
4. Add Singleton pattern using metaclasses

**Classes to create**:
- QueryBuilder for SQL-like operations
- ShapeFactory for creating geometric shapes
- EventSystem with typed observers
- ConfigManager as singleton

---

### Exercise 6: Type-Safe API Client
**Goal**: Build a REST API client with full type safety

**Requirements**:
1. Define response models using dataclasses
2. Create typed HTTP methods
3. Implement error handling with custom exceptions
4. Add request/response validation

**Features**:
- User, Post, Comment models
- GET, POST, PUT, DELETE operations
- Proper error types and handling
- JSON serialization/deserialization

---

## Challenge Exercises (If You Finish Early)

### Challenge 1: ORM-like System
**Goal**: Build a mini ORM with type safety

**Requirements**:
1. Define model base class with metaclass
2. Implement field descriptors with validation
3. Create query builder with type hints
4. Add relationship handling

### Challenge 2: Plugin Architecture
**Goal**: Create extensible plugin system

**Requirements**:
1. Define plugin interface using protocols
2. Implement plugin discovery and loading
3. Create type-safe plugin registry
4. Build example plugins with proper typing

---

## Real-World Application Exercises

### Exercise 7: E-commerce System with Types
**Goal**: Build comprehensive e-commerce backend

**Requirements**:
1. Product catalog with inheritance hierarchy
2. Shopping cart with type-safe operations
3. Order processing with state management
4. Payment system with multiple providers

**Models to create**:
- Product hierarchy (Physical, Digital, Subscription)
- User accounts with roles and permissions
- Order workflow with status tracking
- Payment processing with provider abstraction

---

### Exercise 8: Data Pipeline Framework
**Goal**: Create type-safe data processing pipeline

**Requirements**:
1. Define processor interfaces using protocols
2. Implement various data transformers
3. Create pipeline builder with type checking
4. Add monitoring and error handling

**Components**:
- Data source connectors (CSV, JSON, API)
- Transformation processors (filter, map, aggregate)
- Output sinks (file, database, API)
- Pipeline orchestration with type safety

---

## Advanced Type System Exercises

### Exercise 9: Custom Type System
**Goal**: Extend Python's type system

**Tasks**:
1. Create custom generic types
2. Implement type validators
3. Build runtime type checking decorators
4. Create domain-specific type aliases

### Exercise 10: Metaclass Magic
**Goal**: Use metaclasses for advanced patterns

**Tasks**:
1. Create automatic property generation
2. Implement field validation metaclass
3. Build registry pattern with metaclasses
4. Add method interception capabilities

---

## Testing and Validation

### Exercise 11: Type-Safe Testing
**Goal**: Write tests that validate type safety

**Tasks**:
1. Test generic classes with different types
2. Validate protocol implementations
3. Test error conditions with proper types
4. Use mypy in CI/CD pipeline

### Exercise 12: Runtime Validation
**Goal**: Implement runtime type checking

**Tasks**:
1. Create validation decorators
2. Build schema validation system
3. Implement API request/response validation
4. Add performance monitoring for type checks

---

## Self-Assessment Checklist

After completing these exercises, you should be able to:

- [ ] Add comprehensive type hints to functions and classes
- [ ] Use generic types and type variables effectively
- [ ] Create and use dataclasses for clean data structures
- [ ] Define interfaces using protocols
- [ ] Implement abstract base classes properly
- [ ] Use advanced typing features (Callable, Literal, etc.)
- [ ] Create type-safe design patterns
- [ ] Build runtime type validation systems
- [ ] Use mypy for static type checking
- [ ] Debug type-related issues effectively

## Type Checking Best Practices

### 1. Start Simple, Add Complexity Gradually
```python
# Start with basic types
def process_data(data: list) -> dict:
    pass

# Evolve to more specific types
def process_data(data: List[Dict[str, Any]]) -> Dict[str, float]:
    pass
```

### 2. Use Type Aliases for Complex Types
```python
# Good: Clear type aliases
UserId = int
UserData = Dict[str, Any]
UserList = List[UserData]

def get_users() -> UserList:
    pass

# Avoid: Inline complex types everywhere
def get_users() -> List[Dict[str, Any]]:
    pass
```

### 3. Leverage Protocols for Flexibility
```python
# Good: Protocol-based design
class Renderable(Protocol):
    def render(self) -> str: ...

def display_item(item: Renderable) -> None:
    print(item.render())

# Avoid: Rigid inheritance requirements
class Renderable(ABC):
    @abstractmethod
    def render(self) -> str: ...
```

## Common Type System Pitfalls

### 1. Overusing Any
```python
# Bad: Too much Any
def process(data: Any) -> Any:
    pass

# Good: Specific types
def process(data: Dict[str, str]) -> List[str]:
    pass
```

### 2. Ignoring Optional Types
```python
# Bad: Ignoring None possibility
def get_user(id: int) -> User:
    # Might return None!
    pass

# Good: Explicit Optional
def get_user(id: int) -> Optional[User]:
    pass
```

### 3. Complex Union Types
```python
# Bad: Too complex
def process(data: Union[str, int, List[str], Dict[str, Any]]) -> Any:
    pass

# Good: Separate functions or protocols
def process_string(data: str) -> str: pass
def process_number(data: int) -> int: pass
```

## Git Reminder

Save your work:
1. Create `lesson-22-python-types` folder in your repository
2. Save your typed implementations as `.py` files
3. Include mypy configuration file
4. Add type checking to your workflow
5. Commit and push:
   ```bash
   git add .
   git commit -m "Complete Lesson 22: Python Types and Advanced OOP"
   git push
   ```

## Course Completion

Congratulations! You've completed the comprehensive Python programming and data analysis course. You now have:

- **Solid Programming Foundation**: Variables, control flow, functions, OOP
- **Data Analysis Skills**: NumPy, Pandas, statistics, visualization
- **Professional Development**: Testing, version control, virtual environments
- **Advanced Techniques**: Type hints, design patterns, performance optimization
- **Real-World Applications**: ML pipelines, data sources, feature engineering

Continue building projects and exploring specialized libraries in your areas of interest!
