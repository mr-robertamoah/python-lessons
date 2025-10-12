# Lesson 18 Exercises: Advanced Python Concepts

## Guided Exercises (Do with Instructor)

### Exercise 1: Decorators and Context Managers
**Goal**: Master advanced Python patterns

**Tasks**:
1. Create timing decorator
2. Build caching decorator
3. Implement custom context manager
4. Use decorators for data validation

```python
import functools
import time
from contextlib import contextmanager

def timing_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Implement timing logic
        pass
    return wrapper

@contextmanager
def data_processor():
    # Implement context manager
    pass
```

---

### Exercise 2: Generators and Iterators
**Goal**: Implement memory-efficient data processing

**Tasks**:
1. Create data generator functions
2. Build custom iterator classes
3. Use generators for large datasets
4. Implement pipeline processing

```python
def data_generator(filename):
    # Yield data one record at a time
    pass

class DataIterator:
    def __init__(self, data):
        # Initialize iterator
        pass
    
    def __iter__(self):
        return self
    
    def __next__(self):
        # Implement iteration logic
        pass
```

---

### Exercise 3: Object-Oriented Design
**Goal**: Build robust class hierarchies

**Tasks**:
1. Create abstract base classes
2. Implement inheritance and polymorphism
3. Use properties and descriptors
4. Apply design patterns

---

## Independent Exercises (Try on Your Own)

### Exercise 4: Data Processing Framework
**Goal**: Build reusable data processing components

**Tasks**:
1. Create abstract processor base class
2. Implement specific processors (filter, transform, aggregate)
3. Build processing pipeline
4. Add error handling and logging
5. Create configuration system

---

### Exercise 5: Custom Data Structures
**Goal**: Implement specialized data structures

**Tasks**:
1. Build LRU cache implementation
2. Create priority queue
3. Implement trie for text processing
4. Build graph data structure
5. Add comprehensive testing

---

### Exercise 6: Metaclasses and Descriptors
**Goal**: Master advanced Python features

**Tasks**:
1. Create validation descriptors
2. Build automatic property generation
3. Implement singleton metaclass
4. Create ORM-like field definitions
5. Add type checking and validation

---

## Challenge Exercises (If You Finish Early)

### Challenge 1: Async Data Processing
**Goal**: Build asynchronous processing system

**Tasks**:
1. Implement async data fetchers
2. Create concurrent processing pipeline
3. Handle async error management
4. Build rate limiting system
5. Add monitoring and metrics

### Challenge 2: Plugin Architecture
**Goal**: Create extensible system

**Tasks**:
1. Design plugin interface
2. Implement plugin discovery
3. Create plugin manager
4. Add configuration and lifecycle management
5. Build example plugins

---

## Self-Assessment Checklist

After completing these exercises, you should be able to:

- [ ] Create and use decorators effectively
- [ ] Implement context managers
- [ ] Build generators and iterators
- [ ] Design object-oriented systems
- [ ] Use metaclasses and descriptors
- [ ] Implement async programming patterns
- [ ] Create reusable frameworks

## Git Reminder

Save your work:
```bash
git add .
git commit -m "Complete Lesson 18: Advanced Python"
git push
```
