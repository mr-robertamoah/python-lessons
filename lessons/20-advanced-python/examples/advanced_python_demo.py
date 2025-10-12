#!/usr/bin/env python3
"""
Advanced Python Concepts Demo
Decorators, generators, OOP, and more
"""

import functools
import time
from contextlib import contextmanager
from abc import ABC, abstractmethod
from collections import defaultdict
import threading

# Decorators Demo
def timing_decorator(func):
    """Decorator to measure function execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def cache_decorator(maxsize=128):
    """Simple caching decorator"""
    def decorator(func):
        cache = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(sorted(kwargs.items()))
            
            if key in cache:
                print(f"Cache hit for {func.__name__}")
                return cache[key]
            
            result = func(*args, **kwargs)
            
            if len(cache) >= maxsize:
                # Remove oldest entry
                cache.pop(next(iter(cache)))
            
            cache[key] = result
            print(f"Cache miss for {func.__name__}")
            return result
        
        wrapper.cache_info = lambda: f"Cache size: {len(cache)}"
        return wrapper
    return decorator

@timing_decorator
@cache_decorator(maxsize=10)
def fibonacci(n):
    """Calculate fibonacci number (inefficient for demo)"""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Context Managers Demo
@contextmanager
def data_processor(name):
    """Context manager for data processing"""
    print(f"Starting data processing: {name}")
    start_time = time.time()
    
    try:
        yield f"processor_{name}"
    except Exception as e:
        print(f"Error in {name}: {e}")
        raise
    finally:
        end_time = time.time()
        print(f"Finished {name} in {end_time - start_time:.4f} seconds")

class DatabaseConnection:
    """Context manager class example"""
    
    def __init__(self, db_name):
        self.db_name = db_name
        self.connection = None
    
    def __enter__(self):
        print(f"Connecting to {self.db_name}")
        self.connection = f"connection_to_{self.db_name}"
        return self.connection
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"Closing connection to {self.db_name}")
        if exc_type:
            print(f"Exception occurred: {exc_val}")
        return False  # Don't suppress exceptions

# Generators Demo
def data_generator(n):
    """Generate data points one at a time"""
    for i in range(n):
        # Simulate expensive computation
        time.sleep(0.01)
        yield {"id": i, "value": i ** 2, "category": "A" if i % 2 == 0 else "B"}

def file_reader(filename):
    """Generator to read file line by line"""
    try:
        with open(filename, 'r') as file:
            for line_num, line in enumerate(file, 1):
                yield line_num, line.strip()
    except FileNotFoundError:
        print(f"File {filename} not found")

def pipeline_processor(*generators):
    """Chain multiple generators together"""
    for gen in generators:
        yield from gen

# Iterator Class Demo
class DataIterator:
    """Custom iterator for data processing"""
    
    def __init__(self, data, batch_size=3):
        self.data = data
        self.batch_size = batch_size
        self.index = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index >= len(self.data):
            raise StopIteration
        
        batch = self.data[self.index:self.index + self.batch_size]
        self.index += self.batch_size
        return batch

# Object-Oriented Design Demo
class DataProcessor(ABC):
    """Abstract base class for data processors"""
    
    def __init__(self, name):
        self.name = name
        self.processed_count = 0
    
    @abstractmethod
    def process(self, data):
        """Process data - must be implemented by subclasses"""
        pass
    
    def reset(self):
        """Reset processor state"""
        self.processed_count = 0
    
    def __str__(self):
        return f"{self.__class__.__name__}(name={self.name}, processed={self.processed_count})"

class FilterProcessor(DataProcessor):
    """Filter data based on condition"""
    
    def __init__(self, name, condition):
        super().__init__(name)
        self.condition = condition
    
    def process(self, data):
        filtered_data = [item for item in data if self.condition(item)]
        self.processed_count += len(data)
        return filtered_data

class TransformProcessor(DataProcessor):
    """Transform data using function"""
    
    def __init__(self, name, transform_func):
        super().__init__(name)
        self.transform_func = transform_func
    
    def process(self, data):
        transformed_data = [self.transform_func(item) for item in data]
        self.processed_count += len(data)
        return transformed_data

# Property and Descriptor Demo
class ValidatedAttribute:
    """Descriptor for validated attributes"""
    
    def __init__(self, validator=None, default=None):
        self.validator = validator
        self.default = default
        self.name = None
    
    def __set_name__(self, owner, name):
        self.name = name
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj.__dict__.get(self.name, self.default)
    
    def __set__(self, obj, value):
        if self.validator and not self.validator(value):
            raise ValueError(f"Invalid value for {self.name}: {value}")
        obj.__dict__[self.name] = value

class Customer:
    """Customer class with validated attributes"""
    
    name = ValidatedAttribute(
        validator=lambda x: isinstance(x, str) and len(x) > 0,
        default=""
    )
    
    age = ValidatedAttribute(
        validator=lambda x: isinstance(x, int) and 0 <= x <= 150,
        default=0
    )
    
    email = ValidatedAttribute(
        validator=lambda x: isinstance(x, str) and "@" in x,
        default=""
    )
    
    def __init__(self, name="", age=0, email=""):
        self.name = name
        self.age = age
        self.email = email
    
    @property
    def is_adult(self):
        return self.age >= 18
    
    def __repr__(self):
        return f"Customer(name='{self.name}', age={self.age}, email='{self.email}')"

# Metaclass Demo
class SingletonMeta(type):
    """Metaclass for singleton pattern"""
    
    _instances = {}
    _lock = threading.Lock()
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class DatabaseManager(metaclass=SingletonMeta):
    """Singleton database manager"""
    
    def __init__(self):
        self.connections = {}
        print("DatabaseManager initialized")
    
    def get_connection(self, db_name):
        if db_name not in self.connections:
            self.connections[db_name] = f"connection_to_{db_name}"
        return self.connections[db_name]

def demo_decorators():
    """Demonstrate decorators"""
    print("=== Decorators Demo ===")
    
    # Test fibonacci with caching
    print("First call:")
    result1 = fibonacci(10)
    print(f"fibonacci(10) = {result1}")
    
    print("\nSecond call (should use cache):")
    result2 = fibonacci(10)
    print(f"fibonacci(10) = {result2}")
    
    print(f"Cache info: {fibonacci.cache_info()}")

def demo_context_managers():
    """Demonstrate context managers"""
    print("\n=== Context Managers Demo ===")
    
    # Function-based context manager
    with data_processor("customer_analysis") as processor:
        print(f"Using {processor}")
        time.sleep(0.1)
    
    # Class-based context manager
    with DatabaseConnection("customer_db") as conn:
        print(f"Using {conn}")

def demo_generators():
    """Demonstrate generators"""
    print("\n=== Generators Demo ===")
    
    # Simple generator
    print("Data generator:")
    for i, data_point in enumerate(data_generator(5)):
        print(f"  {i}: {data_point}")
    
    # Iterator class
    print("\nBatch iterator:")
    data = list(range(10))
    iterator = DataIterator(data, batch_size=3)
    
    for batch in iterator:
        print(f"  Batch: {batch}")

def demo_oop():
    """Demonstrate object-oriented programming"""
    print("\n=== OOP Demo ===")
    
    # Create processors
    filter_proc = FilterProcessor("even_filter", lambda x: x % 2 == 0)
    transform_proc = TransformProcessor("square", lambda x: x ** 2)
    
    # Process data
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    filtered_data = filter_proc.process(data)
    print(f"Filtered data: {filtered_data}")
    print(f"Filter processor: {filter_proc}")
    
    transformed_data = transform_proc.process(filtered_data)
    print(f"Transformed data: {transformed_data}")
    print(f"Transform processor: {transform_proc}")

def demo_descriptors():
    """Demonstrate descriptors and properties"""
    print("\n=== Descriptors Demo ===")
    
    # Create customer with validation
    try:
        customer = Customer("Alice Johnson", 25, "alice@email.com")
        print(f"Created: {customer}")
        print(f"Is adult: {customer.is_adult}")
        
        # Try invalid data
        customer.age = -5  # Should raise error
    except ValueError as e:
        print(f"Validation error: {e}")

def demo_metaclasses():
    """Demonstrate metaclasses"""
    print("\n=== Metaclasses Demo ===")
    
    # Create multiple instances - should be same object
    db1 = DatabaseManager()
    db2 = DatabaseManager()
    
    print(f"db1 is db2: {db1 is db2}")
    print(f"db1 id: {id(db1)}")
    print(f"db2 id: {id(db2)}")

if __name__ == "__main__":
    demo_decorators()
    demo_context_managers()
    demo_generators()
    demo_oop()
    demo_descriptors()
    demo_metaclasses()
