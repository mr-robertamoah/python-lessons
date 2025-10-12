# Lesson 18: Advanced Python Concepts

## Learning Objectives
By the end of this lesson, you will be able to:
- Use iterators and generators for memory-efficient programming
- Create and use decorators for code enhancement
- Implement context managers for resource management
- Apply object-oriented programming principles
- Handle errors gracefully with advanced exception handling
- Use advanced data structures and algorithms

## Iterators and Generators

### Understanding Iterators
```python
# Iterator protocol
class NumberIterator:
    def __init__(self, max_num):
        self.max_num = max_num
        self.current = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current < self.max_num:
            self.current += 1
            return self.current
        raise StopIteration

# Usage
numbers = NumberIterator(5)
for num in numbers:
    print(num)  # 1, 2, 3, 4, 5

# Built-in iterators
data = [1, 2, 3, 4, 5]
iterator = iter(data)
print(next(iterator))  # 1
print(next(iterator))  # 2
```

### Generators for Memory Efficiency
```python
# Generator function
def fibonacci_generator(n):
    """Generate Fibonacci sequence up to n numbers"""
    a, b = 0, 1
    count = 0
    while count < n:
        yield a
        a, b = b, a + b
        count += 1

# Memory efficient - generates values on demand
fib = fibonacci_generator(10)
for num in fib:
    print(num)

# Generator expressions
squares = (x**2 for x in range(1000000))  # Memory efficient
squares_list = [x**2 for x in range(1000000)]  # Memory intensive

# Data processing with generators
def process_large_file(filename):
    """Process large file line by line"""
    with open(filename, 'r') as file:
        for line in file:
            # Process each line without loading entire file
            yield line.strip().upper()

# Pipeline of generators
def read_numbers(filename):
    with open(filename, 'r') as file:
        for line in file:
            yield float(line.strip())

def filter_positive(numbers):
    for num in numbers:
        if num > 0:
            yield num

def square_numbers(numbers):
    for num in numbers:
        yield num ** 2

# Chain generators together
# result = square_numbers(filter_positive(read_numbers('data.txt')))
```

## Decorators

### Basic Decorators
```python
import time
import functools
from datetime import datetime

# Simple decorator
def timer(func):
    """Decorator to measure function execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(1)
    return "Done"

# Decorator with parameters
def retry(max_attempts=3, delay=1):
    """Decorator to retry function on failure"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise e
                    print(f"Attempt {attempt + 1} failed: {e}")
                    time.sleep(delay)
        return wrapper
    return decorator

@retry(max_attempts=3, delay=0.5)
def unreliable_function():
    import random
    if random.random() < 0.7:
        raise ValueError("Random failure")
    return "Success"
```

### Advanced Decorators for Data Analysis
```python
def validate_input(input_type=None, min_length=None, max_length=None):
    """Decorator to validate function inputs"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Validate first argument if specified
            if args and input_type:
                if not isinstance(args[0], input_type):
                    raise TypeError(f"Expected {input_type}, got {type(args[0])}")
                
                if hasattr(args[0], '__len__'):
                    length = len(args[0])
                    if min_length and length < min_length:
                        raise ValueError(f"Input too short: {length} < {min_length}")
                    if max_length and length > max_length:
                        raise ValueError(f"Input too long: {length} > {max_length}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def cache_result(maxsize=128):
    """Decorator to cache function results"""
    def decorator(func):
        cache = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = str(args) + str(sorted(kwargs.items()))
            
            if key in cache:
                print(f"Cache hit for {func.__name__}")
                return cache[key]
            
            result = func(*args, **kwargs)
            
            # Manage cache size
            if len(cache) >= maxsize:
                # Remove oldest entry
                oldest_key = next(iter(cache))
                del cache[oldest_key]
            
            cache[key] = result
            return result
        
        wrapper.cache_info = lambda: {'hits': 0, 'misses': 0, 'size': len(cache)}
        wrapper.cache_clear = lambda: cache.clear()
        
        return wrapper
    return decorator

@validate_input(input_type=list, min_length=1)
@cache_result(maxsize=50)
@timer
def calculate_statistics(data):
    """Calculate statistics with validation, caching, and timing"""
    import statistics
    return {
        'mean': statistics.mean(data),
        'median': statistics.median(data),
        'stdev': statistics.stdev(data) if len(data) > 1 else 0
    }
```

## Context Managers

### Built-in Context Managers
```python
# File handling
with open('data.txt', 'r') as file:
    content = file.read()
# File automatically closed

# Database connections
import sqlite3
with sqlite3.connect('database.db') as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users")
    results = cursor.fetchall()
# Connection automatically closed
```

### Custom Context Managers
```python
import time
from contextlib import contextmanager

class Timer:
    """Context manager for timing code blocks"""
    
    def __init__(self, name="Operation"):
        self.name = name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        print(f"Starting {self.name}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        duration = end_time - self.start_time
        print(f"{self.name} completed in {duration:.4f} seconds")
        
        if exc_type:
            print(f"Exception occurred: {exc_val}")
        
        return False  # Don't suppress exceptions

# Usage
with Timer("Data Processing"):
    # Simulate work
    time.sleep(1)
    result = sum(range(1000000))

# Context manager using decorator
@contextmanager
def database_transaction(connection):
    """Context manager for database transactions"""
    transaction = connection.begin()
    try:
        yield connection
        transaction.commit()
        print("Transaction committed")
    except Exception as e:
        transaction.rollback()
        print(f"Transaction rolled back: {e}")
        raise

# Data analysis context manager
@contextmanager
def temporary_setting(obj, attr, new_value):
    """Temporarily change an object's attribute"""
    old_value = getattr(obj, attr)
    setattr(obj, attr, new_value)
    try:
        yield obj
    finally:
        setattr(obj, attr, old_value)

# Example usage with pandas
import pandas as pd

df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

with temporary_setting(pd.options.display, 'max_rows', 2):
    print(df)  # Only shows 2 rows
print(df)  # Back to normal display
```

## Object-Oriented Programming

### Classes for Data Analysis
```python
class DataAnalyzer:
    """Class for comprehensive data analysis"""
    
    def __init__(self, data=None):
        self._data = data
        self._results = {}
    
    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, value):
        if not hasattr(value, '__iter__'):
            raise ValueError("Data must be iterable")
        self._data = value
        self._results = {}  # Clear cached results
    
    def __len__(self):
        return len(self._data) if self._data else 0
    
    def __repr__(self):
        return f"DataAnalyzer(samples={len(self)})"
    
    def __str__(self):
        if not self._data:
            return "DataAnalyzer: No data loaded"
        return f"DataAnalyzer: {len(self)} samples, mean={self.mean():.2f}"
    
    @cache_result()
    def mean(self):
        """Calculate mean with caching"""
        if not self._data:
            return None
        return sum(self._data) / len(self._data)
    
    @cache_result()
    def std(self):
        """Calculate standard deviation"""
        if not self._data or len(self._data) < 2:
            return None
        
        mean_val = self.mean()
        variance = sum((x - mean_val) ** 2 for x in self._data) / (len(self._data) - 1)
        return variance ** 0.5
    
    def summary(self):
        """Generate comprehensive summary"""
        if not self._data:
            return "No data available"
        
        return {
            'count': len(self._data),
            'mean': self.mean(),
            'std': self.std(),
            'min': min(self._data),
            'max': max(self._data)
        }

# Inheritance example
class AdvancedDataAnalyzer(DataAnalyzer):
    """Extended analyzer with additional methods"""
    
    def __init__(self, data=None, confidence_level=0.95):
        super().__init__(data)
        self.confidence_level = confidence_level
    
    def confidence_interval(self):
        """Calculate confidence interval for mean"""
        if not self._data or len(self._data) < 2:
            return None
        
        import math
        from scipy import stats
        
        mean_val = self.mean()
        std_val = self.std()
        n = len(self._data)
        
        # t-distribution critical value
        alpha = 1 - self.confidence_level
        t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
        
        margin_error = t_crit * (std_val / math.sqrt(n))
        
        return {
            'lower': mean_val - margin_error,
            'upper': mean_val + margin_error,
            'confidence_level': self.confidence_level
        }
    
    def outliers(self, method='iqr'):
        """Detect outliers using specified method"""
        if not self._data:
            return []
        
        if method == 'iqr':
            sorted_data = sorted(self._data)
            n = len(sorted_data)
            q1 = sorted_data[n//4]
            q3 = sorted_data[3*n//4]
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            return [x for x in self._data if x < lower_bound or x > upper_bound]
        
        elif method == 'zscore':
            mean_val = self.mean()
            std_val = self.std()
            
            if std_val == 0:
                return []
            
            return [x for x in self._data if abs(x - mean_val) / std_val > 3]
        
        else:
            raise ValueError("Method must be 'iqr' or 'zscore'")

# Usage
analyzer = AdvancedDataAnalyzer([1, 2, 3, 4, 5, 100])  # 100 is outlier
print(analyzer)
print(analyzer.summary())
print(f"Outliers: {analyzer.outliers()}")
print(f"95% CI: {analyzer.confidence_interval()}")
```

## Advanced Exception Handling

### Custom Exceptions
```python
class DataAnalysisError(Exception):
    """Base exception for data analysis errors"""
    pass

class InsufficientDataError(DataAnalysisError):
    """Raised when there's not enough data for analysis"""
    
    def __init__(self, required, actual):
        self.required = required
        self.actual = actual
        super().__init__(f"Need at least {required} data points, got {actual}")

class InvalidDataError(DataAnalysisError):
    """Raised when data is invalid for analysis"""
    pass

def robust_analysis(data, min_samples=10):
    """Perform analysis with comprehensive error handling"""
    
    # Input validation
    if not data:
        raise InvalidDataError("Data cannot be empty")
    
    if not all(isinstance(x, (int, float)) for x in data):
        raise InvalidDataError("All data points must be numeric")
    
    if len(data) < min_samples:
        raise InsufficientDataError(min_samples, len(data))
    
    try:
        # Perform analysis
        mean_val = sum(data) / len(data)
        
        # Check for potential issues
        if any(x != x for x in data):  # Check for NaN
            raise InvalidDataError("Data contains NaN values")
        
        return {
            'mean': mean_val,
            'count': len(data),
            'status': 'success'
        }
    
    except ZeroDivisionError:
        raise DataAnalysisError("Division by zero in calculation")
    
    except OverflowError:
        raise DataAnalysisError("Numerical overflow in calculation")

# Exception handling with logging
import logging

def safe_data_processing(data_sources):
    """Process multiple data sources with error recovery"""
    
    results = {}
    errors = {}
    
    for source_name, data in data_sources.items():
        try:
            result = robust_analysis(data)
            results[source_name] = result
            logging.info(f"Successfully processed {source_name}")
        
        except InsufficientDataError as e:
            error_msg = f"Insufficient data in {source_name}: {e}"
            errors[source_name] = error_msg
            logging.warning(error_msg)
        
        except InvalidDataError as e:
            error_msg = f"Invalid data in {source_name}: {e}"
            errors[source_name] = error_msg
            logging.error(error_msg)
        
        except Exception as e:
            error_msg = f"Unexpected error in {source_name}: {e}"
            errors[source_name] = error_msg
            logging.critical(error_msg)
    
    return {
        'results': results,
        'errors': errors,
        'success_rate': len(results) / len(data_sources) if data_sources else 0
    }
```

## Advanced Data Structures

### Custom Collections
```python
from collections import defaultdict, Counter, deque
import heapq

class LRUCache:
    """Least Recently Used Cache implementation"""
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.access_order = deque()
    
    def get(self, key):
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key, value):
        if key in self.cache:
            # Update existing
            self.cache[key] = value
            self.access_order.remove(key)
            self.access_order.append(key)
        else:
            # Add new
            if len(self.cache) >= self.capacity:
                # Remove least recently used
                lru_key = self.access_order.popleft()
                del self.cache[lru_key]
            
            self.cache[key] = value
            self.access_order.append(key)

class DataBuffer:
    """Circular buffer for streaming data"""
    
    def __init__(self, size):
        self.size = size
        self.buffer = [None] * size
        self.index = 0
        self.count = 0
    
    def add(self, item):
        self.buffer[self.index] = item
        self.index = (self.index + 1) % self.size
        self.count = min(self.count + 1, self.size)
    
    def get_all(self):
        if self.count < self.size:
            return self.buffer[:self.count]
        else:
            # Return in correct order
            return self.buffer[self.index:] + self.buffer[:self.index]
    
    def mean(self):
        data = [x for x in self.get_all() if x is not None]
        return sum(data) / len(data) if data else 0

# Priority queue for data processing
class DataProcessor:
    """Process data items by priority"""
    
    def __init__(self):
        self.heap = []
        self.counter = 0
    
    def add_task(self, priority, data, task_id=None):
        if task_id is None:
            task_id = self.counter
            self.counter += 1
        
        # Use negative priority for max heap behavior
        heapq.heappush(self.heap, (-priority, task_id, data))
    
    def process_next(self):
        if not self.heap:
            return None
        
        neg_priority, task_id, data = heapq.heappop(self.heap)
        priority = -neg_priority
        
        return {
            'priority': priority,
            'task_id': task_id,
            'data': data
        }
    
    def size(self):
        return len(self.heap)

# Usage examples
cache = LRUCache(3)
cache.put('a', 1)
cache.put('b', 2)
cache.put('c', 3)
print(cache.get('a'))  # 1
cache.put('d', 4)  # Evicts 'b'
print(cache.get('b'))  # None

buffer = DataBuffer(5)
for i in range(10):
    buffer.add(i)
print(buffer.get_all())  # [5, 6, 7, 8, 9]
print(buffer.mean())     # 7.0
```

## Functional Programming Concepts

### Higher-Order Functions
```python
from functools import reduce, partial
from operator import add, mul

# Map, filter, reduce patterns
def process_data_functional(data, operations):
    """Process data using functional programming approach"""
    
    result = data
    
    for operation in operations:
        if operation['type'] == 'map':
            result = list(map(operation['func'], result))
        elif operation['type'] == 'filter':
            result = list(filter(operation['func'], result))
        elif operation['type'] == 'reduce':
            result = reduce(operation['func'], result, operation.get('initial', 0))
    
    return result

# Example operations
operations = [
    {'type': 'filter', 'func': lambda x: x > 0},
    {'type': 'map', 'func': lambda x: x ** 2},
    {'type': 'reduce', 'func': add, 'initial': 0}
]

data = [-2, -1, 0, 1, 2, 3]
result = process_data_functional(data, operations)
print(result)  # Sum of squares of positive numbers

# Partial functions for data analysis
def calculate_percentile(data, percentile):
    """Calculate specific percentile of data"""
    sorted_data = sorted(data)
    index = int(len(sorted_data) * percentile / 100)
    return sorted_data[min(index, len(sorted_data) - 1)]

# Create specialized functions
median = partial(calculate_percentile, percentile=50)
q1 = partial(calculate_percentile, percentile=25)
q3 = partial(calculate_percentile, percentile=75)

data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(f"Median: {median(data)}")  # 5
print(f"Q1: {q1(data)}")         # 2
print(f"Q3: {q3(data)}")         # 7
```

## Metaclasses and Advanced Class Features

### Descriptors
```python
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

class DataModel:
    """Data model with validated attributes"""
    
    # Validators
    positive_number = ValidatedAttribute(
        validator=lambda x: isinstance(x, (int, float)) and x > 0,
        default=1.0
    )
    
    non_empty_string = ValidatedAttribute(
        validator=lambda x: isinstance(x, str) and len(x) > 0,
        default="unnamed"
    )
    
    def __init__(self, name, value):
        self.non_empty_string = name
        self.positive_number = value

# Usage
model = DataModel("test", 5.0)
print(model.non_empty_string)  # "test"
print(model.positive_number)   # 5.0

# This will raise ValueError
# model.positive_number = -1
```

## Key Terminology

- **Iterator**: Object that implements the iterator protocol (__iter__ and __next__)
- **Generator**: Function that uses yield to produce values lazily
- **Decorator**: Function that modifies or enhances another function
- **Context Manager**: Object that defines runtime context for executing code blocks
- **Descriptor**: Object that defines how attribute access is handled
- **Metaclass**: Class whose instances are classes themselves
- **Higher-Order Function**: Function that takes other functions as arguments
- **Closure**: Function that captures variables from its enclosing scope

## Looking Ahead

In Lesson 19, we'll learn about:
- **Performance Optimization**: Profiling and speeding up Python code
- **Memory Management**: Understanding and optimizing memory usage
- **Parallel Processing**: Using multiprocessing and threading
- **Cython and NumPy**: Accelerating numerical computations
- **Best Practices**: Writing efficient, maintainable code
