# Lesson 18 Solutions: Advanced Python Concepts

import functools
import time
from contextlib import contextmanager
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import threading
import weakref

print("=== GUIDED EXERCISE SOLUTIONS ===\n")

# Exercise 1: Decorators and Context Managers
print("Exercise 1: Decorators and Context Managers")
print("-" * 50)

def timing_decorator(func):
    """Decorator to measure function execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def cache_decorator(maxsize=128):
    """Caching decorator with LRU eviction"""
    def decorator(func):
        cache = {}
        access_order = deque()
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = str(args) + str(sorted(kwargs.items()))
            
            if key in cache:
                # Move to end (most recently used)
                access_order.remove(key)
                access_order.append(key)
                return cache[key]
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Evict if cache is full
            if len(cache) >= maxsize:
                oldest_key = access_order.popleft()
                del cache[oldest_key]
            
            # Add to cache
            cache[key] = result
            access_order.append(key)
            
            return result
        
        wrapper.cache_info = lambda: {"size": len(cache), "maxsize": maxsize}
        wrapper.cache_clear = lambda: (cache.clear(), access_order.clear())
        
        return wrapper
    return decorator

def validate_input(validator):
    """Decorator for input validation"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not validator(*args, **kwargs):
                raise ValueError(f"Invalid input for {func.__name__}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

@contextmanager
def data_processor(name, setup_func=None, cleanup_func=None):
    """Context manager for data processing with setup/cleanup"""
    print(f"Setting up data processor: {name}")
    
    if setup_func:
        setup_result = setup_func()
    else:
        setup_result = None
    
    start_time = time.time()
    
    try:
        yield setup_result
    except Exception as e:
        print(f"Error in {name}: {e}")
        raise
    finally:
        end_time = time.time()
        print(f"Cleaning up {name} (took {end_time - start_time:.4f}s)")
        
        if cleanup_func:
            cleanup_func(setup_result)

# Demo decorators and context managers
@timing_decorator
@cache_decorator(maxsize=5)
@validate_input(lambda x: isinstance(x, int) and x >= 0)
def factorial(n):
    """Calculate factorial with decorators"""
    if n <= 1:
        return 1
    return n * factorial(n - 1)

print("Testing decorated factorial function:")
print(f"factorial(5) = {factorial(5)}")
print(f"factorial(5) = {factorial(5)}")  # Should use cache
print(f"Cache info: {factorial.cache_info()}")

# Test context manager
def setup_data():
    print("  Loading data...")
    return {"data": [1, 2, 3, 4, 5]}

def cleanup_data(data):
    print(f"  Saving results for {len(data['data'])} items")

with data_processor("analysis", setup_data, cleanup_data) as data:
    print(f"  Processing {data}")
    time.sleep(0.1)

# Exercise 2: Generators and Iterators
print("\n" + "="*50)
print("Exercise 2: Generators and Iterators")
print("-" * 50)

def data_generator(source, batch_size=1000):
    """Generator for processing large datasets"""
    if isinstance(source, str):  # File path
        try:
            with open(source, 'r') as file:
                batch = []
                for line in file:
                    batch.append(line.strip())
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []
                if batch:  # Yield remaining items
                    yield batch
        except FileNotFoundError:
            print(f"File {source} not found")
    elif hasattr(source, '__iter__'):  # Iterable
        batch = []
        for item in source:
            batch.append(item)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

def pipeline_processor(*processors):
    """Generator pipeline for data processing"""
    def process_data(data):
        result = data
        for processor in processors:
            if hasattr(processor, '__call__'):
                result = processor(result)
            elif hasattr(processor, 'process'):
                result = processor.process(result)
        return result
    
    return process_data

class DataIterator:
    """Custom iterator with filtering and transformation"""
    
    def __init__(self, data, filter_func=None, transform_func=None):
        self.data = data
        self.filter_func = filter_func or (lambda x: True)
        self.transform_func = transform_func or (lambda x: x)
        self.index = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        while self.index < len(self.data):
            item = self.data[self.index]
            self.index += 1
            
            if self.filter_func(item):
                return self.transform_func(item)
        
        raise StopIteration

# Demo generators and iterators
print("Testing data generator:")
sample_data = list(range(25))
for i, batch in enumerate(data_generator(sample_data, batch_size=7)):
    print(f"  Batch {i}: {batch}")

print("\nTesting custom iterator:")
numbers = list(range(10))
even_squares = DataIterator(
    numbers,
    filter_func=lambda x: x % 2 == 0,
    transform_func=lambda x: x ** 2
)

print("  Even squares:", list(even_squares))

# Exercise 3: Object-Oriented Design
print("\n" + "="*50)
print("Exercise 3: Object-Oriented Design")
print("-" * 50)

class DataProcessor(ABC):
    """Abstract base class for data processors"""
    
    def __init__(self, name):
        self.name = name
        self.processed_count = 0
        self._observers = []
    
    @abstractmethod
    def process(self, data):
        """Process data - must be implemented by subclasses"""
        pass
    
    def add_observer(self, observer):
        """Add observer for notifications"""
        self._observers.append(observer)
    
    def notify_observers(self, event, data):
        """Notify all observers of an event"""
        for observer in self._observers:
            observer.notify(event, data)
    
    def reset(self):
        """Reset processor state"""
        self.processed_count = 0
    
    def __str__(self):
        return f"{self.__class__.__name__}(name='{self.name}', processed={self.processed_count})"

class FilterProcessor(DataProcessor):
    """Filter data based on condition"""
    
    def __init__(self, name, condition):
        super().__init__(name)
        self.condition = condition
    
    def process(self, data):
        filtered_data = [item for item in data if self.condition(item)]
        self.processed_count += len(data)
        self.notify_observers("filtered", {"original": len(data), "filtered": len(filtered_data)})
        return filtered_data

class TransformProcessor(DataProcessor):
    """Transform data using function"""
    
    def __init__(self, name, transform_func):
        super().__init__(name)
        self.transform_func = transform_func
    
    def process(self, data):
        transformed_data = [self.transform_func(item) for item in data]
        self.processed_count += len(data)
        self.notify_observers("transformed", {"count": len(transformed_data)})
        return transformed_data

class AggregateProcessor(DataProcessor):
    """Aggregate data using function"""
    
    def __init__(self, name, aggregate_func):
        super().__init__(name)
        self.aggregate_func = aggregate_func
    
    def process(self, data):
        if not data:
            return None
        
        result = self.aggregate_func(data)
        self.processed_count += len(data)
        self.notify_observers("aggregated", {"result": result})
        return result

class ProcessingObserver:
    """Observer for processing events"""
    
    def __init__(self, name):
        self.name = name
    
    def notify(self, event, data):
        print(f"  Observer {self.name}: {event} event with data {data}")

# Demo OOP design
print("Testing OOP design:")

# Create processors
filter_proc = FilterProcessor("positive_filter", lambda x: x > 0)
transform_proc = TransformProcessor("square", lambda x: x ** 2)
aggregate_proc = AggregateProcessor("sum", sum)

# Add observer
observer = ProcessingObserver("logger")
filter_proc.add_observer(observer)
transform_proc.add_observer(observer)
aggregate_proc.add_observer(observer)

# Process data through pipeline
data = [-2, -1, 0, 1, 2, 3, 4, 5]
print(f"Original data: {data}")

filtered_data = filter_proc.process(data)
print(f"Filtered data: {filtered_data}")

transformed_data = transform_proc.process(filtered_data)
print(f"Transformed data: {transformed_data}")

result = aggregate_proc.process(transformed_data)
print(f"Final result: {result}")

print("\n=== INDEPENDENT EXERCISE SOLUTIONS ===\n")

# Exercise 4: Data Processing Framework
print("Exercise 4: Data Processing Framework")
print("-" * 50)

class ProcessingPipeline:
    """Framework for chaining data processors"""
    
    def __init__(self, name):
        self.name = name
        self.processors = []
        self.error_handlers = []
    
    def add_processor(self, processor):
        """Add processor to pipeline"""
        self.processors.append(processor)
        return self
    
    def add_error_handler(self, handler):
        """Add error handler"""
        self.error_handlers.append(handler)
        return self
    
    def process(self, data):
        """Process data through pipeline"""
        result = data
        
        for i, processor in enumerate(self.processors):
            try:
                print(f"  Step {i+1}: {processor.name}")
                result = processor.process(result)
                
                if result is None:
                    print(f"    Warning: Processor {processor.name} returned None")
                    break
                    
            except Exception as e:
                print(f"    Error in {processor.name}: {e}")
                
                # Try error handlers
                handled = False
                for handler in self.error_handlers:
                    if handler.can_handle(e, processor):
                        result = handler.handle(e, processor, result)
                        handled = True
                        break
                
                if not handled:
                    raise
        
        return result

class ErrorHandler:
    """Base error handler"""
    
    def can_handle(self, error, processor):
        """Check if this handler can handle the error"""
        return True
    
    def handle(self, error, processor, data):
        """Handle the error"""
        print(f"    Handled error: {error}")
        return data  # Return original data

# Demo processing framework
print("Testing processing framework:")

pipeline = ProcessingPipeline("data_analysis")
pipeline.add_processor(FilterProcessor("non_zero", lambda x: x != 0))
pipeline.add_processor(TransformProcessor("absolute", abs))
pipeline.add_processor(AggregateProcessor("average", lambda x: sum(x) / len(x) if x else 0))
pipeline.add_error_handler(ErrorHandler())

test_data = [-3, -2, -1, 0, 1, 2, 3]
print(f"Input data: {test_data}")

result = pipeline.process(test_data)
print(f"Pipeline result: {result}")

# Exercise 5: Custom Data Structures
print("\n" + "="*50)
print("Exercise 5: Custom Data Structures")
print("-" * 50)

class LRUCache:
    """Least Recently Used cache implementation"""
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.access_order = deque()
    
    def get(self, key):
        """Get value by key"""
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key, value):
        """Put key-value pair"""
        if key in self.cache:
            # Update existing key
            self.cache[key] = value
            self.access_order.remove(key)
            self.access_order.append(key)
        else:
            # Add new key
            if len(self.cache) >= self.capacity:
                # Remove least recently used
                oldest_key = self.access_order.popleft()
                del self.cache[oldest_key]
            
            self.cache[key] = value
            self.access_order.append(key)
    
    def size(self):
        """Get current size"""
        return len(self.cache)
    
    def keys(self):
        """Get all keys in access order"""
        return list(self.access_order)

class PriorityQueue:
    """Simple priority queue implementation"""
    
    def __init__(self):
        self.items = []
    
    def push(self, item, priority):
        """Add item with priority"""
        self.items.append((priority, item))
        self.items.sort(key=lambda x: x[0])  # Sort by priority
    
    def pop(self):
        """Remove and return highest priority item"""
        if self.items:
            return self.items.pop(0)[1]  # Return item, not priority
        raise IndexError("pop from empty priority queue")
    
    def peek(self):
        """Return highest priority item without removing"""
        if self.items:
            return self.items[0][1]
        return None
    
    def is_empty(self):
        """Check if queue is empty"""
        return len(self.items) == 0
    
    def size(self):
        """Get queue size"""
        return len(self.items)

# Demo custom data structures
print("Testing LRU Cache:")
cache = LRUCache(3)

cache.put("a", 1)
cache.put("b", 2)
cache.put("c", 3)
print(f"  After adding a,b,c: keys = {cache.keys()}")

print(f"  Get 'a': {cache.get('a')}")
print(f"  After accessing 'a': keys = {cache.keys()}")

cache.put("d", 4)  # Should evict 'b'
print(f"  After adding 'd': keys = {cache.keys()}")

print("\nTesting Priority Queue:")
pq = PriorityQueue()

pq.push("low priority task", 3)
pq.push("high priority task", 1)
pq.push("medium priority task", 2)

print(f"  Queue size: {pq.size()}")
print(f"  Peek: {pq.peek()}")

while not pq.is_empty():
    print(f"  Pop: {pq.pop()}")

print("\n" + "="*50)
print("Advanced Python exercise solutions complete!")
print("Key concepts demonstrated:")
print("- Decorators for timing, caching, and validation")
print("- Context managers for resource management")
print("- Generators and iterators for memory-efficient processing")
print("- Object-oriented design with abstract base classes")
print("- Observer pattern for event notification")
print("- Processing pipeline framework with error handling")
print("- Custom data structures (LRU cache, priority queue)")
