# Lesson 19: Performance Optimization

## Learning Objectives
By the end of this lesson, you will be able to:
- Profile Python code to identify performance bottlenecks
- Optimize algorithms and data structures for better performance
- Use vectorization and NumPy for numerical computations
- Implement parallel processing with multiprocessing and threading
- Apply memory optimization techniques
- Write efficient, scalable code for data analysis

## Code Profiling and Performance Measurement

### Basic Timing
```python
import time
import timeit
from functools import wraps

def measure_time(func):
    """Decorator to measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__}: {end - start:.6f} seconds")
        return result
    return wrapper

# Using timeit for precise measurements
def slow_sum(n):
    return sum(range(n))

def fast_sum(n):
    return n * (n - 1) // 2

# Compare performance
n = 1000000
slow_time = timeit.timeit(lambda: slow_sum(n), number=10)
fast_time = timeit.timeit(lambda: fast_sum(n), number=10)

print(f"Slow sum: {slow_time:.6f} seconds")
print(f"Fast sum: {fast_time:.6f} seconds")
print(f"Speedup: {slow_time / fast_time:.2f}x")
```

### Advanced Profiling
```python
import cProfile
import pstats
from pstats import SortKey

def profile_function(func, *args, **kwargs):
    """Profile a function and return statistics"""
    profiler = cProfile.Profile()
    profiler.enable()
    
    result = func(*args, **kwargs)
    
    profiler.disable()
    
    # Create stats object
    stats = pstats.Stats(profiler)
    stats.sort_stats(SortKey.CUMULATIVE)
    
    return result, stats

# Memory profiling
from memory_profiler import profile

@profile
def memory_intensive_function():
    # Create large lists
    data1 = [i for i in range(1000000)]
    data2 = [i**2 for i in range(1000000)]
    
    # Process data
    result = [x + y for x, y in zip(data1, data2)]
    
    return result

# Line-by-line profiling
def detailed_analysis():
    """Function for detailed line-by-line analysis"""
    import numpy as np
    
    # Inefficient operations
    data = []
    for i in range(100000):
        data.append(i**2)
    
    # More efficient
    data_np = np.arange(100000) ** 2
    
    return data, data_np
```

## Algorithm Optimization

### Data Structure Selection
```python
import time
from collections import deque, defaultdict
import bisect

def compare_data_structures():
    """Compare performance of different data structures"""
    
    n = 100000
    
    # List vs Deque for append/pop operations
    print("List vs Deque for queue operations:")
    
    # List (inefficient for queue)
    start = time.time()
    lst = []
    for i in range(n):
        lst.append(i)
    for i in range(n):
        lst.pop(0)  # O(n) operation
    list_time = time.time() - start
    
    # Deque (efficient for queue)
    start = time.time()
    dq = deque()
    for i in range(n):
        dq.append(i)
    for i in range(n):
        dq.popleft()  # O(1) operation
    deque_time = time.time() - start
    
    print(f"List: {list_time:.4f}s")
    print(f"Deque: {deque_time:.4f}s")
    print(f"Deque is {list_time/deque_time:.1f}x faster")

# Efficient searching
def binary_search_demo():
    """Demonstrate binary search efficiency"""
    import random
    
    data = sorted(random.randint(1, 1000000) for _ in range(100000))
    target = random.choice(data)
    
    # Linear search
    start = time.time()
    linear_result = target in data
    linear_time = time.time() - start
    
    # Binary search
    start = time.time()
    binary_result = bisect.bisect_left(data, target) < len(data) and data[bisect.bisect_left(data, target)] == target
    binary_time = time.time() - start
    
    print(f"Linear search: {linear_time:.6f}s")
    print(f"Binary search: {binary_time:.6f}s")
    print(f"Binary search is {linear_time/binary_time:.1f}x faster")

# Efficient counting and grouping
def efficient_counting():
    """Demonstrate efficient counting techniques"""
    import random
    from collections import Counter
    
    data = [random.randint(1, 100) for _ in range(100000)]
    
    # Manual counting (slow)
    start = time.time()
    counts_manual = {}
    for item in data:
        counts_manual[item] = counts_manual.get(item, 0) + 1
    manual_time = time.time() - start
    
    # Counter (optimized)
    start = time.time()
    counts_counter = Counter(data)
    counter_time = time.time() - start
    
    print(f"Manual counting: {manual_time:.6f}s")
    print(f"Counter: {counter_time:.6f}s")
    print(f"Counter is {manual_time/counter_time:.1f}x faster")
```

## Vectorization with NumPy

### NumPy vs Pure Python
```python
import numpy as np
import time

def compare_vectorization():
    """Compare vectorized vs loop-based operations"""
    
    size = 1000000
    
    # Pure Python
    data1 = list(range(size))
    data2 = list(range(size, 2*size))
    
    start = time.time()
    result_python = [x + y for x, y in zip(data1, data2)]
    python_time = time.time() - start
    
    # NumPy vectorized
    arr1 = np.arange(size)
    arr2 = np.arange(size, 2*size)
    
    start = time.time()
    result_numpy = arr1 + arr2
    numpy_time = time.time() - start
    
    print(f"Python loops: {python_time:.6f}s")
    print(f"NumPy vectorized: {numpy_time:.6f}s")
    print(f"NumPy is {python_time/numpy_time:.1f}x faster")

# Advanced vectorization techniques
def advanced_vectorization():
    """Advanced NumPy optimization techniques"""
    
    # Broadcasting for memory efficiency
    data = np.random.randn(1000, 1000)
    
    # Inefficient: explicit loops
    start = time.time()
    result_loop = np.zeros_like(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            result_loop[i, j] = data[i, j] ** 2 + np.sin(data[i, j])
    loop_time = time.time() - start
    
    # Efficient: vectorized operations
    start = time.time()
    result_vectorized = data ** 2 + np.sin(data)
    vectorized_time = time.time() - start
    
    print(f"Explicit loops: {loop_time:.6f}s")
    print(f"Vectorized: {vectorized_time:.6f}s")
    print(f"Vectorized is {loop_time/vectorized_time:.1f}x faster")
    
    # Memory-efficient operations
    # Use in-place operations when possible
    large_array = np.random.randn(10000, 10000)
    
    # Memory inefficient
    start = time.time()
    result1 = large_array * 2 + 1  # Creates intermediate arrays
    inefficient_time = time.time() - start
    
    # Memory efficient
    start = time.time()
    large_array *= 2  # In-place multiplication
    large_array += 1  # In-place addition
    efficient_time = time.time() - start
    
    print(f"Memory inefficient: {inefficient_time:.6f}s")
    print(f"Memory efficient: {efficient_time:.6f}s")

# Optimized data analysis functions
def optimized_statistics(data):
    """Optimized statistical calculations"""
    
    # Convert to NumPy if not already
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    # Vectorized calculations
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    
    # Percentiles in one call
    percentiles = np.percentile(data, [25, 50, 75])
    
    # Outlier detection (vectorized)
    z_scores = np.abs((data - mean) / std)
    outliers = data[z_scores > 3]
    
    return {
        'count': n,
        'mean': mean,
        'std': std,
        'q1': percentiles[0],
        'median': percentiles[1],
        'q3': percentiles[2],
        'outliers': outliers
    }
```

## Parallel Processing

### Multiprocessing for CPU-bound Tasks
```python
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import time

def cpu_intensive_task(n):
    """Simulate CPU-intensive computation"""
    result = 0
    for i in range(n):
        result += i ** 2
    return result

def compare_parallel_processing():
    """Compare serial vs parallel processing"""
    
    tasks = [1000000] * 8  # 8 CPU-intensive tasks
    
    # Serial processing
    start = time.time()
    serial_results = [cpu_intensive_task(n) for n in tasks]
    serial_time = time.time() - start
    
    # Parallel processing
    start = time.time()
    with ProcessPoolExecutor() as executor:
        parallel_results = list(executor.map(cpu_intensive_task, tasks))
    parallel_time = time.time() - start
    
    print(f"Serial: {serial_time:.4f}s")
    print(f"Parallel: {parallel_time:.4f}s")
    print(f"Speedup: {serial_time/parallel_time:.2f}x")

# Data processing with multiprocessing
def process_data_chunk(chunk):
    """Process a chunk of data"""
    import numpy as np
    
    # Simulate data processing
    processed = np.array(chunk) ** 2 + np.sin(np.array(chunk))
    return processed.tolist()

def parallel_data_processing(data, chunk_size=10000):
    """Process large dataset in parallel"""
    
    # Split data into chunks
    chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
    
    # Process chunks in parallel
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_data_chunk, chunks))
    
    # Combine results
    return [item for sublist in results for item in sublist]

# Threading for I/O-bound tasks
def io_intensive_task(url):
    """Simulate I/O-intensive task"""
    import time
    time.sleep(1)  # Simulate network delay
    return f"Data from {url}"

def compare_threading():
    """Compare serial vs threaded I/O operations"""
    
    urls = [f"http://api{i}.example.com" for i in range(10)]
    
    # Serial I/O
    start = time.time()
    serial_results = [io_intensive_task(url) for url in urls]
    serial_time = time.time() - start
    
    # Threaded I/O
    start = time.time()
    with ThreadPoolExecutor(max_workers=5) as executor:
        threaded_results = list(executor.map(io_intensive_task, urls))
    threaded_time = time.time() - start
    
    print(f"Serial I/O: {serial_time:.4f}s")
    print(f"Threaded I/O: {threaded_time:.4f}s")
    print(f"Speedup: {serial_time/threaded_time:.2f}x")
```

### Async Programming for I/O
```python
import asyncio
import aiohttp
import time

async def fetch_data_async(session, url):
    """Async function to fetch data"""
    async with session.get(url) as response:
        return await response.text()

async def async_data_processing():
    """Process multiple URLs asynchronously"""
    
    urls = [f"http://httpbin.org/delay/1" for _ in range(10)]
    
    start = time.time()
    
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_data_async(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
    
    async_time = time.time() - start
    
    print(f"Async processing: {async_time:.4f}s")
    print(f"Processed {len(results)} requests")

# Run async function
# asyncio.run(async_data_processing())
```

## Memory Optimization

### Memory-Efficient Data Structures
```python
import sys
from array import array
import numpy as np

def compare_memory_usage():
    """Compare memory usage of different data structures"""
    
    size = 1000000
    
    # Python list
    python_list = list(range(size))
    list_memory = sys.getsizeof(python_list)
    
    # Array (more memory efficient for numbers)
    int_array = array('i', range(size))
    array_memory = sys.getsizeof(int_array)
    
    # NumPy array (most efficient)
    numpy_array = np.arange(size, dtype=np.int32)
    numpy_memory = numpy_array.nbytes
    
    print(f"Python list: {list_memory:,} bytes")
    print(f"Array: {array_memory:,} bytes")
    print(f"NumPy array: {numpy_memory:,} bytes")
    
    print(f"Array is {list_memory/array_memory:.1f}x more memory efficient")
    print(f"NumPy is {list_memory/numpy_memory:.1f}x more memory efficient")

# Generator for memory-efficient processing
def memory_efficient_processing(filename):
    """Process large file without loading into memory"""
    
    def read_and_process():
        with open(filename, 'r') as file:
            for line in file:
                # Process line by line
                yield float(line.strip()) ** 2
    
    # Calculate statistics without storing all data
    count = 0
    total = 0
    sum_squares = 0
    
    for value in read_and_process():
        count += 1
        total += value
        sum_squares += value ** 2
    
    mean = total / count
    variance = (sum_squares / count) - (mean ** 2)
    
    return {
        'count': count,
        'mean': mean,
        'variance': variance
    }

# Memory pooling for frequent allocations
class ObjectPool:
    """Object pool to reduce memory allocation overhead"""
    
    def __init__(self, create_func, reset_func=None, max_size=100):
        self.create_func = create_func
        self.reset_func = reset_func
        self.max_size = max_size
        self.pool = []
    
    def get(self):
        if self.pool:
            obj = self.pool.pop()
            if self.reset_func:
                self.reset_func(obj)
            return obj
        return self.create_func()
    
    def put(self, obj):
        if len(self.pool) < self.max_size:
            self.pool.append(obj)

# Example usage
def create_list():
    return []

def reset_list(lst):
    lst.clear()

list_pool = ObjectPool(create_list, reset_list)

# Use pooled objects
temp_list = list_pool.get()
temp_list.extend([1, 2, 3, 4, 5])
# Process list...
list_pool.put(temp_list)  # Return to pool
```

## Caching and Memoization

### Advanced Caching Strategies
```python
import functools
import time
from collections import OrderedDict
import pickle
import hashlib

class PersistentCache:
    """Cache that persists to disk"""
    
    def __init__(self, cache_dir="cache", max_size=1000):
        import os
        self.cache_dir = cache_dir
        self.max_size = max_size
        
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    def _get_cache_key(self, func_name, args, kwargs):
        """Generate cache key from function arguments"""
        key_data = f"{func_name}_{args}_{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, func_name, args, kwargs):
        """Get cached result"""
        key = self._get_cache_key(func_name, args, kwargs)
        cache_file = f"{self.cache_dir}/{key}.pkl"
        
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
    
    def put(self, func_name, args, kwargs, result):
        """Store result in cache"""
        key = self._get_cache_key(func_name, args, kwargs)
        cache_file = f"{self.cache_dir}/{key}.pkl"
        
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)

def persistent_cache(cache_instance):
    """Decorator for persistent caching"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Try to get from cache
            cached_result = cache_instance.get(func.__name__, args, kwargs)
            if cached_result is not None:
                print(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache_instance.put(func.__name__, args, kwargs, result)
            print(f"Computed and cached {func.__name__}")
            return result
        
        return wrapper
    return decorator

# Time-based cache expiration
class TTLCache:
    """Cache with time-to-live expiration"""
    
    def __init__(self, ttl_seconds=3600):
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.timestamps = {}
    
    def get(self, key):
        if key in self.cache:
            if time.time() - self.timestamps[key] < self.ttl_seconds:
                return self.cache[key]
            else:
                # Expired
                del self.cache[key]
                del self.timestamps[key]
        return None
    
    def put(self, key, value):
        self.cache[key] = value
        self.timestamps[key] = time.time()

# Example: Expensive computation with caching
cache = PersistentCache()

@persistent_cache(cache)
def expensive_computation(n):
    """Simulate expensive computation"""
    print(f"Computing for n={n}...")
    time.sleep(2)  # Simulate work
    return sum(i**2 for i in range(n))

# First call: computed and cached
result1 = expensive_computation(1000)

# Second call: retrieved from cache
result2 = expensive_computation(1000)
```

## Code Optimization Best Practices

### Profiling-Driven Optimization
```python
import cProfile
import pstats
from line_profiler import LineProfiler

class PerformanceOptimizer:
    """Class to help with systematic performance optimization"""
    
    def __init__(self):
        self.profiler = cProfile.Profile()
        self.line_profiler = LineProfiler()
    
    def profile_function(self, func, *args, **kwargs):
        """Profile function execution"""
        
        # Function-level profiling
        self.profiler.enable()
        result = func(*args, **kwargs)
        self.profiler.disable()
        
        # Generate report
        stats = pstats.Stats(self.profiler)
        stats.sort_stats('cumulative')
        
        return result, stats
    
    def compare_implementations(self, implementations, *args, **kwargs):
        """Compare multiple implementations of the same function"""
        
        results = {}
        
        for name, func in implementations.items():
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            
            results[name] = {
                'result': result,
                'time': end_time - start_time
            }
        
        # Find fastest implementation
        fastest = min(results.keys(), key=lambda k: results[k]['time'])
        
        print("Performance Comparison:")
        for name, data in results.items():
            speedup = results[fastest]['time'] / data['time']
            print(f"{name}: {data['time']:.6f}s (speedup: {speedup:.2f}x)")
        
        return results

# Example: Optimizing data processing
def slow_data_processing(data):
    """Slow implementation"""
    result = []
    for item in data:
        if item > 0:
            result.append(item ** 2)
    return result

def fast_data_processing(data):
    """Optimized implementation"""
    import numpy as np
    arr = np.array(data)
    return (arr[arr > 0] ** 2).tolist()

def fastest_data_processing(data):
    """Most optimized implementation"""
    import numpy as np
    arr = np.array(data)
    mask = arr > 0
    arr[mask] **= 2
    return arr[mask].tolist()

# Compare implementations
optimizer = PerformanceOptimizer()
test_data = list(range(-1000, 1000))

implementations = {
    'slow': slow_data_processing,
    'fast': fast_data_processing,
    'fastest': fastest_data_processing
}

results = optimizer.compare_implementations(implementations, test_data)
```

### Memory and CPU Monitoring
```python
import psutil
import os
from contextlib import contextmanager

@contextmanager
def monitor_resources():
    """Context manager to monitor resource usage"""
    
    process = psutil.Process(os.getpid())
    
    # Initial measurements
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    initial_cpu = process.cpu_percent()
    
    start_time = time.time()
    
    try:
        yield
    finally:
        # Final measurements
        end_time = time.time()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        final_cpu = process.cpu_percent()
        
        print(f"Execution time: {end_time - start_time:.4f}s")
        print(f"Memory usage: {final_memory:.2f} MB (change: {final_memory - initial_memory:+.2f} MB)")
        print(f"CPU usage: {final_cpu:.1f}%")

# Usage example
with monitor_resources():
    # Your code here
    large_data = [i**2 for i in range(1000000)]
    result = sum(large_data)
```

## Key Terminology

- **Profiling**: Measuring program performance to identify bottlenecks
- **Vectorization**: Using array operations instead of explicit loops
- **Memoization**: Caching function results to avoid recomputation
- **Multiprocessing**: Using multiple CPU cores for parallel computation
- **Threading**: Concurrent execution within a single process
- **Async Programming**: Non-blocking I/O operations
- **Memory Pool**: Reusing allocated memory to reduce allocation overhead
- **Time Complexity**: How execution time scales with input size
- **Space Complexity**: How memory usage scales with input size

## Looking Ahead

In Lesson 20, we'll learn about:
- **Real-world Data Project**: End-to-end data analysis project
- **Project Planning**: Breaking down complex analysis tasks
- **Data Pipeline**: Building complete data processing workflows
- **Documentation**: Creating professional project documentation
- **Presentation**: Communicating results effectively
