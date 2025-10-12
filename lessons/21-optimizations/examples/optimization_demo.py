#!/usr/bin/env python3
"""
Performance Optimization Demo
Profiling, benchmarking, and optimization techniques
"""

import time
import timeit
import cProfile
import pstats
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import multiprocessing
import numpy as np

def profiling_demo():
    """Demonstrate profiling techniques"""
    print("=== Profiling Demo ===")
    
    def slow_function():
        """Intentionally slow function for profiling"""
        total = 0
        for i in range(100000):
            total += i ** 2
        
        # Simulate some string operations
        text = ""
        for i in range(1000):
            text += str(i)
        
        return total, len(text)
    
    # Profile with cProfile
    print("Profiling with cProfile:")
    profiler = cProfile.Profile()
    profiler.enable()
    
    result = slow_function()
    
    profiler.disable()
    
    # Print stats
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # Top 10 functions
    
    # Benchmark with timeit
    print(f"\nBenchmarking with timeit:")
    execution_time = timeit.timeit(slow_function, number=10)
    print(f"Average time per call: {execution_time/10:.4f} seconds")

def algorithm_optimization_demo():
    """Demonstrate algorithm optimization"""
    print("\n=== Algorithm Optimization Demo ===")
    
    # Inefficient vs efficient search
    def linear_search(data, target):
        """O(n) linear search"""
        for i, item in enumerate(data):
            if item == target:
                return i
        return -1
    
    def binary_search(data, target):
        """O(log n) binary search - requires sorted data"""
        left, right = 0, len(data) - 1
        
        while left <= right:
            mid = (left + right) // 2
            if data[mid] == target:
                return mid
            elif data[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return -1
    
    # Test with large dataset
    data = list(range(100000))
    target = 75000
    
    # Time linear search
    start_time = time.time()
    linear_result = linear_search(data, target)
    linear_time = time.time() - start_time
    
    # Time binary search
    start_time = time.time()
    binary_result = binary_search(data, target)
    binary_time = time.time() - start_time
    
    print(f"Linear search: found at index {linear_result} in {linear_time:.6f}s")
    print(f"Binary search: found at index {binary_result} in {binary_time:.6f}s")
    print(f"Binary search is {linear_time/binary_time:.1f}x faster")

def memory_optimization_demo():
    """Demonstrate memory optimization techniques"""
    print("\n=== Memory Optimization Demo ===")
    
    # Generator vs list
    def list_squares(n):
        """Memory-intensive: creates full list"""
        return [i**2 for i in range(n)]
    
    def generator_squares(n):
        """Memory-efficient: yields one at a time"""
        for i in range(n):
            yield i**2
    
    # Compare memory usage (conceptually)
    n = 1000000
    
    print("List approach:")
    start_time = time.time()
    squares_list = list_squares(n)
    list_time = time.time() - start_time
    print(f"  Created list of {len(squares_list)} items in {list_time:.4f}s")
    
    print("Generator approach:")
    start_time = time.time()
    squares_gen = generator_squares(n)
    # Process first 10 items
    first_10 = [next(squares_gen) for _ in range(10)]
    gen_time = time.time() - start_time
    print(f"  Generated first 10 items: {first_10} in {gen_time:.6f}s")
    
    # Object pooling example
    class ObjectPool:
        """Simple object pool for reusing expensive objects"""
        
        def __init__(self, create_func, max_size=10):
            self.create_func = create_func
            self.pool = []
            self.max_size = max_size
        
        def get_object(self):
            if self.pool:
                return self.pool.pop()
            return self.create_func()
        
        def return_object(self, obj):
            if len(self.pool) < self.max_size:
                # Reset object state if needed
                self.pool.append(obj)
    
    # Demo object pool
    def create_expensive_object():
        """Simulate expensive object creation"""
        time.sleep(0.001)  # Simulate work
        return {"data": [0] * 1000}
    
    pool = ObjectPool(create_expensive_object, max_size=5)
    
    print("Object pooling demo:")
    start_time = time.time()
    
    # Use objects from pool
    objects = []
    for i in range(10):
        obj = pool.get_object()
        objects.append(obj)
    
    # Return objects to pool
    for obj in objects:
        pool.return_object(obj)
    
    pool_time = time.time() - start_time
    print(f"  Used object pool for 10 objects in {pool_time:.4f}s")

def caching_demo():
    """Demonstrate caching strategies"""
    print("\n=== Caching Demo ===")
    
    # Without caching
    def fibonacci_slow(n):
        """Slow recursive fibonacci"""
        if n <= 1:
            return n
        return fibonacci_slow(n-1) + fibonacci_slow(n-2)
    
    # With LRU cache
    @lru_cache(maxsize=128)
    def fibonacci_fast(n):
        """Fast cached fibonacci"""
        if n <= 1:
            return n
        return fibonacci_fast(n-1) + fibonacci_fast(n-2)
    
    # Compare performance
    n = 30
    
    print(f"Computing fibonacci({n}):")
    
    # Time slow version
    start_time = time.time()
    slow_result = fibonacci_slow(n)
    slow_time = time.time() - start_time
    
    # Time fast version
    start_time = time.time()
    fast_result = fibonacci_fast(n)
    fast_time = time.time() - start_time
    
    print(f"  Without cache: {slow_result} in {slow_time:.4f}s")
    print(f"  With cache: {fast_result} in {fast_time:.6f}s")
    print(f"  Speedup: {slow_time/fast_time:.1f}x")
    
    # Show cache info
    print(f"  Cache info: {fibonacci_fast.cache_info()}")

def parallel_processing_demo():
    """Demonstrate parallel processing"""
    print("\n=== Parallel Processing Demo ===")
    
    def cpu_bound_task(n):
        """CPU-intensive task"""
        total = 0
        for i in range(n):
            total += i ** 2
        return total
    
    def io_bound_task(duration):
        """I/O-intensive task (simulated)"""
        time.sleep(duration)
        return f"Task completed after {duration}s"
    
    # CPU-bound: multiprocessing
    print("CPU-bound tasks (multiprocessing):")
    tasks = [100000] * 4
    
    # Sequential
    start_time = time.time()
    sequential_results = [cpu_bound_task(task) for task in tasks]
    sequential_time = time.time() - start_time
    
    # Parallel
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=4) as executor:
        parallel_results = list(executor.map(cpu_bound_task, tasks))
    parallel_time = time.time() - start_time
    
    print(f"  Sequential: {sequential_time:.4f}s")
    print(f"  Parallel: {parallel_time:.4f}s")
    print(f"  Speedup: {sequential_time/parallel_time:.1f}x")
    
    # I/O-bound: threading
    print("\nI/O-bound tasks (threading):")
    io_tasks = [0.1] * 4
    
    # Sequential
    start_time = time.time()
    sequential_io = [io_bound_task(task) for task in io_tasks]
    sequential_io_time = time.time() - start_time
    
    # Parallel with threads
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        parallel_io = list(executor.map(io_bound_task, io_tasks))
    parallel_io_time = time.time() - start_time
    
    print(f"  Sequential: {sequential_io_time:.4f}s")
    print(f"  Parallel: {parallel_io_time:.4f}s")
    print(f"  Speedup: {sequential_io_time/parallel_io_time:.1f}x")

def vectorization_demo():
    """Demonstrate vectorization with NumPy"""
    print("\n=== Vectorization Demo ===")
    
    # Pure Python approach
    def python_sum_squares(data):
        """Pure Python sum of squares"""
        return sum(x**2 for x in data)
    
    # NumPy vectorized approach
    def numpy_sum_squares(data):
        """NumPy vectorized sum of squares"""
        arr = np.array(data)
        return np.sum(arr**2)
    
    # Test with large dataset
    data = list(range(1000000))
    
    # Time Python version
    start_time = time.time()
    python_result = python_sum_squares(data)
    python_time = time.time() - start_time
    
    # Time NumPy version
    start_time = time.time()
    numpy_result = numpy_sum_squares(data)
    numpy_time = time.time() - start_time
    
    print(f"Python sum of squares: {python_result} in {python_time:.4f}s")
    print(f"NumPy sum of squares: {numpy_result} in {numpy_time:.4f}s")
    print(f"NumPy is {python_time/numpy_time:.1f}x faster")

class PerformanceMonitor:
    """Simple performance monitoring decorator"""
    
    def __init__(self):
        self.stats = {}
    
    def monitor(self, func):
        """Decorator to monitor function performance"""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            try:
                result = func(*args, **kwargs)
                success = True
            except Exception as e:
                result = None
                success = False
                raise
            finally:
                end_time = time.time()
                end_memory = self._get_memory_usage()
                
                # Record stats
                func_name = func.__name__
                if func_name not in self.stats:
                    self.stats[func_name] = {
                        'calls': 0,
                        'total_time': 0,
                        'total_memory': 0,
                        'errors': 0
                    }
                
                self.stats[func_name]['calls'] += 1
                self.stats[func_name]['total_time'] += (end_time - start_time)
                self.stats[func_name]['total_memory'] += (end_memory - start_memory)
                
                if not success:
                    self.stats[func_name]['errors'] += 1
            
            return result
        
        return wrapper
    
    def _get_memory_usage(self):
        """Get current memory usage (simplified)"""
        import psutil
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except:
            return 0
    
    def get_stats(self):
        """Get performance statistics"""
        return self.stats
    
    def print_stats(self):
        """Print performance statistics"""
        print("\nPerformance Statistics:")
        print("-" * 50)
        
        for func_name, stats in self.stats.items():
            calls = stats['calls']
            avg_time = stats['total_time'] / calls if calls > 0 else 0
            avg_memory = stats['total_memory'] / calls if calls > 0 else 0
            error_rate = stats['errors'] / calls if calls > 0 else 0
            
            print(f"{func_name}:")
            print(f"  Calls: {calls}")
            print(f"  Avg time: {avg_time:.4f}s")
            print(f"  Avg memory: {avg_memory:.2f}MB")
            print(f"  Error rate: {error_rate:.2%}")

def performance_monitoring_demo():
    """Demonstrate performance monitoring"""
    print("\n=== Performance Monitoring Demo ===")
    
    monitor = PerformanceMonitor()
    
    @monitor.monitor
    def test_function_1():
        time.sleep(0.1)
        return sum(range(10000))
    
    @monitor.monitor
    def test_function_2():
        time.sleep(0.05)
        return [i**2 for i in range(5000)]
    
    # Run functions multiple times
    for _ in range(3):
        test_function_1()
        test_function_2()
    
    # Print statistics
    monitor.print_stats()

if __name__ == "__main__":
    profiling_demo()
    algorithm_optimization_demo()
    memory_optimization_demo()
    caching_demo()
    parallel_processing_demo()
    vectorization_demo()
    performance_monitoring_demo()
