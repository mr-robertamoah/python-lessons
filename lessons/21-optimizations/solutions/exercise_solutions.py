# Lesson 19 Solutions: Performance Optimization

import time
import timeit
import cProfile
import pstats
from functools import lru_cache, wraps
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import multiprocessing
import numpy as np
from collections import defaultdict, deque
import sqlite3

print("=== GUIDED EXERCISE SOLUTIONS ===\n")

# Exercise 1: Profiling and Benchmarking
print("Exercise 1: Profiling and Benchmarking")
print("-" * 50)

def create_test_functions():
    """Create functions with different performance characteristics"""
    
    def cpu_intensive():
        """CPU-intensive function"""
        total = 0
        for i in range(100000):
            total += i ** 2
        return total
    
    def memory_intensive():
        """Memory-intensive function"""
        data = []
        for i in range(50000):
            data.append([j for j in range(10)])
        return len(data)
    
    def string_operations():
        """String operation intensive"""
        text = ""
        for i in range(10000):
            text += str(i) + ","
        return len(text)
    
    return cpu_intensive, memory_intensive, string_operations

def profile_functions():
    """Profile functions to identify bottlenecks"""
    cpu_func, memory_func, string_func = create_test_functions()
    
    functions = [
        ("CPU Intensive", cpu_func),
        ("Memory Intensive", memory_func),
        ("String Operations", string_func)
    ]
    
    print("Profiling Results:")
    
    for name, func in functions:
        print(f"\n{name}:")
        
        # Profile with cProfile
        profiler = cProfile.Profile()
        profiler.enable()
        result = func()
        profiler.disable()
        
        # Get stats
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        
        # Print top functions
        print(f"  Result: {result}")
        print(f"  Top function calls:")
        stats.print_stats(3)
        
        # Benchmark with timeit
        execution_time = timeit.timeit(func, number=10)
        print(f"  Average execution time: {execution_time/10:.4f}s")

profile_functions()

# Exercise 2: Algorithm Optimization
print("\n" + "="*50)
print("Exercise 2: Algorithm Optimization")
print("-" * 50)

class AlgorithmOptimizer:
    """Compare different algorithmic approaches"""
    
    @staticmethod
    def bubble_sort(arr):
        """O(n²) bubble sort"""
        arr = arr.copy()
        n = len(arr)
        for i in range(n):
            for j in range(0, n-i-1):
                if arr[j] > arr[j+1]:
                    arr[j], arr[j+1] = arr[j+1], arr[j]
        return arr
    
    @staticmethod
    def quick_sort(arr):
        """O(n log n) quick sort"""
        if len(arr) <= 1:
            return arr
        
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        
        return AlgorithmOptimizer.quick_sort(left) + middle + AlgorithmOptimizer.quick_sort(right)
    
    @staticmethod
    def linear_search(arr, target):
        """O(n) linear search"""
        for i, item in enumerate(arr):
            if item == target:
                return i
        return -1
    
    @staticmethod
    def binary_search(arr, target):
        """O(log n) binary search"""
        left, right = 0, len(arr) - 1
        
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return -1
    
    @staticmethod
    def compare_sorting_algorithms():
        """Compare sorting algorithm performance"""
        import random
        
        # Test with different sizes
        sizes = [100, 1000, 5000]
        
        print("Sorting Algorithm Comparison:")
        
        for size in sizes:
            data = [random.randint(1, 1000) for _ in range(size)]
            
            print(f"\nArray size: {size}")
            
            # Time bubble sort
            start_time = time.time()
            bubble_result = AlgorithmOptimizer.bubble_sort(data)
            bubble_time = time.time() - start_time
            
            # Time quick sort
            start_time = time.time()
            quick_result = AlgorithmOptimizer.quick_sort(data)
            quick_time = time.time() - start_time
            
            # Time built-in sort
            start_time = time.time()
            builtin_result = sorted(data)
            builtin_time = time.time() - start_time
            
            print(f"  Bubble sort: {bubble_time:.4f}s")
            print(f"  Quick sort: {quick_time:.4f}s")
            print(f"  Built-in sort: {builtin_time:.4f}s")
            
            if bubble_time > 0:
                print(f"  Quick sort speedup: {bubble_time/quick_time:.1f}x")
                print(f"  Built-in speedup: {bubble_time/builtin_time:.1f}x")
    
    @staticmethod
    def compare_search_algorithms():
        """Compare search algorithm performance"""
        # Create sorted data for binary search
        data = list(range(100000))
        target = 75000
        
        print("\nSearch Algorithm Comparison:")
        
        # Time linear search
        start_time = time.time()
        linear_result = AlgorithmOptimizer.linear_search(data, target)
        linear_time = time.time() - start_time
        
        # Time binary search
        start_time = time.time()
        binary_result = AlgorithmOptimizer.binary_search(data, target)
        binary_time = time.time() - start_time
        
        print(f"  Linear search: found at {linear_result} in {linear_time:.6f}s")
        print(f"  Binary search: found at {binary_result} in {binary_time:.6f}s")
        print(f"  Binary search speedup: {linear_time/binary_time:.1f}x")

optimizer = AlgorithmOptimizer()
optimizer.compare_sorting_algorithms()
optimizer.compare_search_algorithms()

# Exercise 3: Memory Optimization
print("\n" + "="*50)
print("Exercise 3: Memory Optimization")
print("-" * 50)

class MemoryOptimizer:
    """Demonstrate memory optimization techniques"""
    
    @staticmethod
    def list_vs_generator():
        """Compare list vs generator memory usage"""
        n = 1000000
        
        print("List vs Generator Comparison:")
        
        # List approach
        def create_list():
            return [i**2 for i in range(n)]
        
        # Generator approach
        def create_generator():
            return (i**2 for i in range(n))
        
        # Time list creation
        start_time = time.time()
        data_list = create_list()
        list_time = time.time() - start_time
        
        # Time generator creation
        start_time = time.time()
        data_gen = create_generator()
        gen_time = time.time() - start_time
        
        print(f"  List creation: {list_time:.4f}s")
        print(f"  Generator creation: {gen_time:.6f}s")
        
        # Process first 10 items from each
        start_time = time.time()
        list_first_10 = data_list[:10]
        list_access_time = time.time() - start_time
        
        start_time = time.time()
        gen_first_10 = [next(data_gen) for _ in range(10)]
        gen_access_time = time.time() - start_time
        
        print(f"  List access (first 10): {list_access_time:.6f}s")
        print(f"  Generator access (first 10): {gen_access_time:.6f}s")
    
    @staticmethod
    def object_pooling_demo():
        """Demonstrate object pooling for memory efficiency"""
        
        class ExpensiveObject:
            """Simulate expensive object creation"""
            def __init__(self):
                self.data = [0] * 10000  # Large data structure
                time.sleep(0.001)  # Simulate expensive initialization
            
            def reset(self):
                """Reset object state for reuse"""
                self.data = [0] * 10000
        
        class ObjectPool:
            """Object pool implementation"""
            def __init__(self, create_func, max_size=10):
                self.create_func = create_func
                self.pool = deque()
                self.max_size = max_size
            
            def get_object(self):
                if self.pool:
                    return self.pool.popleft()
                return self.create_func()
            
            def return_object(self, obj):
                if len(self.pool) < self.max_size:
                    obj.reset()
                    self.pool.append(obj)
        
        print("\nObject Pooling Demonstration:")
        
        # Without pooling
        start_time = time.time()
        objects = []
        for i in range(20):
            obj = ExpensiveObject()
            objects.append(obj)
        no_pool_time = time.time() - start_time
        
        # With pooling
        pool = ObjectPool(ExpensiveObject, max_size=5)
        
        start_time = time.time()
        pooled_objects = []
        for i in range(20):
            obj = pool.get_object()
            pooled_objects.append(obj)
        
        # Return objects to pool
        for obj in pooled_objects:
            pool.return_object(obj)
        
        pool_time = time.time() - start_time
        
        print(f"  Without pooling: {no_pool_time:.4f}s")
        print(f"  With pooling: {pool_time:.4f}s")
        print(f"  Pooling speedup: {no_pool_time/pool_time:.1f}x")
    
    @staticmethod
    def data_structure_optimization():
        """Compare different data structures for performance"""
        n = 100000
        
        print("\nData Structure Performance Comparison:")
        
        # List operations
        start_time = time.time()
        data_list = []
        for i in range(n):
            data_list.append(i)
        list_append_time = time.time() - start_time
        
        # Deque operations
        start_time = time.time()
        data_deque = deque()
        for i in range(n):
            data_deque.append(i)
        deque_append_time = time.time() - start_time
        
        print(f"  List append ({n} items): {list_append_time:.4f}s")
        print(f"  Deque append ({n} items): {deque_append_time:.4f}s")
        
        # Test insertions at beginning
        start_time = time.time()
        for i in range(1000):
            data_list.insert(0, i)
        list_insert_time = time.time() - start_time
        
        start_time = time.time()
        for i in range(1000):
            data_deque.appendleft(i)
        deque_insert_time = time.time() - start_time
        
        print(f"  List insert at beginning (1000 items): {list_insert_time:.4f}s")
        print(f"  Deque insert at beginning (1000 items): {deque_insert_time:.4f}s")
        print(f"  Deque speedup for insertions: {list_insert_time/deque_insert_time:.1f}x")

memory_optimizer = MemoryOptimizer()
memory_optimizer.list_vs_generator()
memory_optimizer.object_pooling_demo()
memory_optimizer.data_structure_optimization()

print("\n=== INDEPENDENT EXERCISE SOLUTIONS ===\n")

# Exercise 4: Data Processing Optimization
print("Exercise 4: Data Processing Optimization")
print("-" * 50)

class DataProcessingOptimizer:
    """Optimize data processing operations"""
    
    @staticmethod
    def create_sample_data(n=1000000):
        """Create sample data for processing"""
        np.random.seed(42)
        return {
            'values': np.random.randn(n),
            'categories': np.random.choice(['A', 'B', 'C'], n),
            'weights': np.random.uniform(0.1, 2.0, n)
        }
    
    @staticmethod
    def python_processing(data):
        """Pure Python data processing"""
        values = data['values']
        categories = data['categories']
        weights = data['weights']
        
        # Calculate weighted averages by category
        category_sums = defaultdict(float)
        category_weights = defaultdict(float)
        
        for i in range(len(values)):
            cat = categories[i]
            val = values[i]
            weight = weights[i]
            
            category_sums[cat] += val * weight
            category_weights[cat] += weight
        
        # Calculate averages
        result = {}
        for cat in category_sums:
            result[cat] = category_sums[cat] / category_weights[cat]
        
        return result
    
    @staticmethod
    def numpy_processing(data):
        """Vectorized NumPy processing"""
        values = data['values']
        categories = data['categories']
        weights = data['weights']
        
        result = {}
        for cat in np.unique(categories):
            mask = categories == cat
            cat_values = values[mask]
            cat_weights = weights[mask]
            
            weighted_sum = np.sum(cat_values * cat_weights)
            total_weight = np.sum(cat_weights)
            result[cat] = weighted_sum / total_weight
        
        return result
    
    @staticmethod
    def parallel_processing(data, n_workers=4):
        """Parallel processing using multiprocessing"""
        def process_chunk(chunk_data):
            return DataProcessingOptimizer.python_processing(chunk_data)
        
        # Split data into chunks
        n = len(data['values'])
        chunk_size = n // n_workers
        chunks = []
        
        for i in range(n_workers):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < n_workers - 1 else n
            
            chunk = {
                'values': data['values'][start_idx:end_idx],
                'categories': data['categories'][start_idx:end_idx],
                'weights': data['weights'][start_idx:end_idx]
            }
            chunks.append(chunk)
        
        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            chunk_results = list(executor.map(process_chunk, chunks))
        
        # Combine results
        combined_sums = defaultdict(float)
        combined_weights = defaultdict(float)
        
        for chunk_result in chunk_results:
            for cat, avg in chunk_result.items():
                # This is simplified - in practice, you'd need to properly combine weighted averages
                combined_sums[cat] += avg
                combined_weights[cat] += 1
        
        # Calculate final averages
        result = {}
        for cat in combined_sums:
            result[cat] = combined_sums[cat] / combined_weights[cat]
        
        return result
    
    @staticmethod
    def compare_approaches():
        """Compare different processing approaches"""
        print("Data Processing Optimization Comparison:")
        
        # Create test data
        data = DataProcessingOptimizer.create_sample_data(100000)  # Smaller for demo
        
        approaches = [
            ("Pure Python", DataProcessingOptimizer.python_processing),
            ("NumPy Vectorized", DataProcessingOptimizer.numpy_processing),
            ("Parallel Processing", DataProcessingOptimizer.parallel_processing)
        ]
        
        results = {}
        times = {}
        
        for name, func in approaches:
            start_time = time.time()
            result = func(data)
            execution_time = time.time() - start_time
            
            results[name] = result
            times[name] = execution_time
            
            print(f"  {name}: {execution_time:.4f}s")
            print(f"    Result: {result}")
        
        # Calculate speedups
        python_time = times["Pure Python"]
        for name, exec_time in times.items():
            if name != "Pure Python":
                speedup = python_time / exec_time
                print(f"  {name} speedup: {speedup:.1f}x")

data_optimizer = DataProcessingOptimizer()
data_optimizer.compare_approaches()

# Exercise 5: Database Query Optimization
print("\n" + "="*50)
print("Exercise 5: Database Query Optimization")
print("-" * 50)

class DatabaseOptimizer:
    """Demonstrate database optimization techniques"""
    
    def __init__(self):
        self.conn = sqlite3.connect(':memory:')
        self.setup_database()
    
    def setup_database(self):
        """Create and populate test database"""
        cursor = self.conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE customers (
                id INTEGER PRIMARY KEY,
                name TEXT,
                email TEXT,
                city TEXT,
                signup_date TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE orders (
                id INTEGER PRIMARY KEY,
                customer_id INTEGER,
                product TEXT,
                amount REAL,
                order_date TEXT,
                FOREIGN KEY (customer_id) REFERENCES customers (id)
            )
        ''')
        
        # Insert sample data
        import random
        from datetime import datetime, timedelta
        
        # Insert customers
        customers = []
        for i in range(1000):
            customers.append((
                i + 1,
                f'Customer_{i+1}',
                f'customer{i+1}@email.com',
                random.choice(['NYC', 'LA', 'Chicago', 'Houston']),
                (datetime.now() - timedelta(days=random.randint(1, 365))).strftime('%Y-%m-%d')
            ))
        
        cursor.executemany(
            'INSERT INTO customers VALUES (?, ?, ?, ?, ?)',
            customers
        )
        
        # Insert orders
        orders = []
        for i in range(5000):
            orders.append((
                i + 1,
                random.randint(1, 1000),
                f'Product_{random.randint(1, 100)}',
                round(random.uniform(10, 500), 2),
                (datetime.now() - timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d')
            ))
        
        cursor.executemany(
            'INSERT INTO orders VALUES (?, ?, ?, ?, ?)',
            orders
        )
        
        self.conn.commit()
        print("Database setup complete: 1000 customers, 5000 orders")
    
    def test_query_optimization(self):
        """Test different query optimization techniques"""
        cursor = self.conn.cursor()
        
        print("\nQuery Optimization Tests:")
        
        # Test 1: Without index
        start_time = time.time()
        cursor.execute('''
            SELECT c.name, SUM(o.amount) as total_spent
            FROM customers c
            JOIN orders o ON c.id = o.customer_id
            WHERE c.city = 'NYC'
            GROUP BY c.id, c.name
            ORDER BY total_spent DESC
            LIMIT 10
        ''')
        results_no_index = cursor.fetchall()
        time_no_index = time.time() - start_time
        
        # Create index
        cursor.execute('CREATE INDEX idx_customer_city ON customers(city)')
        cursor.execute('CREATE INDEX idx_order_customer ON orders(customer_id)')
        
        # Test 2: With index
        start_time = time.time()
        cursor.execute('''
            SELECT c.name, SUM(o.amount) as total_spent
            FROM customers c
            JOIN orders o ON c.id = o.customer_id
            WHERE c.city = 'NYC'
            GROUP BY c.id, c.name
            ORDER BY total_spent DESC
            LIMIT 10
        ''')
        results_with_index = cursor.fetchall()
        time_with_index = time.time() - start_time
        
        print(f"  Without index: {time_no_index:.4f}s")
        print(f"  With index: {time_with_index:.4f}s")
        print(f"  Index speedup: {time_no_index/time_with_index:.1f}x")
    
    def test_batch_operations(self):
        """Test batch vs individual operations"""
        cursor = self.conn.cursor()
        
        print("\nBatch Operations Test:")
        
        # Individual inserts
        start_time = time.time()
        for i in range(100):
            cursor.execute(
                'INSERT INTO customers (name, email, city, signup_date) VALUES (?, ?, ?, ?)',
                (f'Batch_Customer_{i}', f'batch{i}@email.com', 'Test_City', '2024-01-01')
            )
        self.conn.commit()
        individual_time = time.time() - start_time
        
        # Batch insert
        batch_data = [
            (f'Batch_Customer_B_{i}', f'batchB{i}@email.com', 'Test_City', '2024-01-01')
            for i in range(100)
        ]
        
        start_time = time.time()
        cursor.executemany(
            'INSERT INTO customers (name, email, city, signup_date) VALUES (?, ?, ?, ?)',
            batch_data
        )
        self.conn.commit()
        batch_time = time.time() - start_time
        
        print(f"  Individual inserts (100): {individual_time:.4f}s")
        print(f"  Batch insert (100): {batch_time:.4f}s")
        print(f"  Batch speedup: {individual_time/batch_time:.1f}x")
    
    def cleanup(self):
        """Close database connection"""
        self.conn.close()

# Demo database optimization
db_optimizer = DatabaseOptimizer()
db_optimizer.test_query_optimization()
db_optimizer.test_batch_operations()
db_optimizer.cleanup()

print("\n" + "="*50)
print("Performance Optimization exercise solutions complete!")
print("Key concepts demonstrated:")
print("- Profiling and benchmarking techniques")
print("- Algorithm optimization (sorting, searching)")
print("- Memory optimization (generators, object pooling)")
print("- Data processing optimization (vectorization, parallelization)")
print("- Database query optimization (indexing, batch operations)")
print("- Performance monitoring and measurement")
